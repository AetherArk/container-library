"""Small sequential lifecycle; imports no torch/chai_lab until work is pending."""
import argparse
import gc
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import uuid

from .benchmark import Benchmark, SCHEMA, load_metadata, error_label, write_failure
from .errors import is_cuda_oom, release_exception

# Mirror of the image label ai.aetherark.cofold.image_id, written by the Dockerfile
# from CONTAINER_IMAGE_ID so the worker can read its own image identity.
IMAGE_LABEL = Path('/opt/cofold/image-label.json')


@dataclass(frozen=True)
class Settings:
    samples: int = 1
    recycles: int = 3
    steps: int = 200
    low_memory: bool = True
    max_parallel_samples: int = 1
    output_format: str = 'mmcif'
    derived: str = 'tensors'


@dataclass
class Stop:
    requested: bool = False


class JobTimeout(RuntimeError):
    """One input x seed job exceeded --job-timeout at a checked boundary."""
    error_kind = 'timeout'


def job_deadline(begin, limit, boundary):
    """Record a job timeout at a boundary the caller chose.

    Guarantee: elapsed wall time is compared around the native call and at job and
    sample boundaries, never with SIGALRM, so no signal can fire inside CUDA code.
    A job that overruns is reported with error_kind='timeout' at the first such
    boundary after the overrun; there is no in-kernel cancellation, so the worker
    cannot interrupt a single stuck native call, only stop between calls.
    """
    if limit is not None and time.perf_counter() - begin > limit:
        raise JobTimeout('job exceeded --job-timeout of %g seconds at %s' % (limit, boundary))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(obj, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def input_identity(source):
    """Bind source bytes and referenced local native MSA/template bytes."""
    result = {'path': str(source.resolve()), 'sha256': sha(source), 'references': {}}
    try:
        import yaml
        data = yaml.safe_load(source.read_text())
    except (ImportError, ValueError):
        return result
    except Exception:
        # Invalid YAML is an isolated preparation failure, not a global stop.
        return result

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ('msa', 'cif', 'pdb', 'path') and isinstance(child, str) and child not in ('empty', 'auto'):
                    p = Path(child)
                    if not p.is_absolute():
                        p = source.parent / p
                    result['references'][str(p.resolve())] = sha(p) if p.is_file() else None
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(data)
    return result


def inventory(attempt):
    return {str(p.relative_to(attempt)): sha(p) for p in sorted(attempt.rglob('*'))
            if p.is_file() and p.name != 'receipt.json'}


def image_identity():
    """Identify the installed image: baked label first, launcher env second.

    A direct `docker run` (no launcher env, no label) records an explicit null and
    the reason instead of a bare null.
    """
    label = None
    try:
        data = json.loads(IMAGE_LABEL.read_text())
        candidate = data.get('image_id') if isinstance(data, dict) else None
        if isinstance(candidate, str) and candidate and candidate != 'unknown':
            label = candidate
    except (OSError, ValueError):
        pass
    if label is not None:
        return {'image_id': label, 'image_id_source': 'image_label'}
    env = os.environ.get('COFOLD_IMAGE_ID')
    if isinstance(env, str) and env:
        return {'image_id': env, 'image_id_source': 'env'}
    return {'image_id': None, 'image_id_source': None,
            'image_id_reason': 'no image label and COFOLD_IMAGE_ID unset (direct docker run)'}


def completed(output, key, identity):
    """Reuse requires every recorded path to still hash-match; extra files are tolerated."""
    for receipt in sorted((output / 'jobs' / key).glob('*/receipt.json'), reverse=True):
        try:
            saved = json.loads(receipt.read_text())
            if saved['status'] != 'completed' or saved['identity'] != identity:
                continue
            recorded = saved['sha256'] or {}
            if not recorded:
                continue
            actual = inventory(receipt.parent)
            if all(actual.get(path) == digest for path, digest in recorded.items()):
                return receipt.parent, len(set(actual) - set(recorded))
        except (OSError, ValueError, KeyError):
            continue
    return None


def run(inputs, output, assets, checkpoint, seeds, settings, resume=False,
        engine_factory=None, stop=None, benchmark_metadata=None, job_timeout=None):
    """Run every input x seed once against one resident model.

    `job_timeout` bounds one input x seed job; an overrun is recorded as a failed
    job with error_kind='timeout' at a sample/job boundary and the remaining jobs
    still run (see job_deadline for what cannot be cancelled).
    """
    inputs, output, checkpoint = map(Path, (inputs, output, checkpoint))
    # The model directory doubles as the asset directory in this image; hash it once.
    assets = Path(assets) if assets else checkpoint
    if not seeds or any(type(s) is not int or s < 0 or s > 4294967295 for s in seeds):
        raise ValueError('seeds must be explicit integers in [0, 4294967295]')
    if any(x < 1 for x in (settings.samples, settings.steps, settings.max_parallel_samples)) or settings.recycles < 0:
        raise ValueError('invalid recycle/sampling settings')
    if settings.max_parallel_samples != 1:
        raise ValueError('Chai-1 folds one sample per native call; --max-parallel-samples must be 1')
    if settings.output_format != 'mmcif':
        raise ValueError('Chai-1 writes mmCIF only; output_format must be mmcif (pdb is refused)')
    if settings.derived not in ('tensors', 'none'):
        raise ValueError("derived must be 'tensors' or 'none'")
    if any(seed * 1000 + settings.samples - 1 > 2147483647 for seed in seeds):
        raise ValueError('seed mapping exceeds the engine signed 32-bit range')
    if job_timeout is not None and not (job_timeout > 0 and job_timeout != float('inf')):
        raise ValueError('job timeout must be a positive finite number of seconds')
    sources = sorted(p for p in inputs.iterdir() if p.suffix.lower() in ('.yaml', '.yml') and p.is_file())
    if not sources:
        raise ValueError('input directory contains no native YAML inputs')
    if output.exists() and any(p.name != '.runtime' for p in output.iterdir()) and not resume:
        raise FileExistsError('output is not empty; use --resume to retain and verify prior attempts')
    output.mkdir(parents=True, exist_ok=True)
    # Reject two simultaneous workers rather than racing receipts/run.json.
    import fcntl
    with (output / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with Benchmark(output, load_metadata(benchmark_metadata), sample_gpu=engine_factory is None) as benchmark:
            return _run(sources, output, assets, checkpoint, seeds, settings, resume,
                        engine_factory, stop or Stop(), benchmark, job_timeout)


def asset_identity(assets):
    """Identify immutable packaged assets once per invocation, never per job."""
    assets = Path(assets)
    archive = assets / 'mols.tar'
    marker = assets / 'mols' / '.complete.json'
    if archive.is_file():
        # The normal installer binds extracted mols/ to this archive. Assets
        # remain immutable; a different package must not reuse earlier jobs.
        return {'kind': 'archive', 'sha256': sha(archive),
                'extraction_marker': sha(marker) if marker.is_file() else None}
    digest = hashlib.sha256()
    for path in sorted(assets.rglob('*')):
        if path.is_file():
            digest.update(str(path.relative_to(assets)).encode())
            digest.update(b'\0')
            digest.update(sha(path).encode())
    return {'kind': 'files', 'sha256': digest.hexdigest()}


def _run(sources, output, assets, checkpoint, seeds, settings, resume, engine_factory, stop, benchmark,
         job_timeout=None):
    from .identity import directory_identity
    with benchmark.stage('identity_verification'):
        checkpoint_identity = (directory_identity(checkpoint) if checkpoint.is_dir()
                               else {'kind': 'file', 'sha256': sha(checkpoint)})
        checkpoint_identity = dict(checkpoint_identity, path=str(checkpoint.resolve()))
        checkpoint_hash = checkpoint_identity['sha256']
        assets_identity = ({'kind': 'checkpoint_directory', 'sha256': checkpoint_hash}
                           if assets.resolve() == checkpoint.resolve() else asset_identity(assets))
    implementation = dict(image_identity(), code={str(p.relative_to(Path(__file__).parent.parent)): sha(p)
                                                  for folder in ('resident',)
                                                  for p in sorted((Path(__file__).parent.parent / folder).glob('*.py'))})
    jobs = []
    for source in sources:
        source_identity = input_identity(source)
        occurrences = {}
        for seed in seeds:
            occurrence = occurrences.get(seed, 0)
            occurrences[seed] = occurrence + 1
            identity = {'input': source_identity, 'seed': seed, 'settings': asdict(settings),
                        'checkpoint': checkpoint_identity, 'assets': assets_identity,
                        'implementation': implementation, 'worker_version': 1}
            # Seed occurrence preserves even duplicate explicitly supplied seeds.
            key = hashlib.sha256(json.dumps([identity, occurrence], sort_keys=True).encode()).hexdigest()
            jobs.append({'input': str(source), 'seed': seed, 'seed_occurrence': occurrence,
                         'job_index': len(jobs), 'status': 'pending', 'output': None,
                         'identity': identity, 'key': key, 'reused': False})
    meta = {'schema': SCHEMA, 'requested_seeds': seeds, 'settings': asdict(settings), 'checkpoint': checkpoint_identity,
            'assets': assets_identity,
            'implementation': implementation, 'checkpoint_load_count': 0,
            'pid': os.getpid(), 'status': 'running', 'jobs': jobs}
    engine = None
    loaded = False
    active = [None]
    counts = {'completed': 0, 'failed': 0, 'skipped': 0}
    emitted = set()
    def record(job):
        if job['key'] in emitted:
            return
        emitted.add(job['key'])
        for sample_index in range(settings.samples):
            benchmark.append('predictions.jsonl', {
                'job_key': job['key'], 'input': job['input'], 'seed': job['seed'],
                'sample_index': sample_index, 'seed_occurrence': job['seed_occurrence'],
                'job_index': job['job_index'], 'status': job['status'], 'reused': job['reused'],
                'verified': job['status'] == 'completed',
                'error_kind': job.get('error_kind'), 'error': job.get('error'),
                'reason': job.get('reason'), 'blocked_by': job.get('blocked_by'),
                'phase': job.get('phase'),
                'model_was_loaded_before_cell': job.get('model_was_loaded_before_cell'),
                'output': job['output'], 'metrics_scope': 'input_seed_batch_not_individual_sample',
                'derived': job.get('derived'),
                'derived_null_reason': job.get('derived_null_reason'),
                'metrics': job.get('benchmark_metrics'),
                'metrics_null_reason': ('reused_not_measured' if job['reused'] else 'native_metrics_unavailable')
                    if not job.get('benchmark_metrics') else None})
    def save(final=False):
        # Keep live updates bounded; rewriting all jobs after every transition
        # would make metadata I/O quadratic for large input directories.
        current = active[0]
        if current and current['status'] in ('completed', 'failed', 'skipped'):
            record(current)
        atomic_json(output / 'progress.json', {
            'schema': SCHEMA, 'updated_at_unix': time.time(), 'pid': os.getpid(),
            'status': meta['status'], 'jobs_total': len(jobs),
            'jobs_completed': counts['completed'], 'jobs_failed': counts['failed'],
            'jobs_skipped': counts['skipped'],
            'checkpoint_load_count': meta['checkpoint_load_count'],
            'current': {k: current[k] for k in ('input', 'seed', 'status')} if current else None})
        if final:
            atomic_json(output / 'run.json', meta)
    save(final=True)

    def block_system(job):
        for other in jobs:
            if other['input'] == job['input'] and other['status'] == 'pending':
                prior = completed(output, other['key'], other['identity']) if resume else None
                if prior is not None:
                    attempt_dir, extras = prior
                    other.update(status='completed', output=str(attempt_dir.relative_to(output)),
                                 reused=True, inventory_extras=extras)
                    counts['completed'] += 1
                else:
                    other.update(status='skipped', error_kind='cuda_out_of_memory',
                                 reason='same_input_settings_blocked_after_cuda_oom', blocked_by=job['key'])
                    counts['skipped'] += 1
                record(other)

    def recovery_valid():
        # Outside exception handlers: failed tensor-bearing traceback refs are gone.
        gc.collect()
        try:
            return (loaded and engine.snapshot() == meta.get('model')
                    and engine.healthy() and engine.snapshot() == meta.get('model'))
        except Exception:
            return False

    try:
        for job in jobs:
            active[0] = job
            if job['status'] != 'pending':
                continue
            if stop.requested:
                break
            with benchmark.stage('resume_verification', job_key=job['key']):
                prior = completed(output, job['key'], job['identity']) if resume else None
            if prior is not None:
                attempt_dir, extras = prior
                job.update(status='completed', output=str(attempt_dir.relative_to(output)),
                           reused=True, inventory_extras=extras)
                counts['completed'] += 1
                save()
                continue
            attempt = output / 'jobs' / job['key'] / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            began = time.perf_counter()
            job.update(output=str(attempt.relative_to(output)), status='preparing', model_was_loaded_before_cell=loaded)
            save()
            prepared = None
            preparation_failed = preparation_oom = False
            try:
                if engine is not None:
                    engine.benchmark_metrics = None
                if engine is None:
                    with benchmark.stage('initialization'):
                        if engine_factory is None:
                            from .engine import Chai1Engine
                            engine_factory = Chai1Engine
                        engine = engine_factory(assets, checkpoint, settings)
                        files = checkpoint_identity.get('files')
                        if isinstance(files, dict):
                            # Reuse the invocation's single hashing pass for per-job locks.
                            engine.checkpoint_files = files
                with benchmark.stage('preparation', job_key=job['key']):
                    prepared = engine.prepare(Path(job['input']), attempt, job['seed'])
                if input_identity(Path(job['input'])) != job['identity']['input']:
                    raise ValueError('input or referenced assets changed during preparation')
                job_deadline(began, job_timeout, 'preparation')
            except Exception as exc:
                job.update(status='failed', phase='preparation', error=error_label(exc))
                preparation_failed = True
                preparation_oom = is_cuda_oom(exc)
                job['error_kind'] = ('timeout' if isinstance(exc, JobTimeout)
                                     else 'cuda_out_of_memory' if preparation_oom else 'native_error')
                if preparation_oom:
                    block_system(job)
                write_failure(attempt, job, exc)
                release_exception(exc)
                counts['failed'] += 1
                job['benchmark_metrics'] = getattr(engine, 'benchmark_metrics', None)
                if job['benchmark_metrics'] is not None:
                    atomic_json(attempt / 'benchmark_metrics.json', job['benchmark_metrics'])
                atomic_json(attempt / 'receipt.json', dict(job, schema=SCHEMA))
                save()
            if preparation_failed:
                prepared = None
                if preparation_oom and not recovery_valid():
                    meta.update(status='fatal', fatal_reason='cuda_oom_recovery_failed_or_model_not_loaded',
                                blocked_by=job['key'])
                    break
                continue
            prediction_failed = False
            try:
                if not loaded:
                    job['phase'] = 'load'
                    job['status'] = 'loading'
                    save()
                    with benchmark.stage('load', job_key=job['key']):
                        engine.load()
                    loaded = True
                    meta['checkpoint_load_count'] += 1
                    meta['model'] = engine.snapshot()
                job.update(status='predicting', model_before=engine.snapshot())
                job['phase'] = 'prediction'
                save()
                hook = job_timeout is not None and hasattr(engine, 'sample_deadline')
                if hook:
                    # Portable between-sample hook; never a signal inside CUDA code.
                    setattr(engine, 'sample_deadline', lambda: job_deadline(began, job_timeout, 'sample'))
                try:
                    with benchmark.stage('prediction', job_key=job['key']):
                        engine.predict(prepared, attempt, job['seed'])
                finally:
                    if hook:
                        setattr(engine, 'sample_deadline', None)
                job_deadline(began, job_timeout, 'prediction')
                with benchmark.stage('verification', job_key=job['key']):
                    engine.verify(prepared, attempt)
                job['model_after'] = engine.snapshot()
                if job['model_before'] != job['model_after']:
                    raise RuntimeError('resident model identity/device/storage changed')
                if settings.derived == 'tensors':
                    if hasattr(engine, 'derive'):
                        with benchmark.stage('derived_metrics', job_key=job['key']):
                            derived_info = engine.derive(prepared, attempt)
                        job['derived'] = derived_info.get('path')
                        job['derived_null_reason'] = derived_info.get('null_reason')
                    else:
                        job['derived'] = None
                        job['derived_null_reason'] = (
                            'engine does not implement derived metrics; nothing was written')
                else:
                    job['derived'] = None
                    job['derived_null_reason'] = 'disabled_by_--derived_none'
                job['status'] = 'completed'
                with benchmark.stage('receipt_verification_and_write', job_key=job['key']):
                    atomic_json(attempt / 'receipt.json', {'schema': SCHEMA, 'status': 'completed',
                                                           'identity': job['identity'],
                                                           'derived': job.get('derived'),
                                                           'derived_null_reason': job.get('derived_null_reason'),
                                                           'sha256': inventory(attempt)})
                counts['completed'] += 1
            except Exception as exc:
                prediction_failed = True
                job.update(status='failed', error=error_label(exc),
                           error_kind=('timeout' if isinstance(exc, JobTimeout)
                                       else 'cuda_out_of_memory' if is_cuda_oom(exc) else 'native_error'))
                if job['error_kind'] == 'cuda_out_of_memory':
                    block_system(job)
                write_failure(attempt, job, exc)
                release_exception(exc)
                counts['failed'] += 1
                if loaded:
                    try:
                        job['model_after'] = engine.snapshot()
                    except Exception:
                        job['model_after'] = None
            finally:
                prepared = None
                job['benchmark_metrics'] = getattr(engine, 'benchmark_metrics', None)
                if prediction_failed:
                    if job['benchmark_metrics'] is not None:
                        atomic_json(attempt / 'benchmark_metrics.json', job['benchmark_metrics'])
                    atomic_json(attempt / 'receipt.json', dict(job, schema=SCHEMA))
                benchmark.event('prediction_terminal', job_key=job['key'], status=job['status'])
                if engine is not None and hasattr(engine, 'checkpoint_load_count'):
                    meta['checkpoint_load_count'] = engine.checkpoint_load_count
                save()
            if prediction_failed and not recovery_valid():
                meta.update(status='fatal', fatal_reason='recovery_failed_or_model_identity_changed',
                            blocked_by=job['key'])
                break
        if meta['status'] != 'fatal':
            meta['status'] = 'stopped' if stop.requested else ('completed' if all(j['status'] == 'completed' for j in jobs) else 'failed')
    except BaseException as exc:
        meta.update(status='fatal', error=error_label(exc))
        raise
    finally:
        if meta['status'] == 'fatal':
            for job in jobs:
                if job['status'] == 'pending':
                    job.update(reason='resident_context_unavailable', blocked_by=meta.get('blocked_by'))
        save(final=True)
        benchmark.summary.update(status=meta['status'], checkpoint_load_count=meta['checkpoint_load_count'],
                                 jobs_total=len(jobs), jobs_completed=counts['completed'], jobs_failed=counts['failed'],
                                 jobs_skipped=counts['skipped'],
                                 requested_predictions=len(jobs) * settings.samples,
                                 verified_predictions=counts['completed'] * settings.samples,
                                 requested_seeds=seeds, settings=asdict(settings),
                                 checkpoint=checkpoint_identity, assets=assets_identity, implementation=implementation)
        for job in jobs:
            record(job)
    return 143 if meta['status'] == 'stopped' else (0 if meta['status'] == 'completed' else 1)


def parser():
    """The installed worker CLI; shared flags are checked against cofolding/cli-contract.json."""
    p = argparse.ArgumentParser(description='One resident Chai-1 model for native YAML inputs × explicit seeds')
    for name in ('input', 'output', 'assets', 'checkpoint'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--samples', type=int, default=1)
    p.add_argument('--recycles', dest='recycles', type=int, default=3,
                   help='Chai-1 num_trunk_recycles')
    p.add_argument('--loops', dest='recycles', type=int, help='alias for --recycles')
    p.add_argument('--steps', type=int, default=200, help='Chai-1 num_diffn_timesteps')
    p.add_argument('--no-low-memory', dest='low_memory', action='store_false',
                   help='disable Chai-1 low_memory (default: on, as in the released pipeline)')
    p.add_argument('--max-parallel-samples', type=int, default=1)
    p.add_argument('--output-format', choices=('mmcif',), default='mmcif',
                   help='Chai-1 writes mmCIF only; pdb is refused rather than ignored')
    p.add_argument('--derived', choices=('none', 'tensors'), default='tensors',
                   help="'tensors' (default) computes ipSAE-family interface scores from each "
                        "sample's PAE and writes metrics/derived.json; 'none' writes no derived "
                        "metrics and records why")
    p.add_argument('--resume', action='store_true')
    p.add_argument('--benchmark-metadata', type=Path, help='optional nonsecret allowlisted caller JSON')
    p.add_argument('--job-timeout', type=float,
                   help='wall seconds one input x seed job may take; an overrun is recorded as '
                        "error_kind='timeout' at the next sample/job boundary and later jobs still run")
    return p


def main():
    parser_instance = parser()
    args = parser_instance.parse_args()
    if args.job_timeout is not None and not (math.isfinite(args.job_timeout) and args.job_timeout > 0):
        parser_instance.error('--job-timeout must be a positive finite number of seconds')
    stop = Stop()
    previous = signal.signal(signal.SIGTERM, lambda *_: setattr(stop, 'requested', True))
    try:
        if args.max_parallel_samples != 1:
            parser_instance.error('Chai-1 folds one sample per native call; '
                                  '--max-parallel-samples must be 1')
        settings = Settings(**{key: getattr(args, key) for key in Settings.__dataclass_fields__})
        return run(args.input, args.output, args.assets, args.checkpoint, args.seeds, settings, args.resume,
                   stop=stop, benchmark_metadata=args.benchmark_metadata, job_timeout=args.job_timeout)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())

"""Small sequential lifecycle; imports no torch/Boltz until work is pending."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import uuid

from .benchmark import Benchmark, load_metadata, error_label


@dataclass(frozen=True)
class Settings:
    samples: int = 1
    recycles: int = 3
    steps: int = 200
    max_parallel_samples: int = 1
    no_kernels: bool = False
    use_potentials: bool = False
    output_format: str = 'mmcif'


@dataclass
class Stop:
    requested: bool = False


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


def completed(output, key, identity):
    for receipt in sorted((output / 'jobs' / key).glob('*/receipt.json'), reverse=True):
        try:
            saved = json.loads(receipt.read_text())
            if saved['status'] != 'completed' or saved['identity'] != identity:
                continue
            if saved['sha256'] and inventory(receipt.parent) == saved['sha256']:
                return receipt.parent
        except (OSError, ValueError, KeyError):
            continue
    return None


def run(inputs, output, assets, checkpoint, seeds, settings, resume=False,
        engine_factory=None, stop=None, benchmark_metadata=None):
    inputs, output, assets, checkpoint = map(Path, (inputs, output, assets, checkpoint))
    if not seeds or any(type(s) is not int or s < 0 or s > 4294967295 for s in seeds):
        raise ValueError('seeds must be explicit integers in [0, 4294967295]')
    if any(x < 1 for x in (settings.samples, settings.steps, settings.max_parallel_samples)) or settings.recycles < 0:
        raise ValueError('invalid sampling/recycling settings')
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
                        engine_factory, stop or Stop(), benchmark)


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


def _run(sources, output, assets, checkpoint, seeds, settings, resume, engine_factory, stop, benchmark):
    with benchmark.stage('identity_verification'):
        checkpoint_hash = sha(checkpoint)
        assets_identity = asset_identity(assets)
    checkpoint_identity = {'path': str(checkpoint.resolve()), 'sha256': checkpoint_hash}
    implementation = {'image_id': os.environ.get('COFOLD_IMAGE_ID'),
                      'code': {str(p.relative_to(Path(__file__).parent.parent)): sha(p)
                               for folder in ('resident', 'batch_native')
                               for p in sorted((Path(__file__).parent.parent / folder).glob('*.py'))}}
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
    meta = {'requested_seeds': seeds, 'settings': asdict(settings), 'checkpoint': checkpoint_identity,
            'assets': assets_identity,
            'implementation': implementation, 'checkpoint_load_count': 0,
            'pid': os.getpid(), 'status': 'running', 'jobs': jobs}
    engine = None
    loaded = False
    active = [None]
    counts = {'completed': 0, 'failed': 0}
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
                'model_was_loaded_before_cell': job.get('model_was_loaded_before_cell'),
                'output': job['output'], 'metrics_scope': 'input_seed_batch_not_individual_sample',
                'metrics': job.get('benchmark_metrics'),
                'metrics_null_reason': ('reused_not_measured' if job['reused'] else 'native_metrics_unavailable')
                    if not job.get('benchmark_metrics') else None})
    def save(final=False):
        # Keep live updates bounded; rewriting all jobs after every transition
        # would make metadata I/O quadratic for large input directories.
        current = active[0]
        if current and current['status'] in ('completed', 'failed'):
            record(current)
        atomic_json(output / 'progress.json', {
            'status': meta['status'], 'jobs_total': len(jobs),
            'jobs_completed': counts['completed'], 'jobs_failed': counts['failed'],
            'checkpoint_load_count': meta['checkpoint_load_count'],
            'current': {k: current[k] for k in ('input', 'seed', 'status')} if current else None})
        if final:
            atomic_json(output / 'run.json', meta)
    save(final=True)
    try:
        for job in jobs:
            active[0] = job
            if stop.requested:
                break
            with benchmark.stage('resume_verification', job_key=job['key']):
                prior = completed(output, job['key'], job['identity']) if resume else None
            if prior is not None:
                job.update(status='completed', output=str(prior.relative_to(output)), reused=True)
                counts['completed'] += 1
                save()
                continue
            attempt = output / 'jobs' / job['key'] / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            job.update(output=str(attempt.relative_to(output)), status='preparing', model_was_loaded_before_cell=loaded)
            save()
            try:
                if engine is not None:
                    engine.benchmark_metrics = None
                if engine is None:
                    with benchmark.stage('initialization'):
                        if engine_factory is None:
                            from .native import NativeEngine
                            engine_factory = NativeEngine
                        engine = engine_factory(assets, checkpoint, settings)
                with benchmark.stage('preparation', job_key=job['key']):
                    prepared = engine.prepare(Path(job['input']), attempt, job['seed'])
                if input_identity(Path(job['input'])) != job['identity']['input']:
                    raise ValueError('input or referenced assets changed during preparation')
            except Exception as exc:
                job.update(status='failed', phase='preparation', error=error_label(exc))
                counts['failed'] += 1
                atomic_json(attempt / 'receipt.json', job)
                save()
                continue
            try:
                if not loaded:
                    job['status'] = 'loading'
                    save()
                    with benchmark.stage('load', job_key=job['key']):
                        engine.load()
                    loaded = True
                    meta['checkpoint_load_count'] += 1
                    meta['model'] = engine.snapshot()
                job.update(status='predicting', model_before=engine.snapshot())
                save()
                with benchmark.stage('prediction', job_key=job['key']):
                    engine.predict(prepared, attempt, job['seed'])
                with benchmark.stage('verification', job_key=job['key']):
                    engine.verify(prepared, attempt)
                job['model_after'] = engine.snapshot()
                if job['model_before'] != job['model_after']:
                    raise RuntimeError('resident model identity/device/storage changed')
                job['status'] = 'completed'
                with benchmark.stage('receipt_verification_and_write', job_key=job['key']):
                    atomic_json(attempt / 'receipt.json', {'status': 'completed', 'identity': job['identity'],
                                                           'sha256': inventory(attempt)})
                counts['completed'] += 1
            except Exception as exc:
                job.update(status='failed', phase='prediction', error=error_label(exc))
                counts['failed'] += 1
                if loaded:
                    job['model_after'] = engine.snapshot()
                atomic_json(attempt / 'receipt.json', job)
                if not loaded or job.get('model_after') != meta.get('model') or not engine.healthy():
                    meta['status'] = 'fatal'
                    break
            finally:
                job['benchmark_metrics'] = getattr(engine, 'benchmark_metrics', None)
                benchmark.event('prediction_terminal', job_key=job['key'], status=job['status'])
                if engine is not None and hasattr(engine, 'checkpoint_load_count'):
                    meta['checkpoint_load_count'] = engine.checkpoint_load_count
                save()
        if meta['status'] != 'fatal':
            meta['status'] = 'stopped' if stop.requested else ('completed' if all(j['status'] == 'completed' for j in jobs) else 'failed')
    except BaseException as exc:
        meta.update(status='fatal', error=error_label(exc))
        raise
    finally:
        save(final=True)
        benchmark.summary.update(status=meta['status'], checkpoint_load_count=meta['checkpoint_load_count'],
                                 jobs_total=len(jobs), jobs_completed=counts['completed'], jobs_failed=counts['failed'],
                                 requested_predictions=len(jobs) * settings.samples,
                                 verified_predictions=counts['completed'] * settings.samples,
                                 requested_seeds=seeds, settings=asdict(settings),
                                 checkpoint=checkpoint_identity, assets=assets_identity, implementation=implementation)
        for job in jobs:
            record(job)
    return 143 if meta['status'] == 'stopped' else (0 if meta['status'] == 'completed' else 1)


def main():
    parser = argparse.ArgumentParser(description='One resident Boltz2 model for native YAML inputs × explicit seeds')
    for name in ('input', 'output', 'assets', 'checkpoint'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--recycles', type=int, default=3)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--max-parallel-samples', type=int, default=1)
    parser.add_argument('--no-kernels', action='store_true')
    parser.add_argument('--use-potentials', action='store_true')
    parser.add_argument('--output-format', choices=('mmcif', 'pdb'), default='mmcif')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--benchmark-metadata', type=Path, help='optional nonsecret allowlisted caller JSON')
    args = parser.parse_args()
    stop = Stop()
    previous = signal.signal(signal.SIGTERM, lambda *_: setattr(stop, 'requested', True))
    try:
        settings = Settings(**{key: getattr(args, key) for key in Settings.__dataclass_fields__})
        return run(args.input, args.output, args.assets, args.checkpoint, args.seeds, settings, args.resume, stop=stop, benchmark_metadata=args.benchmark_metadata)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())

"""Bounded, credential-free HTTP transport around the unchanged resident ESMFold2 worker.

Only the host holds R2 credentials. Presigned URLs are capabilities: never log them.
"""
import argparse
import contextlib
import io
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request

from resident.worker import Settings, sha, atomic_json
from resident.benchmark import Benchmark, PROVIDER_KEYS, sanitize_metadata, load_metadata
import threading


def redact_log_line(line):
    line = re.sub(r'https?://[^\s\x1b<>"\']+', '[URL_REDACTED]', line)
    return re.sub(r'''(?i)(?:x-amz-(?:signature|credential|security-token)|authorization|api[_-]?key|access[_-]?key|secret[_-]?access[_-]?key|password|token)["']?\s*[:=].*''',
                  '[CREDENTIAL_REDACTED]\n', line)


def drain_logs(child, destination, observer=None):
    """Redact child output in memory before disk; never spool raw capabilities."""
    if not isinstance(child.stdout, io.IOBase):
        return None  # explicit subprocess unit-test doubles do not expose a pipe
    def drain():
        try:
            while True:
                raw = child.stdout.readline(65537)
                if not raw:
                    break
                if len(raw) > 65536:
                    while raw and not raw.endswith(b'\n'):
                        raw = child.stdout.readline(65537)
                    line = '[oversized log line suppressed]\n'
                else:
                    line = redact_log_line(raw.decode('utf-8', errors='replace'))
                destination.write(line.encode())
                destination.flush()
                if observer is not None and line.startswith('{'):
                    try:
                        event = json.loads(line)
                        if event.get('schema') == 1 and event.get('event') in {
                                'gpu_sample', 'stage_started', 'stage_finished', 'prediction_terminal',
                                'invocation_started', 'invocation_finished'}:
                            observer.event('worker_event', worker_event=event)
                    except (ValueError, AttributeError):
                        pass
        finally:
            child.stdout.close()
    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return thread


def small_json(path, limit):
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('observation exceeds snapshot read limit')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('observation must be an object')
    return value


class ProgressUploads:
    """Bounded create-only snapshot slots, each verified by a separate GET."""
    def __init__(self, http, job, capabilities, outputs, observer):
        self.http, self.job, self.capabilities = http, job, capabilities
        self.outputs, self.observer = outputs, observer
        self.interval = job.get('progress_interval_seconds', 60)
        self.disabled_reason = None
        if type(self.interval) is not int or not 10 <= self.interval <= 3600:
            self.interval = 60
        if not isinstance(capabilities, list) or len(capabilities) > 256:
            self.capabilities = []
            self.disabled_reason = 'invalid_capability_configuration'
        elif not capabilities:
            self.disabled_reason = 'no_capabilities'
        self.index = 0
        self.verified = []
        self.failed = []
        self.stop = threading.Event()
        self.thread = None
        self.timeout_seconds = 10.0
        self.closed = False

    def snapshot(self):
        # Only our own schema fields, bounded reads. Never upload logs or URLs.
        result = {'schema': 1, 'job_id': self.job['job_id'], 'attempt_id': self.job['attempt_id'],
                  'timestamp_unix': time.time(), 'slot': self.index, 'progress': None,
                  'gpu_sample': None, 'hardware': None, 'latest_prediction': None,
                  'last_stage_event': None, 'null_reason': 'worker_not_started_or_not_yet_observed'}
        observed = getattr(self.observer, 'summary', {})
        result['hardware'] = observed.get('hardware')
        result['caller_metadata'] = sanitize_metadata(observed.get('caller_metadata', {}))
        try:
            value = small_json(self.outputs / 'benchmark.json', 65536)
            result['hardware'] = value.get('hardware')
            result['caller_metadata'] = sanitize_metadata(value.get('caller_metadata', {}))
        except (OSError, ValueError):
            pass
        try:
            with (self.outputs / 'predictions.jsonl').open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 32768))
                last = stream.read(32768).splitlines()[-1]
            result['latest_prediction'] = json.loads(last)
        except (OSError, ValueError, IndexError):
            pass
        try:
            value = small_json(self.outputs / 'progress.json', 16384)
            result['progress'] = {k: value.get(k) for k in ('status', 'jobs_total', 'jobs_completed', 'jobs_failed', 'checkpoint_load_count', 'current')}
            result['null_reason'] = None
        except (OSError, ValueError):
            pass
        try:
            with (self.outputs / 'events.jsonl').open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 65536))
                lines = stream.read(65536).splitlines()
            for line in reversed(lines):
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if value.get('schema') == 1 and value.get('event') in ('stage_started', 'stage_finished') and result['last_stage_event'] is None:
                    result['last_stage_event'] = value
                if value.get('schema') == 1 and value.get('event') == 'gpu_sample' and result['gpu_sample'] is None:
                    result['gpu_sample'] = {k: value.get(k) for k in ('timestamp_unix', 'devices', 'null_reason')}
                if result['last_stage_event'] is not None and result['gpu_sample'] is not None:
                    break
        except OSError:
            pass
        return result

    def publish(self):
        if self.closed or self.index >= len(self.capabilities):
            return
        slot = self.index
        self.index += 1  # never retry ambiguous PUT; consume this bounded slot
        try:
            snapshot = self.snapshot()
            snapshot['slot'] = slot
            body = json_bytes(snapshot)
            if len(body) > 65536:
                raise ValueError('snapshot too large')
            cap = self.capabilities[slot]
            if not isinstance(cap, dict) or set(cap) != {'put', 'get'}:
                raise ValueError('missing progress capability')
            self.http.progress_request(cap['put'], 'PUT', data=body, timeout_seconds=self.timeout_seconds)
            got = self.http.progress_request(cap['get'], max_bytes=len(body), timeout_seconds=self.timeout_seconds)
            if got != (hashlib.sha256(body).hexdigest(), len(body)):
                raise TransportError('progress readback mismatch')
            self.verified.append(slot)
            self.observer.event('progress_upload_verified', slot=slot)
        except Exception:
            self.failed.append(slot)
            self.observer.event('progress_upload_unverified', slot=slot, reason='missing_expired_or_failed_transport')

    def start(self):
        if not self.capabilities:
            self.observer.event('progress_upload_disabled', reason=self.disabled_reason)
            return
        self.publish()
        def monitor():
            while not self.stop.wait(self.interval):
                self.publish()
                if self.index >= len(self.capabilities):
                    break
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()

    def shutdown(self):
        self.stop.set()
        if self.thread:
            # At most an in-flight PUT + GET deadline, plus process cleanup.
            self.thread.join(timeout=2 * self.timeout_seconds + 2)
            if self.thread.is_alive():
                # Fail closed: never freeze/archive with a writer still active.
                raise TransportError('progress publisher did not stop; archive refused')

    def close(self):
        self.shutdown()
        if not self.closed and self.capabilities:
            self.publish()
        self.closed = True
        return {'slots_available': len(self.capabilities), 'slots_attempted': self.index,
                'verified_slots': self.verified, 'unverified_slots': self.failed,
                'interval_seconds': self.interval, 'disabled_reason': self.disabled_reason,
                'durability': 'only successfully readback-verified slots; no tensors or resume state'}


MAX_ARCHIVE_BYTES = 1024 ** 3
MAX_INPUT_BYTES = 4 * 1024 ** 3
MAX_MEMBERS = 10000
DISK_RESERVE = 512 * 1024 ** 2


class TransportError(RuntimeError):
    """Deliberately contains no HTTP exception text (which may include a URL)."""


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def token(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError('invalid object path component')
    return value


def validate_settings(seeds, settings):
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s <= 4294967295 for s in seeds):
        raise ValueError('seeds must be explicit uint32 integers')
    if not isinstance(settings, dict) or set(settings) != set(asdict(Settings())):
        raise ValueError('invalid settings fields')
    for name in ('samples', 'steps', 'loops', 'max_parallel_samples'):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError('invalid sampling settings')
    if settings['max_parallel_samples'] != 1:
        raise ValueError('this engine samples one trajectory per native call')
    if settings['backend'] not in ('fused', 'reference'):
        raise ValueError('invalid kernel backend')
    if settings['output_format'] != 'mmcif':
        raise ValueError('this engine writes mmCIF only')
    if settings['derived'] not in ('none', 'tensors'):
        raise ValueError('invalid derived-metrics mode')


def safe_name(name):
    p = PurePosixPath(name)
    if (not name or not p.parts or '\\' in name or p.is_absolute() or '..' in p.parts
            or str(p) != name or any(ord(c) < 32 for c in name)):
        raise ValueError('unsafe archive path')
    return p


class InputTarInfo(tarfile.TarInfo):
    def _proc_member(self, archive):
        # Host creates USTAR. Reject extended/sparse metadata BEFORE tarfile can
        # allocate attacker-declared PAX/GNU header sizes or recurse into headers.
        if self.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
            raise ValueError('unsupported input archive record')
        return super()._proc_member(archive)


def extract_inputs(archive, destination):
    """Two-pass validation; no links, aliases, special files or unbounded inflate."""
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    total, seen = 0, set()
    with tarfile.open(archive, 'r:*', tarinfo=InputTarInfo) as tar:
        members = []
        for member in tar:
            p = safe_name(member.name)
            if p.parts[0] != 'inputs' or not (member.isfile() or member.isdir()) or member.name in seen:
                raise ValueError('unsafe or duplicate archive member')
            seen.add(member.name)
            total += member.size
            if len(seen) > MAX_MEMBERS or total > MAX_INPUT_BYTES:
                raise ValueError('input extraction limit exceeded')
            if member.size < 0 or (member.isdir() and member.size):
                raise ValueError('invalid archive size')
            members.append(member)
        if total + DISK_RESERVE > shutil.disk_usage(destination).free:
            raise ValueError('insufficient extraction disk space')
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                target.chmod(0o600)
    return destination / 'inputs'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HTTP:
    def __init__(self, expires_at):
        if type(expires_at) is not int or expires_at <= time.time():
            raise ValueError('missing or expired URL expiration')
        self.expires_at = expires_at
        self.opener = urllib.request.build_opener(NoRedirect())

    def require_unexpired(self):
        if self.expires_at <= time.time():
            raise TransportError('cloud attempt expired before dispatch')

    def progress_request(self, url, method='GET', data=None, max_bytes=65536, timeout_seconds=10.0):
        """Tiny progress transfers only: monotonic end-to-end process deadline.

        urllib socket timeouts cannot bound DNS, headers or slow-drip bodies.
        Isolate the operation so deadline expiry kills and reaps it, including
        during blocking connect/read. Capabilities travel on stdin, never argv
        or logs; the child has no output-file access through this API.
        Large input/result streaming keeps the original request semantics.
        """
        if method not in ('GET', 'PUT') or not 0 <= max_bytes <= 65536 or (data is not None and len(data) > 65536):
            raise TransportError('invalid progress transfer')
        remaining = min(timeout_seconds, self.expires_at - time.time())
        if remaining <= 0:
            raise TransportError('progress transfer expired')
        deadline = time.monotonic() + remaining
        payload = json_bytes({'expires_at': self.expires_at, 'url': url, 'method': method,
                              'data': data.decode('utf-8') if data is not None else None,
                              'max_bytes': max_bytes, 'timeout_seconds': remaining})
        code = ('import sys; sys.path.insert(0, ' + repr(str(Path(__file__).resolve().parent)) +
                '); from cloud_job import progress_http_child; progress_http_child()')
        child = None
        try:
            child = subprocess.Popen([sys.executable, '-I', '-c', code], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     env={})
            output, _ = child.communicate(payload, timeout=max(0, deadline - time.monotonic()))
            if child.returncode or time.monotonic() >= deadline:
                raise ValueError('progress deadline or transfer failure')
            result = json.loads(output)
            return tuple(result) if method == 'GET' else None
        except Exception:
            raise TransportError('progress HTTP transfer failed (details suppressed)') from None
        finally:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.wait()  # reap even after timeout; no orphan request process
                for stream in (child.stdin, child.stdout):
                    if stream is not None:
                        stream.close()

    def request(self, url, method='GET', path=None, data=None, max_bytes=MAX_ARCHIVE_BYTES, timeout_seconds=120.0):
        """Streaming file PUT/GET; every PUT is create-only, with no retries."""
        try:
            parsed = urllib.parse.urlsplit(url)
            if (parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost'))):
                raise ValueError('invalid transport URL')
            if parsed.username or parsed.password or parsed.fragment:
                raise ValueError('invalid transport URL')
            remaining = self.expires_at - time.time()
            if remaining <= 0:
                raise ValueError('expired')
            headers = {}
            stream = None
            if method == 'PUT':
                headers['If-None-Match'] = '*'
                if path is not None:
                    stream = Path(path).open('rb')
                    data = stream
                    headers['Content-Length'] = str(Path(path).stat().st_size)
                else:
                    headers['Content-Length'] = str(len(data))
            try:
                request = urllib.request.Request(url, data=data, headers=headers, method=method)
                with self.opener.open(request, timeout=min(timeout_seconds, remaining)) as response:
                    if method == 'GET':
                        size, digest = 0, hashlib.sha256()
                        sink = Path(path).open('xb') if path is not None else contextlib.nullcontext(None)
                        with sink as output:
                            while block := response.read(1024 * 1024):
                                size += len(block)
                                if size > max_bytes or time.time() >= self.expires_at:
                                    raise ValueError('download limit exceeded')
                                if output is not None:
                                    if len(block) + DISK_RESERVE > shutil.disk_usage(Path(path).parent).free:
                                        raise ValueError('insufficient download disk space')
                                    output.write(block)
                                digest.update(block)
                        return digest.hexdigest(), size
                    response.read(4096)
            finally:
                if stream is not None:
                    stream.close()
        except Exception:
            raise TransportError('cloud HTTP transfer failed (details suppressed)') from None


def progress_http_child():
    """Private isolated transport entry point; never emit capabilities/errors."""
    try:
        value = json.loads(sys.stdin.buffer.read())
        data = value.pop('data')
        http = HTTP(value.pop('expires_at'))
        result = http.request(data=data.encode('utf-8') if data is not None else None, **value)
        sys.stdout.write(json.dumps(result))
    except Exception:
        raise SystemExit(1) from None


def child_environment(runtime, image):
    # Explicit allowlist: no R2/AWS/Salad credentials or presigned capabilities.
    allowed = {'PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
               'NVIDIA_DRIVER_CAPABILITIES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'OPENBLAS_NUM_THREADS', 'PYTHONPATH', 'VIRTUAL_ENV', 'LANG', 'LC_ALL'}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    for key, suffix in {'HOME': 'home', 'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'cache',
                        'HF_HOME': 'cache/hf', 'TRITON_CACHE_DIR': 'cache/triton',
                        'NUMBA_CACHE_DIR': 'cache/numba', 'ESMFOLD2_CACHE': 'cache/esmfold2'}.items():
        path = runtime / suffix
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['COFOLD_IMAGE_ID'] = image
    return env


WORKER_FLAGS = {'samples': '--samples', 'steps': '--steps', 'loops': '--recycles',
                'max_parallel_samples': '--max-parallel-samples', 'output_format': '--output-format',
                'derived': '--derived'}


def worker_command(job, inputs, outputs, assets, checkpoint):
    cmd = ['cofold-run', '--input', str(inputs), '--output', str(outputs),
           '--assets', str(assets), '--checkpoint', str(checkpoint),
           '--seeds', *map(str, job['seeds'])]
    for key, value in job['settings'].items():
        if key == 'backend':
            cmd.extend(['--backend', str(value)])
        elif key in WORKER_FLAGS:
            cmd.extend([WORKER_FLAGS[key], str(value)])
        else:
            raise ValueError('unknown worker setting: ' + key)
    return cmd


def validate_job(job):
    if job.get('schema') != 1:
        raise ValueError('unsupported cloud job schema')
    for key in ('project', 'experiment', 'job_id', 'attempt_id'):
        token(job[key])
    if not re.fullmatch(r'[^\s?]+@sha256:[0-9a-f]{64}', job['image']):
        raise ValueError('an immutable image digest is required')
    validate_settings(job['seeds'], job['settings'])
    if not re.fullmatch('[0-9a-f]{64}', job['bundle_sha256']):
        raise ValueError('invalid bundle hash')
    if not isinstance(job['files'], dict) or not job['files'] or len(job['files']) > MAX_MEMBERS:
        raise ValueError('invalid input inventory')
    for name, digest in job['files'].items():
        safe_name(name)
        if not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('invalid input hash')


def archive_outputs(outputs, target):
    files = sorted(outputs.rglob('*'))
    if any(p.is_symlink() or not (p.is_dir() or p.is_file()) for p in files):
        raise ValueError('unsafe worker output')
    size = sum(p.stat().st_size for p in files if p.is_file())
    # Use an uncompressed tar: predictable disk requirement, lossless full outputs.
    overhead = (len(files) + 1) * 4096 + 10240
    if size + overhead + DISK_RESERVE > shutil.disk_usage(target.parent).free:
        raise ValueError('insufficient archive disk space')
    with tarfile.open(target, 'w') as tar:
        for path in files:
            tar.add(path, arcname='outputs/' + path.relative_to(outputs).as_posix(), recursive=False)


def run_cloud(spec, work, assets=None, checkpoint=None, benchmark_metadata=None):
    job, urls = spec['job'], spec['urls']
    validate_job(job)
    http = HTTP(job['expires_at'])
    if set(urls) != {'input_get', 'claim_put', 'result_put', 'result_get', 'receipt_put', 'receipt_get'}:
        raise ValueError('missing scoped transport URLs')
    # Remote create-only claim fences restarts AND replicas before any inference.
    http.request(urls['claim_put'], 'PUT', data=json_bytes({'job_id': job['job_id'], 'attempt_id': job['attempt_id']}))
    print(f"cloud job {job['job_id']} attempt {job['attempt_id']}: claimed", flush=True)
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    outputs = work / 'outputs'
    outputs.mkdir()
    metadata = load_metadata(benchmark_metadata)
    metadata.update(sanitize_metadata({k: os.environ[k] for k in PROVIDER_KEYS if k in os.environ}))
    transport_dir = work / 'transport'
    transport_dir.mkdir()
    observer = Benchmark(transport_dir, metadata, sample_gpu=False)
    env = child_environment(work / 'runtime', job['image'])
    metadata_path = work / 'runtime' / 'benchmark-metadata.json'
    atomic_json(metadata_path, metadata)
    progress = ProgressUploads(http, job, spec.get('progress_urls', []), outputs, observer)
    receipt = {'schema': 1, 'job_id': job['job_id'], 'attempt_id': job['attempt_id'],
               'status': 'failed', 'worker_exit_code': None, 'phase': 'inputs',
               'bundle_sha256': job['bundle_sha256'], 'image': job['image']}
    child = None
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        observer.__enter__()
        progress.start()
        try:
            with observer.stage('download_and_input_verification'):
                archive = work / 'inputs.tar.gz'
                print('cloud phase: downloading inputs', flush=True)
                actual, _ = http.request(urls['input_get'], path=archive)
                if actual != job['bundle_sha256']:
                    raise ValueError('input bundle hash mismatch')
                inputs = extract_inputs(archive, work / 'unpacked')
                actual_files = {p.relative_to(inputs).as_posix(): sha(p) for p in inputs.rglob('*') if p.is_file()}
                if actual_files != job['files']:
                    raise ValueError('input inventory mismatch')
            with observer.stage('assets'):
                receipt['phase'] = 'assets'
                http.require_unexpired()
                if stopped:
                    raise ValueError('stopped before asset preparation')
                print('cloud phase: preparing model assets', flush=True)
                if (assets is None) != (checkpoint is None):
                    raise ValueError('assets and checkpoint must be supplied together')
                if assets is None:
                    assets = work / 'assets'
                    # Downloader uses pinned sizes/hashes; offline flags apply only below.
                    with (work / 'assets.log').open('wb') as log:
                        child = subprocess.Popen([sys.executable, str(Path(__file__).with_name('fetch_assets.py')),
                                                  '--assets', str(assets), '--download'],
                                                 env=env, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        if stopped and child.poll() is None:
                            child.send_signal(signal.SIGTERM)
                        reader = drain_logs(child, log)
                        rc = child.wait()
                        if reader:
                            reader.join()
                    if rc:
                        raise ValueError('asset preparation failed')
                    checkpoint = assets
                else:
                    assets, checkpoint = Path(assets).resolve(), Path(checkpoint).resolve()
                    if not assets.is_dir() or not checkpoint.is_dir():
                        raise ValueError('local assets/checkpoint directories missing')
            with observer.stage('worker_total'):
                receipt['phase'] = 'worker'
                http.require_unexpired()
                if stopped:
                    raise ValueError('stopped before inference')
                env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                print('cloud phase: running ESMFold2; detailed logs retained in results', flush=True)
                log_path = work / 'worker.log'
                with log_path.open('wb') as log:
                    child = subprocess.Popen(worker_command(job, inputs, outputs, assets, checkpoint) +
                                             ['--benchmark-metadata', str(metadata_path)],
                                             env=env, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    if stopped and child.poll() is None:
                        child.send_signal(signal.SIGTERM)
                    reader = drain_logs(child, log, observer)
                    rc = child.wait()
                    if reader:
                        reader.join()
                log_path.rename(outputs / 'worker.log')
                receipt['worker_exit_code'] = rc
                # Native worker verifies outputs before declaring completed. Check its final metadata too.
                meta = json.loads((outputs / 'run.json').read_text())
                if rc == 0 and not stopped and meta.get('status') == 'completed':
                    receipt['status'] = 'completed'
                elif stopped or rc == 143:
                    receipt['status'] = 'stopped'
        except Exception:
            if stopped:
                receipt['status'] = 'stopped'
            # Never serialize exception text: libraries may include signed URLs.
            receipt['error'] = 'cloud job failed; inspect credential-free worker/asset logs if present'
        receipt['finished_at'] = int(time.time())
        for name in ('assets.log', 'worker.log'):
            if (work / name).is_file():
                (work / name).replace(outputs / name)
        receipt['progress_uploads'] = progress.close()
        observer.summary['status'] = receipt['status']
        atomic_json(transport_dir / 'benchmark.json', observer.summary)
        # Freeze a pre-upload snapshot: an archive cannot include its own final upload timing.
        shutil.copytree(transport_dir, outputs / 'cloud_transport')
        result = work / 'results.tar'
        print('cloud phase: archiving and uploading results', flush=True)
        with observer.stage('outputs_archive'):
            archive_outputs(outputs, result)
            digest = sha(result)
            size = result.stat().st_size
        with observer.stage('upload'):
            http.request(urls['result_put'], 'PUT', path=result)
        # No authoritative receipt until the exact uploaded archive reads back intact.
        with observer.stage('upload_verification'):
            read_hash, read_size = http.request(urls['result_get'], max_bytes=size)
        if (read_hash, read_size) != (digest, size):
            raise TransportError('result readback checksum mismatch')
        receipt.update(result_sha256=digest, result_bytes=size, benchmark={
            'stages': observer.summary['stages'], 'caller_metadata': metadata,
            'wrapper_wall_seconds_before_receipt': time.perf_counter() - observer.start,
            'archive_observations_scope': 'pre_archive_and_upload; later timings in this receipt'})
        print('cloud phase: result readback verified; publishing receipt', flush=True)
        receipt_bytes = json_bytes(receipt)
        http.request(urls['receipt_put'], 'PUT', data=receipt_bytes)
        read_hash, read_size = http.request(urls['receipt_get'], max_bytes=len(receipt_bytes))
        if (read_hash, read_size) != (hashlib.sha256(receipt_bytes).hexdigest(), len(receipt_bytes)):
            raise TransportError('receipt readback checksum mismatch')
        print(f"cloud terminal status: {receipt['status']}", flush=True)
        return 0 if receipt['status'] == 'completed' else 1
    finally:
        try:
            progress.shutdown()
        finally:
            observer.__exit__(*sys.exc_info())
            signal.signal(signal.SIGTERM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fetch one cloud attempt, invoke cofold-run, upload full results; no deployment')
    parser.add_argument('--work', type=Path, default=Path('/work/cloud-attempt'))
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--benchmark-metadata', type=Path)
    args = parser.parse_args(argv)
    try:
        spec = json.loads(os.environ.pop('COFOLD_CLOUD_SPEC'))
        return run_cloud(spec, args.work, args.assets, args.checkpoint, args.benchmark_metadata)
    except Exception:
        print('cloud attempt failed (transport/configuration details suppressed); do not restart with the same URLs', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

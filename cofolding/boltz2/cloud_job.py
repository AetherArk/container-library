"""Bounded, credential-free HTTP transport around the unchanged resident worker.

Only the host holds R2 credentials. Presigned URLs are capabilities: never log them.
"""
import argparse
import contextlib
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

from resident.worker import Settings, sha

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
    for name in ('samples', 'steps', 'max_parallel_samples', 'recycles'):
        if type(settings[name]) is not int or settings[name] < (0 if name == 'recycles' else 1):
            raise ValueError('invalid sampling settings')
    if any(type(settings[name]) is not bool for name in ('no_kernels', 'use_potentials')):
        raise ValueError('invalid boolean settings')
    if settings['output_format'] not in ('mmcif', 'pdb'):
        raise ValueError('invalid output format')


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

    def request(self, url, method='GET', path=None, data=None, max_bytes=MAX_ARCHIVE_BYTES):
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
                with self.opener.open(request, timeout=min(120, remaining)) as response:
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


def child_environment(runtime, image):
    # Explicit allowlist: no R2/AWS/Salad credentials or presigned capabilities.
    allowed = {'PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
               'NVIDIA_DRIVER_CAPABILITIES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'OPENBLAS_NUM_THREADS', 'PYTHONPATH', 'VIRTUAL_ENV', 'LANG', 'LC_ALL'}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    for key, suffix in {'HOME': 'home', 'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'cache',
                        'HF_HOME': 'cache/hf', 'TRITON_CACHE_DIR': 'cache/triton',
                        'NUMBA_CACHE_DIR': 'cache/numba', 'BOLTZ_CACHE': 'cache/boltz'}.items():
        path = runtime / suffix
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['COFOLD_IMAGE_ID'] = image
    return env


def worker_command(job, inputs, outputs, assets, checkpoint):
    cmd = ['cofold-run', '--input', str(inputs), '--output', str(outputs),
           '--assets', str(assets), '--checkpoint', str(checkpoint),
           '--seeds', *map(str, job['seeds'])]
    for key, value in job['settings'].items():
        if type(value) is bool:
            if value:
                cmd.append('--' + key.replace('_', '-'))
        else:
            cmd.extend(['--' + key.replace('_', '-'), str(value)])
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


def run_cloud(spec, work, assets=None, checkpoint=None):
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
    env = child_environment(work / 'runtime', job['image'])
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
        try:
            archive = work / 'inputs.tar.gz'
            print('cloud phase: downloading inputs', flush=True)
            actual, _ = http.request(urls['input_get'], path=archive)
            if actual != job['bundle_sha256']:
                raise ValueError('input bundle hash mismatch')
            inputs = extract_inputs(archive, work / 'unpacked')
            actual_files = {p.relative_to(inputs).as_posix(): sha(p) for p in inputs.rglob('*') if p.is_file()}
            if actual_files != job['files']:
                raise ValueError('input inventory mismatch')
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
                                              '--cache', str(assets), '--download', '--structure-only'],
                                             env=env, cwd=work, stdout=log, stderr=subprocess.STDOUT)
                    if stopped and child.poll() is None:
                        child.send_signal(signal.SIGTERM)
                    rc = child.wait()
                if rc:
                    raise ValueError('asset preparation failed')
                checkpoint = assets / 'boltz2_conf.ckpt'
            else:
                assets, checkpoint = Path(assets).resolve(), Path(checkpoint).resolve()
                if not assets.is_dir() or not checkpoint.is_file():
                    raise ValueError('local assets/checkpoint missing')
            receipt['phase'] = 'worker'
            http.require_unexpired()
            if stopped:
                raise ValueError('stopped before inference')
            env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            print('cloud phase: running Boltz2; detailed logs retained in results', flush=True)
            log_path = work / 'worker.log'
            with log_path.open('wb') as log:
                child = subprocess.Popen(worker_command(job, inputs, outputs, assets, checkpoint),
                                         env=env, cwd=work, stdout=log, stderr=subprocess.STDOUT)
                if stopped and child.poll() is None:
                    child.send_signal(signal.SIGTERM)
                rc = child.wait()
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
        result = work / 'results.tar'
        print('cloud phase: archiving and uploading results', flush=True)
        archive_outputs(outputs, result)
        digest = sha(result)
        size = result.stat().st_size
        http.request(urls['result_put'], 'PUT', path=result)
        # No authoritative receipt until the exact uploaded archive reads back intact.
        read_hash, read_size = http.request(urls['result_get'], max_bytes=size)
        if (read_hash, read_size) != (digest, size):
            raise TransportError('result readback checksum mismatch')
        receipt.update(result_sha256=digest, result_bytes=size)
        print('cloud phase: result readback verified; publishing receipt', flush=True)
        receipt_bytes = json_bytes(receipt)
        http.request(urls['receipt_put'], 'PUT', data=receipt_bytes)
        read_hash, read_size = http.request(urls['receipt_get'], max_bytes=len(receipt_bytes))
        if (read_hash, read_size) != (hashlib.sha256(receipt_bytes).hexdigest(), len(receipt_bytes)):
            raise TransportError('receipt readback checksum mismatch')
        print(f"cloud terminal status: {receipt['status']}", flush=True)
        return 0 if receipt['status'] == 'completed' else 1
    finally:
        signal.signal(signal.SIGTERM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fetch one cloud attempt, invoke cofold-run, upload full results; no deployment')
    parser.add_argument('--work', type=Path, default=Path('/work/cloud-attempt'))
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args(argv)
    try:
        spec = json.loads(os.environ.pop('COFOLD_CLOUD_SPEC'))
        return run_cloud(spec, args.work, args.assets, args.checkpoint)
    except Exception:
        print('cloud attempt failed (transport/configuration details suppressed); do not restart with the same URLs', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Launch one installed offline cofold worker; native inputs are never rewritten."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('image', 'inputs', 'assets', 'checkpoint', 'output', 'gpu'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--cpus', type=float, default=8)
    p.add_argument('--memory', default='36g')
    p.add_argument('--timeout', type=float, help='Whole run seconds; default: unlimited')
    p.add_argument('--job-timeout', type=float,
                   help='Per input x seed job seconds, forwarded to the worker; the job is then '
                        "recorded as failed with error_kind='timeout' and later jobs still run")
    p.add_argument('--stop-grace', type=float, default=120, help='Job-boundary drain seconds before owned-container kill')
    p.add_argument('--benchmark-metadata',
                   help='Optional caller-metadata JSON file inside --inputs; forwarded as /inputs/<name>')
    p.add_argument('--lock-root', help='Shared COFOLD_GPU_LOCK_ROOT / XDG user-state GPU locks')
    p.add_argument('--estimate-attempt-bytes', type=int,
                   help='Optional caller estimate of bytes one attempt needs; compared with free '
                        'space on the output filesystem before dispatch. Absent: free space is '
                        'recorded and only the comparison is skipped.')
    p.add_argument('--dry-run', action='store_true')
    for name in ('samples', 'recycles', 'steps', 'max-parallel-samples'):
        p.add_argument('--' + name, type=int)
    for name in ('no-kernels', 'use-potentials', 'resume'):
        p.add_argument('--' + name, action='store_true')
    p.add_argument('--output-format', choices=('mmcif', 'pdb'))
    p.add_argument('--capture', choices=('full', 'none'),
                   help="'none' omits the per-job capture/ tensors; forward this to cofold-run")
    return p


def checked(cmd):
    r = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
    if r.returncode:
        raise ValueError(f'{shlex.join(cmd)}: {r.stderr.strip()}')
    return r.stdout.strip()


def resolve_image(reference):
    if not reference or reference.startswith('-') or any(c.isspace() for c in reference):
        raise ValueError('Invalid image reference')
    identity = checked(['docker', 'image', 'inspect', '--format', '{{.Id}}', reference])
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', identity):
        raise ValueError('Expected an immutable installed image ID; never pulling')
    return identity


def mount_path(value):
    path = Path(value).expanduser().resolve()
    if any(c in str(path) for c in ':,\n\r'):
        raise ValueError('Mount paths cannot contain colons, commas or newlines')
    return path


def validate(a):
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', a.gpu):
        raise ValueError('--gpu requires an explicit full GPU UUID')
    for name in ('inputs', 'assets', 'checkpoint', 'output'):
        setattr(a, name, mount_path(getattr(a, name)))
    if not a.inputs.is_dir():
        raise ValueError('Inputs must be a directory; the selected worker parses its native files')
    if not a.assets.is_dir() or not (a.checkpoint.is_file() or a.checkpoint.is_dir()):
        raise ValueError('Assets directory and explicit checkpoint file or directory must exist')
    for source in (a.inputs, a.assets, a.checkpoint):
        if a.output == source or a.output in source.parents or source in a.output.parents:
            raise ValueError('Output must not overlap read-only inputs/assets/checkpoint')
    if not math.isfinite(a.cpus) or a.cpus <= 0 or not re.fullmatch(r'[1-9][0-9]*[bkmgBKMG]?', a.memory):
        raise ValueError('Invalid CPU/memory limits')
    if any(x is not None and (not math.isfinite(x) or x <= 0) for x in (a.timeout, a.stop_grace, a.job_timeout)):
        raise ValueError('Timeout, job timeout and stop grace must be positive finite seconds')
    if a.estimate_attempt_bytes is not None and a.estimate_attempt_bytes < 0:
        raise ValueError('--estimate-attempt-bytes must be a non-negative integer')
    if a.benchmark_metadata is not None:
        # The container sees only /inputs; a host path is never forwarded.
        metadata = mount_path(a.benchmark_metadata)
        if a.inputs not in metadata.parents or not metadata.is_file():
            raise ValueError('--benchmark-metadata must be an existing file inside --inputs')
        a.benchmark_metadata = metadata
    if any(not 0 <= s <= 4294967295 for s in a.seeds):
        raise ValueError('Seeds must be uint32 integers')
    for name in ('samples', 'recycles', 'steps', 'max_parallel_samples'):
        value = getattr(a, name)
        if value is not None and value < (0 if name == 'recycles' else 1):
            raise ValueError('Invalid native numerical option: ' + name)


def command(a, image, token):
    runtime = a.output / '.runtime'
    cmd = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--sig-proxy=false',
           '--name', 'cofold-' + token, '--cidfile', str(runtime / (token + '.cid')),
           '--user', f'{os.getuid()}:{os.getgid()}', '--gpus', 'device=' + a.gpu,
           '--cpus', str(a.cpus), '--memory', a.memory, '--memory-swap', a.memory,
           '--workdir', '/outputs/.runtime', '--env', 'HOME=/outputs/.runtime/home',
           '--env', 'TMPDIR=/outputs/.runtime/tmp', '--env', 'XDG_CACHE_HOME=/outputs/.runtime/cache',
           '--env', 'TRITON_CACHE_DIR=/outputs/.runtime/cache/triton',
           '--env', 'NUMBA_CACHE_DIR=/outputs/.runtime/cache/numba',
           '--env', 'TORCHINDUCTOR_CACHE_DIR=/outputs/.runtime/cache/torchinductor',
           '--env', 'TORCH_HOME=/outputs/.runtime/cache/torch',
           '--env', 'HF_HOME=/outputs/.runtime/cache/hf', '--env', 'HF_HUB_OFFLINE=1',
           '--env', 'COFOLD_IMAGE_ID=' + image]
    for source, target, readonly in ((a.inputs, '/inputs', True), (a.assets, '/models', True),
                                     (a.checkpoint, '/checkpoint', True), (a.output, '/outputs', False)):
        cmd += ['--mount', f'type=bind,src={source},dst={target}' + (',readonly' if readonly else '')]
    cmd += ['--entrypoint', 'cofold-run', image, '--input', '/inputs', '--output', '/outputs',
            '--assets', '/models', '--checkpoint', '/checkpoint', '--seeds', *map(str, a.seeds)]
    if a.job_timeout is not None:
        cmd += ['--job-timeout', str(a.job_timeout)]
    if a.benchmark_metadata is not None:
        cmd += ['--benchmark-metadata', '/inputs/' + a.benchmark_metadata.relative_to(a.inputs).as_posix()]
    for name in ('samples', 'recycles', 'steps', 'max_parallel_samples', 'no_kernels',
                 'use_potentials', 'output_format', 'capture', 'resume'):
        value = getattr(a, name)
        if value is not None and value is not False:
            cmd.append('--' + name.replace('_', '-'))
            if value is not True:
                cmd.append(str(value))
    return cmd


def gpu_available(gpu):
    found = checked(['nvidia-smi', '-i', gpu, '--query-gpu=uuid', '--format=csv,noheader'])
    if found != gpu:
        raise ValueError('Selected GPU UUID unavailable')
    apps = checked(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    if apps:
        raise ValueError('GPU has active compute processes; release it before launching')


def owned_signal(cidfile, sig):
    """Only an ID written by this invocation's Docker run is ever signalled."""
    if cidfile.is_file():
        cid = cidfile.read_text().strip()
        if re.fullmatch(r'[0-9a-f]{64}', cid):
            subprocess.run(['docker', 'kill', '--signal', sig, cid], timeout=30, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    return False


def run_worker(cmd, cidfile, timeout, grace, termination=None):
    """Run the owned container; fill `termination` when the run did not end normally.

    The dict is populated only for a whole-run timeout, a launcher SIGINT/SIGTERM
    drain, or a container killed by a signal, so the caller can write termination.json
    without ever writing it on a normal successful exit.
    """
    interrupted = [None]
    old = {}
    def receive(sig, frame):
        interrupted[0] = interrupted[0] or sig
    for sig in (signal.SIGINT, signal.SIGTERM):
        old[sig] = signal.signal(sig, receive)
    process = None
    try:
        process = subprocess.Popen(cmd)
        start = time.monotonic()
        draining = None
        term_sent = False
        timed_out = False
        while process.poll() is None:
            now = time.monotonic()
            if timeout is not None and now - start >= timeout:
                timed_out = True
            if interrupted[0] or timed_out:
                if draining is None:
                    draining = now
                if not term_sent:
                    term_sent = owned_signal(cidfile, 'TERM')
                if now - draining >= grace:
                    owned_signal(cidfile, 'KILL')
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.terminate()  # only this invocation's Docker client
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    break
            time.sleep(0.1)
        rc = process.wait()
        if termination is not None:
            if timed_out:
                termination.update(reason='timeout', signal=None)
            elif interrupted[0]:
                termination.update(reason='signal', signal=signal.Signals(interrupted[0]).name)
            elif rc == 137:
                termination.update(reason='signal', signal='SIGKILL')
        if timed_out:
            return 124
        if interrupted[0] and rc == 0:
            return 128 + interrupted[0]
        return rc if rc >= 0 else 128 - rc
    finally:
        if process is not None and process.poll() is None:
            owned_signal(cidfile, 'TERM')
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                owned_signal(cidfile, 'KILL')
                process.kill()
                process.wait()
        for sig, handler in old.items():
            signal.signal(sig, handler)


def container_id(cidfile):
    """The ID this invocation's Docker run wrote, or null; never a guessed ID."""
    try:
        cid = Path(cidfile).read_text().strip()
    except OSError:
        return None
    return cid if re.fullmatch(r'[0-9a-f]{64}', cid) else None


def last_progress(output):
    try:
        data = json.loads((Path(output) / 'progress.json').read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def memory_bytes(value):
    """Parse the same size syntax validate() accepts into bytes."""
    match = re.fullmatch(r'([1-9][0-9]*)([bkmgBKMG]?)', value)
    if not match:
        raise ValueError('Invalid memory size')
    factors = {'': 1, 'b': 1, 'k': 1024, 'm': 1024 ** 2, 'g': 1024 ** 3}
    return int(match.group(1)) * factors[match.group(2).lower()]


def available_memory_bytes():
    """Host MemAvailable in bytes, or None when /proc/meminfo cannot be read."""
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def job_count(inputs, seeds):
    """Number of attempts one invocation will prepare: native YAML inputs x seeds."""
    sources = [p for p in Path(inputs).iterdir()
               if p.is_file() and p.suffix.lower() in ('.yaml', '.yml')]
    return len(sources) * len(seeds)


def preflight(a, output):
    """Measure disk and memory before dispatch; never invents an attempt size."""
    jobs = job_count(a.inputs, a.seeds)
    free = shutil.disk_usage(str(output)).free
    estimate = a.estimate_attempt_bytes
    required = estimate * jobs if estimate is not None else None
    memory_requested = memory_bytes(a.memory)
    memory_available = available_memory_bytes()
    return {
        'schema': 1,
        'jobs': jobs,
        'estimate_attempt_bytes': estimate,
        'estimated_total_bytes': required,
        'free_bytes_output': free,
        'disk_check': 'skipped_no_estimate' if estimate is None else 'compared',
        'memory_requested': a.memory,
        'memory_requested_bytes': memory_requested,
        'memory_available_bytes': memory_available,
        'memory_check': 'skipped_unavailable' if memory_available is None else 'compared',
    }


def check_preflight(record, output):
    """Reject an over-committed attempt before any container starts (exit 2)."""
    estimate = record['estimate_attempt_bytes']
    if estimate is not None and record['free_bytes_output'] < record['estimated_total_bytes']:
        raise ValueError(
            f'preflight: free disk {record["free_bytes_output"]} B on {output} is below the '
            f'estimated {record["estimated_total_bytes"]} B '
            f'({estimate} B/attempt x {record["jobs"]} jobs)')
    available = record['memory_available_bytes']
    if available is not None and record['memory_requested_bytes'] > available:
        raise ValueError(
            f'preflight: --memory {record["memory_requested"]} '
            f'({record["memory_requested_bytes"]} B) exceeds host available memory ({available} B)')


def write_json(path, obj):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(obj, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def merge_preflight(output, report):
    """Attach the measured preflight values to the worker's run.json."""
    path = Path(output) / 'run.json'
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    data['preflight'] = report
    write_json(path, data)


def write_termination(output, cidfile, reason, signal_name, exit_code):
    """Record a launcher-killed or signalled run; never called for a normal exit."""
    payload = {'schema': 1, 'reason': reason, 'signal': signal_name, 'exit_code': int(exit_code),
               'container_id': container_id(cidfile), 'last_progress': last_progress(output),
               'timestamp_unix': time.time()}
    path = Path(output) / 'termination.json'
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def terminal_receipt(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError('run.json must be a JSON object')
    status = str(data.get('status', '')).lower()
    if status not in {'complete', 'completed', 'success', 'succeeded', 'failed', 'fatal', 'error', 'interrupted', 'cancelled', 'stopped', 'partial'}:
        raise ValueError('run.json is missing a recognized terminal status')
    print(f'run.json: {path} (status={status})')
    return status in {'complete', 'completed', 'success', 'succeeded'}


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        validate(a)
        image = resolve_image(a.image)
        token = uuid.uuid4().hex
        cmd = command(a, image, token)
        print(shlex.join(cmd), flush=True)
        if a.dry_run:
            return 0
        a.output.mkdir(parents=True, exist_ok=True)
        runtime = a.output / '.runtime'
        runtime.mkdir(exist_ok=True)
        lockroot = Path(a.lock_root or os.environ.get('COFOLD_GPU_LOCK_ROOT') or
                        Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local/state')) / 'cofolding/gpu-locks').expanduser().resolve()
        lockroot.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            for path in (lockroot / (a.gpu + '.lock'), runtime / 'launcher.lock'):
                lock = stack.enter_context(path.open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            gpu_available(a.gpu)
            report = preflight(a, a.output)
            if report['disk_check'] == 'skipped_no_estimate':
                print(f"preflight: no --estimate-attempt-bytes; recorded "
                      f"{report['free_bytes_output']} B free on {a.output}, disk comparison skipped",
                      flush=True)
            check_preflight(report, a.output)
            for name in ('home', 'tmp', 'cache'):
                (runtime / name).mkdir(exist_ok=True)
            receipt = a.output / 'run.json'
            before = receipt.stat().st_mtime_ns if receipt.exists() else None
            cidfile = runtime / (token + '.cid')
            termination = {}
            rc = run_worker(cmd, cidfile, a.timeout, a.stop_grace, termination)
            if termination:
                try:
                    write_termination(a.output, cidfile, termination['reason'], termination['signal'], rc)
                except OSError as exc:
                    print(f'termination.json not written: {exc}', file=sys.stderr)
            current = False
            try:
                if not receipt.exists() or receipt.stat().st_mtime_ns == before:
                    raise ValueError('No current run.json written by worker')
                success = terminal_receipt(receipt)
                current = True
                if rc == 0 and not success:
                    rc = 1
            except (ValueError, OSError) as exc:
                print(f'Receipt verification failed: {exc}', file=sys.stderr)
                if rc == 0:
                    rc = 1
            # Only a run.json this invocation wrote gets the measured values; merging
            # before the mtime check would make a stale receipt look current.
            if current:
                merge_preflight(a.output, report)
            return rc
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f'cofold launcher: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

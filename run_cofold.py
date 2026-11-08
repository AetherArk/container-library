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
import signal
import subprocess
import sys
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
    p.add_argument('--stop-grace', type=float, default=120, help='Job-boundary drain seconds before owned-container kill')
    p.add_argument('--lock-root', help='Shared COFOLD_GPU_LOCK_ROOT / XDG user-state GPU locks')
    p.add_argument('--dry-run', action='store_true')
    for name in ('samples', 'recycles', 'steps', 'max-parallel-samples'):
        p.add_argument('--' + name, type=int)
    for name in ('no-kernels', 'use-potentials', 'resume'):
        p.add_argument('--' + name, action='store_true')
    p.add_argument('--output-format', choices=('mmcif', 'pdb'))
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
    if any(x is not None and (not math.isfinite(x) or x <= 0) for x in (a.timeout, a.stop_grace)):
        raise ValueError('Timeout and stop grace must be positive finite seconds')
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
           '--env', 'HF_HOME=/outputs/.runtime/cache/hf', '--env', 'HF_HUB_OFFLINE=1',
           '--env', 'COFOLD_IMAGE_ID=' + image]
    for source, target, readonly in ((a.inputs, '/inputs', True), (a.assets, '/models', True),
                                     (a.checkpoint, '/checkpoint', True), (a.output, '/outputs', False)):
        cmd += ['--mount', f'type=bind,src={source},dst={target}' + (',readonly' if readonly else '')]
    cmd += ['--entrypoint', 'cofold-run', image, '--input', '/inputs', '--output', '/outputs',
            '--assets', '/models', '--checkpoint', '/checkpoint', '--seeds', *map(str, a.seeds)]
    for name in ('samples', 'recycles', 'steps', 'max_parallel_samples', 'no_kernels',
                 'use_potentials', 'output_format', 'resume'):
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


def run_worker(cmd, cidfile, timeout, grace):
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
            for name in ('home', 'tmp', 'cache'):
                (runtime / name).mkdir(exist_ok=True)
            receipt = a.output / 'run.json'
            before = receipt.stat().st_mtime_ns if receipt.exists() else None
            rc = run_worker(cmd, runtime / (token + '.cid'), a.timeout, a.stop_grace)
            try:
                if not receipt.exists() or receipt.stat().st_mtime_ns == before:
                    raise ValueError('No current run.json written by worker')
                success = terminal_receipt(receipt)
                if rc == 0 and not success:
                    return 1
            except (ValueError, OSError) as exc:
                print(f'Receipt verification failed: {exc}', file=sys.stderr)
                if rc == 0:
                    return 1
            return rc
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f'cofold launcher: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

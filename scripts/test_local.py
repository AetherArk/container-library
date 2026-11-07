#!/usr/bin/env python3
"""Local-only test levels. Never builds, installs, downloads, pulls or publishes."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('local_launcher', ROOT / 'run_cofold.py')
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def run(cmd):
    print('+ ' + __import__('shlex').join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=True)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def coverage_matches(jobs, sources, seeds):
    expected = Counter((name, seed) for name in sources for seed in seeds)
    actual = Counter((Path(job['input']).name, job['seed']) for job in jobs)
    return actual == expected


def verify_gpu(output, inputs, checkpoint, seeds, samples, output_format):
    """Current resident-worker evidence, not an exit-only or historical check."""
    data = json.loads((output / 'run.json').read_text())
    if data.get('status') != 'completed' or data.get('checkpoint_load_count') != 1:
        raise ValueError('GPU test requires completed real work with exactly one checkpoint load')
    if data.get('requested_seeds') != seeds:
        raise ValueError('Requested seed list was not preserved')
    if data.get('checkpoint', {}).get('sha256') != sha(checkpoint):
        raise ValueError('Worker checkpoint identity does not match supplied bytes')
    sources = sorted(p.name for p in inputs.iterdir() if p.suffix.lower() in ('.yaml', '.yml') and p.is_file())
    jobs = data.get('jobs', [])
    if not coverage_matches(jobs, sources, seeds):
        raise ValueError('Missing, duplicated or unexpected input × seed coverage')
    model = data.get('model')
    if not isinstance(model, dict) or not model:
        raise ValueError('Missing real model identity/device evidence')
    devices = model.get('parameter_devices', {})
    if (not str(model.get('device', '')).startswith('cuda') or not devices
            or any(not device.startswith('cuda') for device in devices)
            or sum(devices.values()) != model.get('named_parameter_count')
            or model.get('named_parameter_count', 0) <= 0
            or not model.get('model_id') or not model.get('parameter_data_ptr')
            or not model.get('named_parameter_storage_sha256')
            or model.get('cpu_to_gpu_transfer_count') != 1):
        raise ValueError('Missing CUDA-resident parameter/model storage or one-transfer evidence')
    extension = '.cif' if output_format == 'mmcif' else '.pdb'
    for job in jobs:
        if job.get('status') != 'completed' or job.get('reused'):
            raise ValueError('GPU acceptance requires freshly completed jobs, not resumed historical artifacts')
        if job.get('model_before') != model or job.get('model_after') != model:
            raise ValueError('Model identity/storage/device did not remain constant across jobs')
        attempt = (output / job['output']).resolve()
        if output.resolve() not in attempt.parents:
            raise ValueError('Worker attempt path escapes output')
        receipt = json.loads((attempt / 'receipt.json').read_text())
        if receipt.get('status') != 'completed' or not receipt.get('sha256'):
            raise ValueError('Missing completed native-output inventory')
        for relative, expected_hash in receipt['sha256'].items():
            path = (attempt / relative).resolve()
            if attempt not in path.parents or not path.is_file():
                raise ValueError('Missing/unsafe native output')
            if sha(path) != expected_hash:
                raise ValueError('Native output hash mismatch: ' + relative)
        structures = [p for p in attempt.rglob('*' + extension) if '_model_' in p.name]
        confidences = list(attempt.rglob('confidence_*.json'))
        if len(structures) != samples or len(confidences) != samples or any(p.stat().st_size == 0 for p in structures):
            raise ValueError('Native structure/confidence sample coverage mismatch')
    print(f'GPU evidence verified: {len(jobs)} fresh input×seed jobs; one checkpoint/model; native outputs hashed')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True, choices=('cpu', 'image', 'image-cloud', 'gpu'))
    p.add_argument('--image')
    for name in ('inputs', 'assets', 'checkpoint', 'output', 'gpu'):
        p.add_argument('--' + name)
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--samples', type=int, default=1)
    p.add_argument('--recycles', type=int, default=3)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--max-parallel-samples', type=int, default=1)
    p.add_argument('--output-format', choices=('mmcif', 'pdb'), default='mmcif')
    p.add_argument('--no-kernels', action='store_true')
    p.add_argument('--use-potentials', action='store_true')
    p.add_argument('--cpus', type=float, default=8)
    p.add_argument('--memory', default='36g')
    p.add_argument('--timeout', type=float)
    p.add_argument('--lock-root')
    a = p.parse_args(argv)
    if a.suite == 'cpu':
        failed = False
        for directory in (ROOT / 'cofolding/boltz2/tests', ROOT / 'tests'):
            try:
                run([sys.executable, '-m', 'unittest', 'discover', '-s', directory, '-v'])
            except subprocess.CalledProcessError:
                failed = True
        return int(failed)
    if not a.image:
        p.error('--image is required for image/gpu suites (explicit installed candidate)')
    if a.suite in ('image', 'image-cloud'):
        identity = launcher.resolve_image(a.image)
        common = ['docker', 'run', '--rm', '--pull=never', '--network=none',
                  '--user', f'{os.getuid()}:{os.getgid()}', '--cpus', str(a.cpus), '--memory', a.memory,
                  '--tmpfs', '/tmp:rw,nosuid,nodev,mode=1777,size=1g', '--workdir', '/tmp',
                  '--env', 'HOME=/tmp', '--env', 'TMPDIR=/tmp',
                  '--env', 'XDG_CACHE_HOME=/tmp/cache', '--env', 'NUMBA_CACHE_DIR=/tmp/numba',
                  '--env', 'TRITON_CACHE_DIR=/tmp/triton', '--env', 'HF_HOME=/tmp/hf',
                  '--env', 'CUDA_VISIBLE_DEVICES=', '--env', 'NVIDIA_VISIBLE_DEVICES=void']
        if a.suite == 'image-cloud':
            run(common + ['--entrypoint', 'cofold-cloud-run', identity, '--help'])
            run(common + ['--entrypoint', 'python', identity, '-c',
                          'import cloud_job, fetch_assets; from pathlib import Path; '
                          'assert Path(fetch_assets.__file__).with_name("assets.json").is_file(); '
                          'assert cloud_job.worker_command; print("installed cloud imports/assets OK")'])
            print('Installed cloud CPU smoke checks passed; no network, GPU or source mounts used')
            return 0
        run(common + ['--entrypoint', 'cofold-run', identity, '--help'])
        run(common + ['--entrypoint', 'python', identity, '-c',
                      'import torch, boltz; import resident.worker; import resident.native; '
                      'assert not torch.cuda.is_available(); print("installed CPU imports OK")'])
        print('Installed-image CPU checks passed; no GPU or source mounts used')
        return 0
    for name in ('inputs', 'assets', 'checkpoint', 'output', 'gpu', 'seeds'):
        if not getattr(a, name):
            p.error('--' + name + ' is required for gpu suite')
    output = Path(a.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        p.error('GPU test requires a fresh empty output, never historical artifacts')
    if len(a.seeds) < 2:
        p.error('GPU model-reuse test requires at least two seeds')
    cmd = [sys.executable, ROOT / 'run_cofold.py']
    for name in ('image', 'inputs', 'assets', 'checkpoint', 'output', 'gpu', 'samples', 'recycles',
                 'steps', 'max_parallel_samples', 'output_format', 'cpus', 'memory', 'timeout', 'lock_root'):
        value = getattr(a, name)
        if value is not None:
            cmd += ['--' + name.replace('_', '-'), str(value)]
    cmd += ['--seeds', *map(str, a.seeds)]
    for name in ('no_kernels', 'use_potentials'):
        if getattr(a, name):
            cmd += ['--' + name.replace('_', '-')]
    run(cmd)
    verify_gpu(output, Path(a.inputs).expanduser().resolve(), Path(a.checkpoint).expanduser().resolve(),
               a.seeds, a.samples, a.output_format)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f'local test failed: {exc}', file=sys.stderr)
        sys.exit(1)

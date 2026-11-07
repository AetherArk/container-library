#!/usr/bin/env python3
"""Prepare immutable Boltz2 input objects and a PRIVATE launch payload; does not deploy."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tarfile
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent / 'cofolding/boltz2'))
from cloud_job import (InputTarInfo, MAX_ARCHIVE_BYTES, MAX_INPUT_BYTES, MAX_MEMBERS,
                       json_bytes, safe_name, token, validate_settings)
from resident.worker import Settings, sha


def read_config(path):
    """Accept shell KEY=value (not executed) or a flat JSON object."""
    text = Path(path).expanduser().read_text()
    if text.lstrip().startswith('{'):
        values = json.loads(text)
    else:
        values = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:]
            key, sep, value = line.partition('=')
            if not sep:
                raise ValueError('invalid R2 config')
            parts = shlex.split(value, comments=True)
            if len(parts) != 1:
                raise ValueError('invalid R2 config value')
            values[key.strip()] = parts[0]
    endpoint = values['AWS_ENDPOINT_URL']
    if not isinstance(endpoint, str) or not re.fullmatch(r'https://[A-Za-z0-9.-]+(?::[0-9]+)?/?', endpoint):
        raise ValueError('R2 endpoint must be an HTTPS origin')
    return {'endpoint_url': endpoint, 'aws_access_key_id': values.get('AWS_ACCESS_KEY_ID') or values['AWS_ACCESS_KEY'],
            'aws_secret_access_key': values['AWS_SECRET_ACCESS_KEY'], 'region_name': 'auto'}


def inspect_inputs(root):
    """Bundle top-level native YAML and its referenced relative dependencies unchanged."""
    import yaml
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('inputs must be a directory')
    files = set()
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('input links/special files are not allowed')
        if path.is_file():
            name = path.relative_to(root).as_posix()
            safe_name(name)
            files.add(name)
    sources = sorted(p for p in root.iterdir() if p.suffix.lower() in ('.yaml', '.yml') and p.is_file())
    if not sources:
        raise ValueError('no top-level native YAML inputs')
    if sum(p.stat().st_size for p in sources) > MAX_INPUT_BYTES:
        raise ValueError('native YAML inputs exceed size limit')

    used = {p.relative_to(root).as_posix() for p in sources}

    def walk(value, source):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ('msa', 'cif', 'pdb', 'path') and isinstance(child, str) and child not in ('empty', 'auto'):
                    p = Path(child)
                    if p.is_absolute():
                        raise ValueError('cloud dependencies must be relative to native YAML')
                    p = (source.parent / p).resolve()
                    if root not in p.parents or not p.is_file() or p.relative_to(root).as_posix() not in files:
                        raise ValueError('missing/out-of-tree native input dependency')
                    used.add(p.relative_to(root).as_posix())
                else:
                    walk(child, source)
        elif isinstance(value, list):
            for child in value:
                walk(child, source)
    for source in sources:
        data = yaml.safe_load(source.read_text())
        if not isinstance(data, dict) or 'seed' in data or 'seeds' in data or data.get('properties'):
            raise ValueError('expected structure-only native YAML with external seeds')
        walk(data, source)
    if len(used) > MAX_MEMBERS or sum((root / name).stat().st_size for name in used) > MAX_INPUT_BYTES:
        raise ValueError('input bundle exceeds limits')
    return root, {name: sha(root / name) for name in sorted(used)}


def private_json(path, value):
    # Exclusive create, never silently overwrite a capability file; independent of umask.
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def prepare(args, client=None):
    settings = {key: getattr(args, key) for key in asdict(Settings())}
    validate_settings(args.seeds, settings)
    for value in (args.project, args.experiment, args.bucket):
        token(value)
    if not re.fullmatch(r'[^\s?]+@sha256:[0-9a-f]{64}', args.image):
        raise ValueError('--image requires an immutable name@sha256 digest')
    if not 60 <= args.expires_in <= 604800:
        raise ValueError('--expires-in must be 60..604800 seconds (S3 presign limit)')
    if not 0 <= args.progress_slots <= 256 or not 10 <= args.progress_interval <= 3600:
        raise ValueError('progress slots must be 0..256 and interval 10..3600 seconds')
    root, files = inspect_inputs(args.inputs)
    output = Path(args.output).expanduser().resolve()
    # Never recursively ingest earlier launch secrets into a new input bundle.
    if output == root or root in output.parents:
        raise ValueError('preparation output must be outside inputs')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    now = datetime.now(timezone.utc)
    job_id = 'boltz2-' + now.strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:12]
    attempt_id = 'attempt-' + uuid.uuid4().hex
    prefix = f'{args.project}/{args.experiment}/{job_id}'
    result_prefix = prefix + '/results/' + attempt_id
    bundle = output / 'inputs.tar.gz'
    with tarfile.open(bundle, 'w:gz', format=tarfile.USTAR_FORMAT, dereference=False) as tar:
        # Files only; implicit directories avoid alias/duplicate paths.
        for name in files:
            tar.add(root / name, arcname='inputs/' + name, recursive=False)
    if bundle.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('compressed input exceeds download limit')
    # Verify the exact packaged bytes, not just the pre-read inventory.
    import hashlib
    with tarfile.open(bundle, 'r:gz', tarinfo=InputTarInfo) as tar:
        packaged = {m.name.removeprefix('inputs/'): hashlib.file_digest(tar.extractfile(m), 'sha256').hexdigest()
                    for m in tar}
    if packaged != files:
        raise ValueError('inputs changed during packaging')
    created_at = int(time.time())
    job = {'schema': 1, 'project': args.project, 'experiment': args.experiment, 'job_id': job_id,
           'attempt_id': attempt_id, 'created_at': created_at, 'expires_at': created_at + args.expires_in,
           'image': args.image, 'seeds': args.seeds, 'settings': settings, 'files': files,
           'bundle_sha256': sha(bundle), 'bundle_bytes': bundle.stat().st_size,
           'input_key': prefix + '/inputs/inputs.tar.gz', 'job_key': prefix + '/inputs/job.json',
           'result_key': result_prefix + '/results.tar', 'receipt_key': result_prefix + '/receipt.json',
           'claim_key': result_prefix + '/started.json'}
    job['progress_interval_seconds'] = args.progress_interval
    job['progress_keys'] = [result_prefix + f'/progress/{i:04d}.json' for i in range(args.progress_slots)]
    (output / 'job.json').write_bytes(json_bytes(job))
    if client is None:
        import boto3
        from botocore.config import Config
        # No credentials read until after local validation/packaging. Never pass them to the node.
        client = boto3.client('s3', **read_config(args.r2_config),
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'},
                                            request_checksum_calculation='when_required',
                                            response_checksum_validation='when_required',
                                            retries={'total_max_attempts': 1}))
    # Fail closed on collision; do not retry an ambiguous create-only write.
    with bundle.open('rb') as stream:
        client.put_object(Bucket=args.bucket, Key=job['input_key'], Body=stream, IfNoneMatch='*')
    client.put_object(Bucket=args.bucket, Key=job['job_key'], Body=json_bytes(job), IfNoneMatch='*')
    # Verify public objects before issuing node capabilities.
    for key, expected in ((job['input_key'], job['bundle_sha256']),
                          (job['job_key'], __import__('hashlib').sha256(json_bytes(job)).hexdigest())):
        response = client.get_object(Bucket=args.bucket, Key=key)
        body = response['Body']
        try:
            digest = __import__('hashlib').sha256()
            for block in iter(lambda: body.read(1024 * 1024), b''):
                digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError('R2 input upload readback hash mismatch')
        finally:
            body.close()
    remaining = job['expires_at'] - int(time.time())
    if remaining < 1:
        raise ValueError('expiration elapsed during host uploads')
    urls = {}
    for name, method, key in (('input_get', 'get_object', 'input_key'),
                              ('claim_put', 'put_object', 'claim_key'),
                              ('result_put', 'put_object', 'result_key'),
                              ('result_get', 'get_object', 'result_key'),
                              ('receipt_put', 'put_object', 'receipt_key'),
                              ('receipt_get', 'get_object', 'receipt_key')):
        params = {'Bucket': args.bucket, 'Key': job[key]}
        if method == 'put_object':
            params['IfNoneMatch'] = '*'
        urls[name] = client.generate_presigned_url(method, Params=params, ExpiresIn=remaining)
    progress_urls = []
    for key in job['progress_keys']:
        progress_urls.append({
            'put': client.generate_presigned_url('put_object', Params={'Bucket': args.bucket, 'Key': key, 'IfNoneMatch': '*'}, ExpiresIn=remaining),
            'get': client.generate_presigned_url('get_object', Params={'Bucket': args.bucket, 'Key': key}, ExpiresIn=remaining)})
    spec = {'job': job, 'urls': urls, 'progress_urls': progress_urls}
    environment = {'COFOLD_CLOUD_SPEC': json.dumps(spec, separators=(',', ':'))}
    private_json(output / 'runtime-env.secret.json', environment)
    private_json(output / 'deployment.secret.json', {'image': args.image,
                 'command': ['cofold-cloud-run'], 'environment_variables': environment})
    # This public command description has no URLs or credentials.
    (output / 'launch.json').write_bytes(json_bytes({'image': args.image, 'command': ['cofold-cloud-run'],
                  'environment_file': 'runtime-env.secret.json', 'deploys': False}))
    return job


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True, help='native top-level YAML + referenced relative-dependency tree; YAML bytes unchanged')
    p.add_argument('--output', type=Path, required=True, help='new private local preparation directory, outside inputs')
    p.add_argument('--bucket', required=True)
    p.add_argument('--r2-config', type=Path, default=Path('~/.r2'))
    p.add_argument('--project', default='salad-benchmarking')
    p.add_argument('--experiment', default='test')
    p.add_argument('--image', required=True, help='immutable image name@sha256:digest')
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--progress-slots', type=int, default=0, help='optional bounded durable snapshots, 0..256 create-only keys')
    p.add_argument('--progress-interval', type=int, default=60, help='seconds between snapshots, 10..3600')
    p.add_argument('--expires-in', type=int, default=86400, help='seconds including scheduling, inference, uploads and readback')
    for key, value in asdict(Settings()).items():
        flag = '--' + key.replace('_', '-')
        if type(value) is bool:
            p.add_argument(flag, action='store_true')
        elif type(value) is int:
            p.add_argument(flag, type=int, default=value)
        else:
            p.add_argument(flag, choices=('mmcif', 'pdb'), default=value)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        job = prepare(args)
        print(f"Prepared {job['job_id']} / {job['attempt_id']}; private launch files written. No container deployed.")
        return 0
    except Exception:
        # Provider exceptions can contain credentials/URLs. Do not print them.
        print('Preparation failed; details suppressed to protect credentials. Check inputs/configuration and use a fresh output directory.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

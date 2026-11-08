#!/usr/bin/env python3
"""Drive one ESMFold2 cloud attempt on a rented Vast.ai GPU.

The scientific worker still runs inside the published image, and every input and result byte
still travels through the prepared R2 prefix. This script only orchestrates the rental:

  prepare  build the input bundle, job.json and private capability files (host only)
  launch   create one Vast instance that boots the image with `cofold-cloud-run`
  watch    wait for the receipt in R2, download results.tar and verify its hash
  logs     stream the provider-side log for the attempt
  destroy  destroy the instance and show the price snapshot

Deviation worth stating plainly: the presigned capability bundle reaches the provider as the
instance environment variable COFOLD_CLOUD_SPEC, because the provider has no other secret channel.
It expires with the attempt, covers exactly one attempt prefix, and contains no R2 credentials.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_cloud_job as host  # noqa: E402


def run(argv, dry_run=False, capture=False):
    if dry_run:
        print('DRY-RUN $ ' + ' '.join(argv))
        return ''
    result = subprocess.run(argv, check=True, capture_output=capture, text=True)
    return result.stdout if capture else ''


def vast_json(argv, dry_run=False):
    output = run(['vastai'] + argv + ['--raw'], dry_run=dry_run, capture=not dry_run)
    return json.loads(output) if output else {}


def secrets_file(prepared):
    return Path(prepared) / 'runtime-env.secret.json'


def spec_environment(prepared):
    return json.loads(secrets_file(prepared).read_text())['COFOLD_CLOUD_SPEC']


def launch(prepared, offer_id, disk, label=None, dry_run=False):
    """Create the instance that boots the image and runs the cloud wrapper."""
    job = json.loads((Path(prepared) / 'job.json').read_text())
    environment = spec_environment(prepared)
    label = label or ('esmfold2-cloud-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    argv = ['create', 'instance', str(offer_id), '--image', job['image'], '--disk', str(disk),
            '--ssh', '--label', label, '--env', 'COFOLD_CLOUD_SPEC=' + environment,
            '--onstart-cmd', 'cofold-cloud-run']
    response = vast_json(argv, dry_run=dry_run)
    if dry_run:
        return {'label': label, 'job_id': job['job_id'], 'offer_id': offer_id}
    return {'label': label, 'job_id': job['job_id'], 'offer_id': offer_id,
            'contract': response.get('new_contract'), 'success': response.get('success')}


def instances(label):
    rows = vast_json(['show', 'instances'], dry_run=False)
    return [row for row in rows if row.get('label') == label]


def wait_running(label, timeout_seconds=900, interval=20, dry_run=False):
    deadline = time.monotonic() + timeout_seconds
    last = None
    while time.monotonic() < deadline:
        rows = instances(label) if not dry_run else []
        if rows:
            row = rows[0]
            last = (row.get('actual_status'), row.get('status_msg'))
            if row.get('actual_status') == 'running':
                return row
            if row.get('actual_status') in ('stopped', 'exited') and row.get('status_msg'):
                raise SystemExit('instance %s did not start: %s' % (rows[0]['id'], row['status_msg']))
        time.sleep(interval if not dry_run else 0)
        if dry_run:
            return {}
    raise SystemExit('instance for %s did not reach running state; last=%s' % (label, last))


def receipt_ready(prepared, client=None, bucket=None, r2_config=None):
    job = json.loads((Path(prepared) / 'job.json').read_text())
    client = client or client_from_config(r2_config)
    try:
        client.head_object(Bucket=bucket, Key=job['receipt_key'])
    except Exception:
        return None
    body = client.get_object(Bucket=bucket, Key=job['receipt_key'])['Body'].read()
    return json.loads(body)


def client_from_config(r2_config):
    """R2 credentials are read from the host config at use time, never stored with the job."""
    connection = host.read_config(r2_config or Path('~/.r2'))
    import boto3
    from botocore.config import Config
    return boto3.client('s3', **connection,
                        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'},
                                      request_checksum_calculation='when_required',
                                      response_checksum_validation='when_required',
                                      retries={'total_max_attempts': 2}))


def download_results(prepared, destination, bucket, client=None, r2_config=None):
    """Copy results.tar and verify size and sha256 against the receipt."""
    client = client or client_from_config(r2_config)
    job = json.loads((Path(prepared) / 'job.json').read_text())
    receipt = receipt_ready(prepared, client=client, bucket=bucket)
    if receipt is None:
        raise SystemExit('no receipt yet; nothing to download')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / 'results.tar'
    partial = destination / 'results.tar.partial'
    digest = hashlib.sha256()
    size = 0
    body = client.get_object(Bucket=bucket, Key=job['result_key'])['Body']
    try:
        with partial.open('xb') as stream:
            for block in iter(lambda: body.read(1024 * 1024), b''):
                digest.update(block)
                size += len(block)
                stream.write(block)
    finally:
        body.close()
    if size != receipt['result_bytes'] or digest.hexdigest() != receipt['result_sha256']:
        partial.unlink()
        raise SystemExit('downloaded archive does not match the receipt; kept nothing')
    os.replace(partial, target)
    (destination / 'receipt.json').write_text(json.dumps(receipt, indent=1, sort_keys=True) + '\n')
    return {'archive': str(target), 'status': receipt['status'],
            'sha256': digest.hexdigest(), 'bytes': size}


def destroy(contract):
    return run(['vastai', 'destroy', 'instance', str(contract)], capture=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)

    prep = sub.add_parser('prepare', help='write the bundle, job.json and private capability files')
    prep.add_argument('--inputs', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--bucket', required=True)
    prep.add_argument('--r2-config', type=Path, default=Path('~/.r2'))
    prep.add_argument('--project', default='esmfold2')
    prep.add_argument('--experiment', default='acceptance')
    prep.add_argument('--image', required=True, help='immutable image name@sha256:digest')
    prep.add_argument('--seeds', nargs='+', type=int, required=True)
    prep.add_argument('--expires-in', type=int, default=21600)

    for name, value in (('samples', 1), ('steps', 200), ('loops', 20), ('max-parallel-samples', 1)):
        prep.add_argument('--' + name, type=int, default=value)
    prep.add_argument('--backend', choices=('fused', 'reference'), default='fused')
    prep.add_argument('--output-format', choices=('mmcif',), default='mmcif')

    start = sub.add_parser('launch', help='create one instance for a prepared attempt')
    start.add_argument('--prepared', type=Path, required=True)
    start.add_argument('--offer-id', type=int, required=True)
    start.add_argument('--disk', type=int, default=120)
    start.add_argument('--label')
    start.add_argument('--wait', action='store_true', help='block until the instance reports running')
    start.add_argument('--dry-run', action='store_true')

    watch = sub.add_parser('watch', help='wait for the receipt, then download and verify results')
    watch.add_argument('--prepared', type=Path, required=True)
    watch.add_argument('--bucket', required=True)
    watch.add_argument('--destination', type=Path, required=True)
    watch.add_argument('--r2-config', type=Path, default=Path('~/.r2'))
    watch.add_argument('--timeout', type=int, default=7200)
    watch.add_argument('--interval', type=int, default=30)

    logs = sub.add_parser('logs', help='print provider logs for the labelled instance')
    logs.add_argument('--label', required=True)

    stop = sub.add_parser('destroy', help='destroy an instance and print its price snapshot')
    stop.add_argument('--instance', type=int, required=True)
    stop.add_argument('--confirm', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'prepare':
        settings = {'samples': args.samples, 'steps': args.steps, 'loops': args.loops,
                    'backend': args.backend, 'max_parallel_samples': args.max_parallel_samples,
                    'output_format': args.output_format}
        host_settings = argparse.Namespace(**settings)  # prepare() creates the 0700 directory
        job = host.prepare(argparse.Namespace(
            inputs=args.inputs, output=args.output, bucket=args.bucket, r2_config=args.r2_config,
            project=args.project, experiment=args.experiment, image=args.image, seeds=args.seeds,
            progress_slots=0, progress_interval=60, expires_in=args.expires_in, **vars(host_settings)))
        print(json.dumps({'job_id': job['job_id'], 'attempt_id': job['attempt_id'],
                          'image': job['image'], 'expires_at': job['expires_at']}))
        return 0
    if args.command == 'launch':
        result = launch(args.prepared, args.offer_id, args.disk, args.label, args.dry_run)
        if args.wait and not args.dry_run:
            row = wait_running(result['label'])
            result['instance'] = row.get('id')
            result['ssh'] = '%s:%s' % (row.get('ssh_host'), row.get('ssh_port'))
        print(json.dumps(result))
        return 0
    if args.command == 'watch':
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            receipt = receipt_ready(args.prepared, bucket=args.bucket, r2_config=args.r2_config)
            if receipt is not None:
                summary = download_results(args.prepared, args.destination, args.bucket,
                                           r2_config=args.r2_config)
                print(json.dumps(summary, sort_keys=True))
                return 0 if summary['status'] == 'completed' else 1
            time.sleep(args.interval)
        raise SystemExit('no receipt within %d seconds' % args.timeout)
    if args.command == 'logs':
        rows = instances(args.label)
        if not rows:
            raise SystemExit('no instance with label ' + args.label)
        print(run(['vastai', 'logs', str(rows[0]['id'])], capture=True))
        return 0
    if args.command == 'destroy':
        rows = vast_json(['show', 'instance', str(args.instance)])
        snapshot = {key: rows.get(key) for key in ('id', 'gpu_name', 'dph_base', 'dph_total',
                                                   'storage_cost', 'label', 'actual_status')}
        print(json.dumps({'price_snapshot': snapshot}))
        if not args.confirm:
            print('re-run with --confirm to destroy', file=sys.stderr)
            return 2
        print(destroy(args.instance).strip())
        return 0
    raise SystemExit('unknown command')


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify the pinned ESMFold2 checkpoint; download only with explicit --download.

Offline mode (the default) never touches the network: it re-hashes every file in
the target directory against assets.json and fails closed on any mismatch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, spec):
    if path.is_symlink():
        raise ValueError('Refusing to verify a symlink; materialize the file: ' + str(path))
    if not path.is_file():
        raise ValueError('Missing asset file: ' + str(path))
    if path.stat().st_size != spec['bytes']:
        raise ValueError('Incorrect size: ' + str(path))
    actual = sha256(path)
    if actual != spec['sha256']:
        raise ValueError('Checksum mismatch: ' + str(path) + ' (' + actual + ')')


def download(url, target, spec):
    partial = target.with_name(target.name + '.partial')
    if partial.exists() or partial.is_symlink():
        raise ValueError('Partial download exists; inspect/remove before retry: ' + str(partial))
    print('Downloading ' + url, flush=True)
    with urllib.request.urlopen(url, timeout=120) as source, partial.open('xb') as destination:
        while chunk := source.read(1024 * 1024):
            destination.write(chunk)
    try:
        verify(partial, spec)
    except ValueError:
        partial.unlink()
        raise
    partial.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True,
                        help='checkpoint directory; receives the pinned files at its top level')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--download', action='store_true',
                      help='fetch missing files from the pinned revision (network)')
    mode.add_argument('--offline', action='store_true',
                      help='explicit default: verify only, never open a network connection')
    parser.add_argument('--json', action='store_true', help='print the verified inventory as JSON')
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('assets.json').read_text())
    assets = args.assets.expanduser().resolve()
    if args.download:
        assets.mkdir(parents=True, exist_ok=True)
    elif not assets.is_dir():
        raise SystemExit('Asset directory does not exist (offline verification): ' + str(assets))
    inventory = {}
    for name, spec in sorted(manifest['files'].items()):
        target = assets / name
        if args.download and not target.exists():
            url = 'https://huggingface.co/{}/resolve/{}/{}'.format(
                manifest['repository'], manifest['revision'], name)
            download(url, target, spec)
        verify(target, spec)
        inventory[name] = {'bytes': spec['bytes'], 'sha256': spec['sha256']}
        print('Verified ' + name, flush=True)
    if args.json:
        print(json.dumps({'repository': manifest['repository'],
                          'revision': manifest['revision'],
                          'assets': str(assets), 'files': inventory}, indent=2, sort_keys=True))
    else:
        print('Assets ready: {} files, revision {}'.format(len(inventory), manifest['revision']))


if __name__ == '__main__':
    main()

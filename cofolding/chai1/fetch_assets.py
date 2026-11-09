#!/usr/bin/env python3
"""Verify the pinned Chai-1 assets; download only with explicit --download.

Offline mode (the default) never touches the network: it re-hashes every file named in
assets.json against its pinned sha256 and fails closed on any mismatch.  The manifest names
the exact relative path chai_lab resolves under ``CHAI_DOWNLOADS_DIR``, so a populated
directory can be mounted read-only at ``--assets`` with no network path at inference time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

DEFAULT_MANIFEST = Path(__file__).with_name('assets.json')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, spec):
    """Hash the file's content; a symlink is followed rather than rejected."""
    path = Path(path)
    if not path.exists():
        raise ValueError('Missing asset file: ' + str(path))
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError('Not a regular file: ' + str(path))
    if not isinstance(spec.get('sha256'), str) or len(spec['sha256']) != 64:
        raise ValueError('Manifest entry has no pinned sha256: ' + str(path))
    size = resolved.stat().st_size
    if size != spec['bytes']:
        raise ValueError('Incorrect size: %s (%d != %d)' % (path, size, spec['bytes']))
    actual = sha256(resolved)
    if actual != spec['sha256']:
        raise ValueError('Checksum mismatch: ' + str(path) + ' (' + actual + ')')
    return {'bytes': size, 'sha256': actual,
            'resolved': None if resolved == Path(path) else str(resolved)}


def download(url, target, spec):
    partial = target.with_name(target.name + '.partial')
    if partial.exists() or partial.is_symlink():
        raise ValueError('Partial download exists; inspect/remove before retry: ' + str(partial))
    target.parent.mkdir(parents=True, exist_ok=True)
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


def run(assets, download_files=False, manifest_path=DEFAULT_MANIFEST, as_json=False):
    manifest = json.loads(Path(manifest_path).read_text())
    files = manifest['files']
    if not isinstance(files, dict) or not files:
        raise ValueError('asset manifest has no files')
    assets = Path(assets).expanduser().resolve()
    if download_files:
        assets.mkdir(parents=True, exist_ok=True)
    elif not assets.is_dir():
        raise ValueError('Asset directory does not exist (offline verification): ' + str(assets))
    inventory = {}
    for name in sorted(files):
        spec = files[name]
        target = assets / name
        if download_files and not target.exists():
            download(spec['url'], target, spec)
        inventory[name] = verify(target, spec)
        print('Verified ' + name, flush=True)
    summary = {'source': manifest.get('source'), 'assets': str(assets), 'files': inventory}
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print('Assets ready: {} files at {}'.format(len(inventory), assets))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True,
                        help='directory that will be mounted as CHAI_DOWNLOADS_DIR')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--download', action='store_true',
                      help='fetch missing files from the pinned URLs (network)')
    mode.add_argument('--offline', action='store_true',
                      help='explicit default: verify only, never open a network connection')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST,
                        help='asset manifest (defaults to assets.json next to this script)')
    parser.add_argument('--json', action='store_true', help='print the verified inventory as JSON')
    args = parser.parse_args(argv)
    try:
        run(args.assets, args.download, args.manifest, args.json)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))


if __name__ == '__main__':
    main()

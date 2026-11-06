#!/usr/bin/env python3
"""Verify pinned model assets; download only with explicit --download."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request


def verify(path, spec):
    if not path.is_file() or path.stat().st_size != spec['bytes']:
        raise ValueError('Missing or incorrect size: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != spec['sha256']:
        raise ValueError('Checksum mismatch: ' + str(path))


def extract(archive, cache):
    # Refuse links, devices, absolute paths, and path traversal before extraction.
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if not parts or parts[0] != 'mols' or '..' in parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive member: ' + member.name)
        tar.extractall(cache, filter='data')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--download', action='store_true')
    a = p.parse_args()
    manifest = json.loads(Path(__file__).with_name('assets.json').read_text())
    cache = a.cache.expanduser().resolve()
    if a.download:
        cache.mkdir(parents=True, exist_ok=True)
    for name, spec in manifest['files'].items():
        target = cache / name
        if not target.exists() and a.download:
            url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{name}"
            partial = target.with_suffix(target.suffix + '.partial')
            if partial.exists():
                raise ValueError('Partial download exists; inspect/remove before retry: ' + str(partial))
            print('Downloading ' + url, flush=True)
            with urllib.request.urlopen(url, timeout=120) as src, partial.open('xb') as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            verify(partial, spec)
            partial.replace(target)
        verify(target, spec)
        print('Verified ' + name, flush=True)
    if not (cache / 'mols').is_dir():
        if not a.download:
            raise ValueError('Missing extracted mols directory; use --download to extract verified archive')
        extract(cache / 'mols.tar', cache)
    print('Assets ready')


if __name__ == '__main__':
    main()

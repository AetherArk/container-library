#!/usr/bin/env python3
"""Verify pinned model assets; download only with explicit --download."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
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


def mols_ready(cache, archive_sha256):
    marker = cache / 'mols' / '.complete.json'
    try:
        return json.loads(marker.read_text()) == {'archive_sha256': archive_sha256}
    except (OSError, ValueError):
        return False


def extract(archive, cache):
    target = cache / 'mols'
    if target.exists() or target.is_symlink():
        raise ValueError('Unverified mols directory exists; use a fresh cache or inspect/move it before extraction')
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    with tempfile.TemporaryDirectory(prefix='.mols-staging-', dir=cache) as tmp:
        staging = Path(tmp)
        # Refuse links, devices, absolute paths, and traversal before extraction.
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if not members:
                raise ValueError('Empty archive')
            for member in members:
                parts = Path(member.name).parts
                if not parts or parts[0] != 'mols' or '..' in parts or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe archive member: ' + member.name)
            tar.extractall(staging, filter='data')
        extracted = staging / 'mols'
        if not extracted.is_dir():
            raise ValueError('Archive did not produce mols directory')
        (extracted / '.complete.json').write_text(json.dumps({'archive_sha256': h.hexdigest()}))
        # Directory and completion marker become visible together on this filesystem.
        extracted.rename(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--download', action='store_true')
    p.add_argument('--structure-only', action='store_true', help='omit unused affinity weights; default still verifies all assets')
    a = p.parse_args()
    manifest = json.loads(Path(__file__).with_name('assets.json').read_text())
    cache = a.cache.expanduser().resolve()
    if a.download:
        cache.mkdir(parents=True, exist_ok=True)
    for name, spec in manifest['files'].items():
        if a.structure_only and name == 'boltz2_aff.ckpt':
            continue
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
    if not mols_ready(cache, manifest['files']['mols.tar']['sha256']):
        if (cache / 'mols').exists() or (cache / 'mols').is_symlink():
            raise ValueError('Unverified/incomplete mols directory; use a fresh cache or inspect/move the existing directory')
        if not a.download:
            raise ValueError('Missing verified mols directory; use --download to extract the verified archive')
        extract(cache / 'mols.tar', cache)
    print('Assets ready')


if __name__ == '__main__':
    main()

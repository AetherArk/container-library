#!/usr/bin/env python3
"""Verify the pinned ESMFold2 checkpoint; download only with explicit --download.

Offline mode (the default) never touches the network: it re-hashes every file in
the target directory against assets.json and fails closed on any mismatch.

Two layouts are accepted, because both occur in practice:

* flat: the pinned files sit directly in the target directory;
* Hugging Face cache: ``<assets>/huggingface/hub/models--<org>--<repo>/snapshots/<rev>``
  holds symlinks into ``blobs/``. Symlinks are followed and the target bytes are
  hashed, so the recorded checksum still decides acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

DEFAULT_MANIFEST = Path(__file__).with_name('assets.json')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def locate(assets, names, repository=None, revision=None):
    """Return the directory that holds the pinned files, or raise with the paths tried."""
    assets = Path(assets)
    candidates = [assets]
    if repository and revision:
        organisation, _, project = repository.partition('/')
        candidates.append(assets / 'huggingface' / 'hub' /
                          ('models--%s--%s' % (organisation, project)) / 'snapshots' / revision)
    candidates.extend(sorted(assets.glob('huggingface/hub/models--*/snapshots/*')))
    seen, tried = set(), []
    for candidate in candidates:
        key = str(candidate)
        if key in seen or not candidate.is_dir():
            continue
        seen.add(key)
        tried.append(key)
        if all((candidate / name).exists() for name in names):
            return candidate
    raise ValueError('No directory holds all pinned files; tried: ' + ', '.join(tried or [str(assets)]))


def verify(path, spec):
    """Hash the file's content; a symlink is followed rather than rejected."""
    path = Path(path)
    if not path.exists():
        raise ValueError('Missing asset file: ' + str(path))
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError('Not a regular file: ' + str(path))
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
    names = sorted(manifest['files'])
    assets = Path(assets).expanduser().resolve()
    if download_files:
        assets.mkdir(parents=True, exist_ok=True)
        root = assets
    else:
        if not assets.is_dir():
            raise ValueError('Asset directory does not exist (offline verification): ' + str(assets))
        root = locate(assets, names, manifest.get('repository'), manifest.get('revision'))
    inventory = {}
    for name in names:
        spec = manifest['files'][name]
        target = root / name
        if download_files and not target.exists():
            url = 'https://huggingface.co/{}/resolve/{}/{}'.format(
                manifest['repository'], manifest['revision'], name)
            download(url, target, spec)
        inventory[name] = verify(target, spec)
        print('Verified ' + name, flush=True)
    summary = {'repository': manifest['repository'], 'revision': manifest['revision'],
               'assets': str(assets), 'root': str(root),
               'layout': 'flat' if root == assets else 'huggingface', 'files': inventory}
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print('Assets ready: {} files at {}, revision {}'.format(
            len(inventory), root, manifest['revision']))
    return summary


def materialize(assets, target, manifest_path=DEFAULT_MANIFEST):
    """Write a flat directory of real files (hardlinks when possible) for container mounts.

    A Hugging Face snapshot is a set of symlinks into ``blobs/``; mounting only the snapshot
    directory leaves them dangling. This copies or hardlinks the verified bytes into one flat
    directory that can be mounted on its own.
    """
    summary = run(assets, False, manifest_path, False)
    source = Path(summary['root'])
    target = Path(target).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text())
    for name, spec in sorted(manifest['files'].items()):
        destination = target / name
        if destination.exists():
            verify(destination, spec)
            print('Present ' + name, flush=True)
            continue
        resolved = (source / name).resolve()
        partial = destination.with_name(destination.name + '.partial')
        if partial.exists():
            partial.unlink()
        try:
            os.link(resolved, partial)
            mode = 'hardlink'
        except OSError:
            shutil.copyfile(resolved, partial)
            mode = 'copy'
        verify(partial, spec)
        partial.replace(destination)
        print('Materialized %s (%s)' % (name, mode), flush=True)
    print('Flat checkpoint ready: {} files at {}'.format(len(manifest['files']), target))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True,
                        help='checkpoint directory, or a Hugging Face cache root holding the snapshot')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--download', action='store_true',
                      help='fetch missing files from the pinned revision (network)')
    mode.add_argument('--offline', action='store_true',
                      help='explicit default: verify only, never open a network connection')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST,
                        help='asset manifest (defaults to assets.json next to this script)')
    parser.add_argument('--materialize', type=Path, metavar='DIR',
                        help='verify, then write a flat directory of real files for a container mount')
    parser.add_argument('--json', action='store_true', help='print the verified inventory as JSON')
    args = parser.parse_args(argv)
    try:
        if args.materialize:
            materialize(args.assets, args.materialize, args.manifest)
        else:
            run(args.assets, args.download, args.manifest, args.json)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))


if __name__ == '__main__':
    main()

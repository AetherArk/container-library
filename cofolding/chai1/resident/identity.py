"""Byte identity helpers with no ML imports, shared by the worker and the engine."""
import hashlib
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def directory_identity(path):
    """Hash a checkpoint/model directory once; the per-file hashes are reused by jobs."""
    path = Path(path)
    digest = hashlib.sha256()
    per_file = {}
    for item in sorted(path.rglob('*')):
        if item.is_file():
            relative = str(item.relative_to(path))
            file_hash = sha(item)
            per_file[relative] = {'sha256': file_hash, 'bytes': item.stat().st_size}
            digest.update(relative.encode())
            digest.update(b'\0')
            digest.update(bytes.fromhex(file_hash))
            digest.update(b'\0')
    return {'kind': 'directory', 'sha256': digest.hexdigest(), 'files': per_file}

"""fetch_assets.py against flat and Hugging Face cache layouts (tiny synthetic manifest)."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'fetch_assets.py'
REPOSITORY = 'example/model'
REVISION = 'a' * 40
PAYLOAD = {'config.json': b'{"model_type": "fixture"}', 'ccd.pkl': b'\x80\x04fixture'}


class AssetsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / 'assets.json'
        self.manifest.write_text(json.dumps({
            'repository': REPOSITORY, 'revision': REVISION,
            'files': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                      for name, data in PAYLOAD.items()}}))
        self.counter = 0

    def call(self, assets, *extra):
        self.counter += 1
        out = self.root / ('out%d.json' % self.counter)
        proc = subprocess.run([sys.executable, str(SCRIPT), '--assets', str(assets),
                               '--offline', '--manifest', str(self.manifest),
                               '--json', *extra], capture_output=True, text=True)
        return proc, out

    def test_flat_layout_verifies(self):
        flat = self.root / 'flat'
        flat.mkdir()
        for name, data in PAYLOAD.items():
            (flat / name).write_bytes(data)
        proc, _ = self.call(flat)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads(proc.stdout[proc.stdout.index('{'):])
        self.assertEqual(summary['layout'], 'flat')
        self.assertEqual(sorted(summary['files']), sorted(PAYLOAD))

    def test_huggingface_cache_layout_verifies_symlinks(self):
        cache = self.root / 'cache'
        blobs = cache / 'huggingface' / 'hub' / 'models--example--model' / 'blobs'
        snapshot = cache / 'huggingface' / 'hub' / 'models--example--model' / 'snapshots' / REVISION
        blobs.mkdir(parents=True)
        snapshot.mkdir(parents=True)
        for name, data in PAYLOAD.items():
            blob = blobs / hashlib.sha256(data).hexdigest()
            blob.write_bytes(data)
            (snapshot / name).symlink_to(Path('../../blobs') / blob.name)
        proc, _ = self.call(cache)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads(proc.stdout[proc.stdout.index('{'):])
        self.assertEqual(summary['layout'], 'huggingface')
        self.assertEqual(summary['root'], str(snapshot.resolve()))
        self.assertTrue(all(record['resolved'] for record in summary['files'].values()))

    def test_corrupted_blob_behind_a_symlink_fails(self):
        cache = self.root / 'cache'
        blobs = cache / 'huggingface' / 'hub' / 'models--example--model' / 'blobs'
        snapshot = cache / 'huggingface' / 'hub' / 'models--example--model' / 'snapshots' / REVISION
        blobs.mkdir(parents=True)
        snapshot.mkdir(parents=True)
        for name, data in PAYLOAD.items():
            blob = blobs / hashlib.sha256(data).hexdigest()
            blob.write_bytes(data + b'corruption' if name == 'ccd.pkl' else data)
            (snapshot / name).symlink_to(Path('../../blobs') / blob.name)
        proc, _ = self.call(cache)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('Incorrect size', proc.stderr + proc.stdout)

    def test_missing_asset_is_named(self):
        flat = self.root / 'flat'
        flat.mkdir()
        (flat / 'config.json').write_bytes(PAYLOAD['config.json'])
        proc, _ = self.call(flat)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('No directory holds all pinned files', proc.stderr)

    def test_absent_directory_fails_closed(self):
        proc, _ = self.call(self.root / 'absent')
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('does not exist', proc.stderr)


if __name__ == '__main__':
    unittest.main()

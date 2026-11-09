"""fetch_assets.py against the pinned flat layout (tiny synthetic manifest)."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'fetch_assets.py'
PAYLOAD = {'models_v2/trunk.pt': b'fixture-trunk', 'conformers_v1.apkl': b'\x80\x04fixture'}


def manifest_for(payload, sha_overrides=None):
    sha_overrides = sha_overrides or {}
    return {'source': 'https://example.invalid',
            'files': {name: {'url': 'https://example.invalid/' + name,
                             'bytes': len(data),
                             'sha256': sha_overrides.get(name, hashlib.sha256(data).hexdigest())}
                      for name, data in payload.items()}}


class AssetsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / 'assets.json'
        self.manifest.write_text(json.dumps(manifest_for(PAYLOAD)))

    def call(self, assets, *extra):
        return subprocess.run([sys.executable, str(SCRIPT), '--assets', str(assets),
                               '--offline', '--manifest', str(self.manifest), '--json', *extra],
                              capture_output=True, text=True)

    def test_flat_layout_verifies(self):
        flat = self.root / 'flat'
        flat.mkdir()
        for name, data in PAYLOAD.items():
            (flat / name).parent.mkdir(parents=True, exist_ok=True)
            (flat / name).write_bytes(data)
        proc = self.call(flat)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads(proc.stdout[proc.stdout.index('{'):])
        self.assertEqual(sorted(summary['files']), sorted(PAYLOAD))

    def test_corrupted_file_fails(self):
        flat = self.root / 'flat'
        flat.mkdir()
        for name, data in PAYLOAD.items():
            (flat / name).parent.mkdir(parents=True, exist_ok=True)
            (flat / name).write_bytes(data + b'corruption' if name.endswith('trunk.pt') else data)
        proc = self.call(flat)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('Incorrect size', proc.stderr + proc.stdout)

    def test_same_size_wrong_bytes_fails(self):
        flat = self.root / 'flat'
        flat.mkdir()
        (flat / 'models_v2').mkdir()
        (flat / 'models_v2' / 'trunk.pt').write_bytes(b'x' * len(PAYLOAD['models_v2/trunk.pt']))
        (flat / 'conformers_v1.apkl').write_bytes(PAYLOAD['conformers_v1.apkl'])
        proc = self.call(flat)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('Checksum mismatch', proc.stderr + proc.stdout)

    def test_missing_asset_is_named(self):
        flat = self.root / 'flat'
        flat.mkdir()
        (flat / 'models_v2').mkdir()
        (flat / 'models_v2' / 'trunk.pt').write_bytes(PAYLOAD['models_v2/trunk.pt'])
        proc = self.call(flat)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('Missing asset file', proc.stderr)

    def test_unpinned_sha256_is_refused(self):
        manifest = manifest_for(PAYLOAD, {'conformers_v1.apkl': None})
        self.manifest.write_text(json.dumps(manifest))
        flat = self.root / 'flat'
        flat.mkdir()
        for name, data in PAYLOAD.items():
            (flat / name).parent.mkdir(parents=True, exist_ok=True)
            (flat / name).write_bytes(data)
        proc = self.call(flat)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('no pinned sha256', proc.stderr)

    def test_absent_directory_fails_closed(self):
        proc = self.call(self.root / 'absent')
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('does not exist', proc.stderr)


if __name__ == '__main__':
    unittest.main()

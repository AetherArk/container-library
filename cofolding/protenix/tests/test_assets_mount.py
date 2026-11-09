"""The two mounts the launcher hands Protenix must be accepted when they agree.

The launcher always mounts the asset directory at /models and the checkpoint at
/checkpoint, so inside the container the paths differ even when the caller passed
the same host directory. Protenix itself resolves everything from one
PROTENIX_ROOT_DIR, so the engine must decide on content rather than path.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resident import engine as engine_module


def make_root(root, payload=b'x'):
    root = Path(root)
    (root / 'checkpoint').mkdir(parents=True)
    (root / 'common').mkdir(parents=True)
    (root / 'checkpoint' / engine_module.CHECKPOINT_NAME).write_bytes(payload)
    for name in engine_module.PROTENIX_COMMON:
        (root / 'common' / name).write_bytes(b'y')
    return root


def engine_with(assets, checkpoint):
    engine = object.__new__(engine_module.ProtenixEngine)
    engine.assets = Path(assets)
    engine.checkpoint = Path(checkpoint)
    engine.checkpoint_files = None
    return engine


class CheckpointMountTests(unittest.TestCase):
    def test_identical_content_at_two_paths_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets = make_root(Path(tmp) / 'models')
            checkpoint = make_root(Path(tmp) / 'checkpoint')
            engine_with(assets, checkpoint).check_checkpoint()

    def test_same_path_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_root(Path(tmp) / 'root')
            engine_with(root, root).check_checkpoint()

    def test_checkpoint_mount_missing_files_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets = make_root(Path(tmp) / 'models')
            checkpoint = Path(tmp) / 'checkpoint'
            checkpoint.mkdir()
            with self.assertRaises(ValueError) as caught:
                engine_with(assets, checkpoint).check_checkpoint()
            message = str(caught.exception)
            self.assertIn(engine_module.CHECKPOINT_NAME, message)
            self.assertIn('checkpoint', message)

    def test_checkpoint_mount_with_a_different_weight_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets = make_root(Path(tmp) / 'models', payload=b'x')
            checkpoint = make_root(Path(tmp) / 'checkpoint', payload=b'a much longer payload')
            with self.assertRaises(ValueError):
                engine_with(assets, checkpoint).check_checkpoint()


if __name__ == '__main__':
    unittest.main()

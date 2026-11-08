"""The per-job protocol must name the exact engine call, including inherited defaults."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resident.engine import ESMFold2Engine, fold_record  # noqa: E402
from resident.worker import Settings  # noqa: E402


def sample_fold(self, model, input, *, num_loops=20, num_sampling_steps=200,
                num_diffusion_samples=1, seed=None, noise_scale=None, lm_dropout=0.3,
                include_embeddings=False, complex_id='pred'):
    return None


class ProtocolTests(unittest.TestCase):
    def test_fold_record_separates_overrides_from_native_defaults(self):
        record = fold_record(sample_fold)
        self.assertEqual(record['explicit'], ['num_loops', 'num_sampling_steps',
                                              'num_diffusion_samples', 'seed',
                                              'include_embeddings', 'complex_id'])
        self.assertEqual(record['native_defaults'], {'noise_scale': None, 'lm_dropout': 0.3})

    @unittest.skipUnless(importlib.util.find_spec('esm'), 'prepare needs the esm package (present in the image)')
    def test_prepare_writes_the_call_record(self):
        for name in ('trpcage-ethanol', 'cyp'):
            with self.subTest(example=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                engine = SampleEngine(root, Settings())
                attempt = root / 'attempt'
                attempt.mkdir()
                prepared = engine.prepare(ROOT / 'examples' / (name + '.yaml'), attempt, 11)
                protocol = json.loads((attempt / 'protocol.json').read_text())
                chemistry = json.loads((attempt / 'chemistry.json').read_text())
            self.assertEqual(protocol['sample_seeds'], [11000])
            self.assertIn('lm_dropout', protocol['fold_call']['native_defaults'])
            self.assertNotIn('num_loops', protocol['fold_call']['native_defaults'])
            self.assertEqual(prepared['sample_seeds'], [11000])
            self.assertIsNotNone(chemistry['ligand'])
            self.assertEqual(chemistry['heme'] is not None, name == 'cyp')


class SampleEngine(ESMFold2Engine):
    """Engine that reuses the native preparation code without touching a checkpoint."""

    def __init__(self, root, settings):
        super().__init__(root / 'assets', root / 'checkpoint', settings)
        self.fold_call = fold_record(sample_fold)


if __name__ == '__main__':
    unittest.main()

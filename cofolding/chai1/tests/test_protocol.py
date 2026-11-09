"""The per-job protocol must name the exact engine call, including inherited defaults."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resident.engine import Chai1Engine, fold_record  # noqa: E402
from resident.worker import Settings  # noqa: E402


def sample_fold(feature_context, *, output_dir, recycle_msa_subsample=0,
                num_trunk_recycles=3, num_diffn_timesteps=200, num_diffn_samples=5,
                seed=None, device=None, low_memory=True):
    return None


class ProtocolTests(unittest.TestCase):
    def test_fold_record_separates_overrides_from_native_defaults(self):
        record = fold_record(sample_fold)
        self.assertEqual(record['explicit'], ['output_dir', 'num_trunk_recycles',
                                              'num_diffn_timesteps', 'num_diffn_samples',
                                              'seed', 'device', 'low_memory'])
        self.assertEqual(record['native_defaults'], {'recycle_msa_subsample': 0})

    def test_prepare_writes_the_call_record_and_native_fasta(self):
        for name in ('trpcage-ethanol', 'two-chain', 'small-two-ligand'):
            with self.subTest(example=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                engine = Chai1Engine(root / 'assets', root / 'checkpoint', Settings())
                engine.fold_call = fold_record(sample_fold)
                attempt = root / 'attempt'
                attempt.mkdir()
                prepared = engine.prepare(ROOT / 'examples' / (name + '.yaml'), attempt, 11)
                protocol = json.loads((attempt / 'protocol.json').read_text())
                chemistry = json.loads((attempt / 'chemistry.json').read_text())
                fasta = (attempt / 'inputs' / (name + '.fasta')).read_text()
                self.assertEqual(protocol['sample_seeds'], [11000])
                self.assertEqual(protocol['settings']['num_trunk_recycles'], 3)
                self.assertEqual(protocol['settings']['num_diffn_timesteps'], 200)
                self.assertEqual(config := protocol['fold_call']['native_defaults'],
                                 {'recycle_msa_subsample': 0})
                self.assertNotIn('num_trunk_recycles', config)
                self.assertEqual(prepared['sample_seeds'], [11000])
                self.assertEqual(fasta.splitlines()[0], '>protein|A')
                self.assertEqual(chemistry['chains'][0]['cif_chain'], 'A')
            if name == 'trpcage-ethanol':
                self.assertEqual([c['kind'] for c in chemistry['chains']], ['protein', 'ligand'])
            if name == 'small-two-ligand':
                self.assertEqual([c['kind'] for c in chemistry['chains']],
                                 ['protein', 'ligand', 'ligand'])

    def test_sample_seed_formula(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = Chai1Engine(root / 'assets', root / 'checkpoint', Settings(samples=3))
            attempt = root / 'attempt'
            attempt.mkdir()
            prepared = engine.prepare(ROOT / 'examples' / 'trpcage-ethanol.yaml', attempt, 12)
        self.assertEqual(prepared['sample_seeds'], [12000, 12001, 12002])


if __name__ == '__main__':
    unittest.main()

"""The per-job protocol must name the exact configuration overrides and the native JSON input."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resident.engine import CONFIG_OVERRIDES, ProtenixEngine, call_record  # noqa: E402
from resident.worker import Settings  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_call_record_separates_overrides_from_inherited_defaults(self):
        record = call_record(Settings(steps=64, recycles=4))
        self.assertEqual(record['explicit'], list(CONFIG_OVERRIDES))
        # Inherited defaults are the pinned wheel's, and the overrides are the ones this
        # container always passes.
        self.assertEqual(record['inherited']['sample_diffusion.N_sample'], 5)
        self.assertEqual(record['native_defaults']['dtype'], 'bf16')
        # A key this container overrides is not also reported as an inherited default.
        self.assertNotIn('sample_diffusion.N_sample', record['native_defaults'])
        self.assertEqual(record['overrides']['sample_diffusion.N_sample'], 1)
        self.assertEqual(record['overrides']['sample_diffusion.N_step'], 64)
        self.assertEqual(record['overrides']['model.N_cycle'], 4)
        self.assertFalse(record['overrides']['use_msa'])
        self.assertFalse(record['overrides']['use_template'])

    def test_prepare_writes_the_call_record_and_native_json(self):
        for name in ('trpcage-ethanol', 'two-chain', 'small-two-ligand'):
            with self.subTest(example=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                engine = ProtenixEngine(root / 'assets', root / 'checkpoint', Settings())
                attempt = root / 'attempt'
                attempt.mkdir()
                prepared = engine.prepare(ROOT / 'examples' / (name + '.yaml'), attempt, 11)
                protocol = json.loads((attempt / 'protocol.json').read_text())
                chemistry = json.loads((attempt / 'chemistry.json').read_text())
                document = json.loads((attempt / 'inputs' / (name + '.json')).read_text())
                self.assertEqual(protocol['model_name'], 'protenix-v2')
                self.assertEqual(protocol['sample_seeds'], [11000])
                self.assertEqual(protocol['settings']['model.N_cycle'], 10)
                self.assertEqual(protocol['settings']['sample_diffusion.N_step'], 200)
                self.assertEqual(protocol['call']['overrides']['model.N_cycle'], 10)
                self.assertFalse(protocol['msa'])
                self.assertFalse(protocol['templates'])
                self.assertEqual(prepared['sample_seeds'], [11000])
                self.assertEqual(document[0]['name'], name)
                self.assertEqual(document[0]['sequences'][0]['proteinChain']['id'], ['A'])
                self.assertEqual(chemistry['chains'][0]['cif_chain'], 'A')
                self.assertEqual(chemistry['chains'][0]['residues'], len('NLYIQWLKDGGPSSGRPPPS'))
                if name == 'trpcage-ethanol':
                    self.assertEqual([c['kind'] for c in chemistry['chains']],
                                     ['protein', 'ligand'])
                    self.assertEqual(chemistry['chains'][1]['cif_chain'], 'B')
                if name == 'small-two-ligand':
                    self.assertEqual([c['kind'] for c in chemistry['chains']],
                                     ['protein', 'ligand', 'ligand'])
                    self.assertEqual([c['cif_chain'] for c in chemistry['chains']],
                                     ['A', 'B', 'C'])

    def test_sample_seed_formula(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = ProtenixEngine(root / 'assets', root / 'checkpoint', Settings(samples=3))
            attempt = root / 'attempt'
            attempt.mkdir()
            prepared = engine.prepare(ROOT / 'examples' / 'trpcage-ethanol.yaml', attempt, 12)
        self.assertEqual(prepared['sample_seeds'], [12000, 12001, 12002])


if __name__ == '__main__':
    unittest.main()

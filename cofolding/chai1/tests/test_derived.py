"""CPU-only tests for derived interface metrics written by the Chai-1 worker."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

from resident import engine as chai_engine  # noqa: E402
from resident.worker import run, Settings  # noqa: E402
from cofolding.common import derived  # noqa: E402


class FakeEngine:
    token_policy = 'good'
    derive_calls = 0

    def __init__(self, assets, checkpoint, settings):
        self.settings = settings
        self.loaded = False

    def prepare(self, source, attempt, seed):
        return {'rid': source.stem,
                'sample_seeds': [seed * 1000 + i for i in range(self.settings.samples)]}

    def load(self):
        self.loaded = True

    def snapshot(self):
        return {'loaded': True, 'device': 'cpu', 'engine': 'chai_lab',
                'checkpoint_load_count': 1}

    def predict(self, prepared, attempt, seed):
        tokens = ([{'index': i, 'kind': 'protein', 'chain': 'A', 'residue_index': i}
                   for i in range(3)]
                  + [{'index': 3, 'kind': 'ligand', 'chain': 'B', 'atom_name': 'C1',
                      'atom_index': 0},
                     {'index': 4, 'kind': 'ligand', 'chain': 'B', 'atom_name': 'O1',
                      'atom_index': 1}])
        pae = np.full((5, 5), 6.0)
        np.fill_diagonal(pae, 0.0)
        for index in range(self.settings.samples):
            np.savez_compressed(attempt / ('pae_%d.npz' % index), pae=pae)
            if self.token_policy == 'none':
                continue
            path = attempt / ('tokens_%d.json' % index)
            if self.token_policy == 'malformed':
                path.write_text('[not json')
            else:
                path.write_text(json.dumps({'schema': 1, 'n_tokens': len(tokens),
                                            'tokens': tokens}))

    def derive(self, prepared, attempt):
        type(self).derive_calls += 1
        predictions = []
        for index in range(self.settings.samples):
            pae_path = attempt / ('pae_%d.npz' % index)
            map_path = attempt / ('tokens_%d.json' % index)
            pae = None
            tokens = None
            reason = None
            try:
                with np.load(pae_path, allow_pickle=False) as arrays:
                    pae = np.asarray(arrays['pae'], dtype=float)
            except (OSError, ValueError, KeyError) as exc:
                reason = 'PAE array is unreadable: %s' % exc
            if not map_path.is_file():
                reason = reason or 'token map is unavailable'
            else:
                try:
                    tokens = derived.read_token_map(map_path)
                except derived.TokenMapError as exc:
                    reason = reason or str(exc)
            pairs = None
            if reason is None:
                pairs, reason = derived.score(pae, tokens)
            predictions.append({'sample_index': index,
                                'sample_seed': prepared['sample_seeds'][index],
                                'pae': pae_path.name if pae is not None else None,
                                'token_map': map_path.name if tokens is not None else None,
                                'pairs': pairs, 'derived_null_reason': reason})
        return derived.write_derived(attempt, predictions)

    def verify(self, prepared, attempt):
        assert (attempt / 'pae_0.npz').is_file()

    def healthy(self):
        return True


def write_engine_attempt(root):
    """An attempt directory with one sample's PAE and a Chai-1 token map."""
    attempt = root / 'attempt'
    native = attempt / 'native'
    native.mkdir(parents=True)
    prepared = {'rid': 'tiny', 'entities': [
        {'id': 'A', 'kind': 'protein', 'sequence': 'AGS'},
        {'id': 'L', 'kind': 'ligand', 'smiles': 'CO', 'heavy_atoms': 2}],
        'sample_seeds': [11000]}
    arrays = {'token_exists_mask': np.ones(5, dtype=bool),
              'token_asym_id': np.array([1, 1, 1, 2, 2]),
              'token_residue_index': np.array([0, 1, 2, 0, 0]),
              'token_entity_type': np.array([0, 0, 0, 3, 3]),
              'atom_token_index': np.array([0, 1, 2, 3, 4]),
              'atom_exists_mask': np.ones(5, dtype=bool),
              'atom_ref_name': ['CA', 'CA', 'CA', 'C1', 'O1'],
              'atom_ref_element': np.array([6, 6, 6, 6, 8])}
    pae = np.full((5, 5), 6.0)
    np.fill_diagonal(pae, 0.0)
    np.savez_compressed(native / 'pae_tiny_sample_1.npz', pae=pae.astype(np.float32))
    tokens = chai_engine.token_map(prepared, arrays, pae)
    (native / 'tokens_tiny_sample_1.json').write_text(json.dumps(tokens))
    return attempt, prepared


class ChaiEngineDeriveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.engine = chai_engine.Chai1Engine(self.root / 'assets', self.root / 'checkpoint',
                                             Settings())

    def test_engine_scores_written_pae_against_token_map(self):
        attempt, prepared = write_engine_attempt(self.root)
        self.engine.token_map_files = {0: 'tokens_tiny_sample_1.json'}
        self.engine.token_map_errors = {}
        info = self.engine.derive(prepared, attempt)
        self.assertEqual(info['path'], 'metrics/derived.json')
        self.assertIsNone(info['null_reason'])
        report = json.loads((attempt / 'metrics' / 'derived.json').read_text())
        self.assertEqual([f['id'] for f in report['formulas']],
                         ['ipsae2', 'iptm_d0chn', 'interface_pae_stats'])
        self.assertEqual(report['predictions'][0]['pairs'][0]['chains'], ['A', 'B'])

    def test_malformed_token_map_is_recorded_not_scored(self):
        attempt, prepared = write_engine_attempt(self.root)
        (attempt / 'native' / 'tokens_tiny_sample_1.json').write_text('[not json')
        self.engine.token_map_files = {0: 'tokens_tiny_sample_1.json'}
        self.engine.token_map_errors = {}
        info = self.engine.derive(prepared, attempt)
        self.assertIn('not valid JSON', info['null_reason'])
        report = json.loads((attempt / 'metrics' / 'derived.json').read_text())
        self.assertIsNone(report['predictions'][0]['pairs'])


class WorkerDerivedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        (self.inputs / 'system.yaml').write_text('version: 1\nsequences: []\n')
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        self.ckpt = self.root / 'checkpoint'
        self.ckpt.mkdir()
        (self.ckpt / 'config.json').write_text('{}')
        self.output = self.root / 'output'
        FakeEngine.token_policy = 'good'
        FakeEngine.derive_calls = 0

    def go(self, settings=None):
        return run(self.inputs, self.output, self.assets, self.ckpt, [11],
                   settings or Settings(samples=2), False, FakeEngine)

    def attempt(self):
        run_json = json.loads((self.output / 'run.json').read_text())
        return self.output / run_json['jobs'][0]['output']

    def test_derived_report_has_one_prediction_per_sample(self):
        self.assertEqual(self.go(), 0)
        report = json.loads((self.attempt() / 'metrics' / 'derived.json').read_text())
        self.assertEqual(report['schema'], 1)
        self.assertEqual(len(report['predictions']), 2)
        self.assertIsNone(report['derived_null_reason'])
        for index, prediction in enumerate(report['predictions']):
            self.assertEqual(prediction['sample_index'], index)
            self.assertEqual(prediction['pairs'][0]['chains'], ['A', 'B'])
            self.assertIsNotNone(prediction['pairs'][0]['by_cutoff'][0]['ipsae'])
        receipt = json.loads((self.attempt() / 'receipt.json').read_text())
        self.assertEqual(receipt['derived'], 'metrics/derived.json')

    def test_derived_none_writes_nothing_and_records_why(self):
        self.assertEqual(self.go(Settings(samples=2, derived='none')), 0)
        self.assertFalse((self.attempt() / 'metrics').exists())
        self.assertEqual(FakeEngine.derive_calls, 0)
        receipt = json.loads((self.attempt() / 'receipt.json').read_text())
        self.assertEqual(receipt['derived_null_reason'], 'disabled_by_--derived_none')

    def test_missing_token_map_is_a_recorded_skip(self):
        FakeEngine.token_policy = 'none'
        self.assertEqual(self.go(), 0)
        report = json.loads((self.attempt() / 'metrics' / 'derived.json').read_text())
        self.assertIn('token map', report['derived_null_reason'])
        self.assertTrue(all(p['pairs'] is None for p in report['predictions']))

    def test_malformed_token_map_is_a_recorded_skip(self):
        FakeEngine.token_policy = 'malformed'
        self.assertEqual(self.go(), 0)
        report = json.loads((self.attempt() / 'metrics' / 'derived.json').read_text())
        self.assertTrue(all(p['pairs'] is None for p in report['predictions']))
        self.assertIn('not valid JSON', report['derived_null_reason'])


if __name__ == '__main__':
    unittest.main()

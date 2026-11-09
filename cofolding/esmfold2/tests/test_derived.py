"""CPU-only tests for the derived interface metrics written by the ESMFold2 worker."""
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

from resident.worker import run, Settings  # noqa: E402
from cofolding.common import derived  # noqa: E402

TOKENS = (
    [{'index': i, 'kind': 'protein', 'chain': 'A', 'residue_index': i} for i in range(3)]
    + [{'index': 3, 'kind': 'ligand', 'chain': 'L', 'atom_name': 'C1', 'atom_index': 0},
       {'index': 4, 'kind': 'ligand', 'chain': 'L', 'atom_name': 'O1', 'atom_index': 1}]
)
PAE = np.full((5, 5), 6.0, dtype=float)
np.fill_diagonal(PAE, 0.0)


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
        return {'loaded': True, 'device': 'cpu', 'parameters': 1,
                'layout_sha256': 'x', 'checkpoint_load_count': 1}

    def predict(self, prepared, attempt, seed):
        for index in range(self.settings.samples):
            np.save(attempt / ('pae_%d.npy' % index), PAE)
        if self.token_policy == 'none':
            return
        for index in range(self.settings.samples):
            path = attempt / ('tokens_%d.json' % index)
            if self.token_policy == 'malformed':
                path.write_text('[not json')
            else:
                path.write_text(json.dumps({'schema': 1, 'n_tokens': len(TOKENS),
                                            'tokens': TOKENS}))

    def derive(self, prepared, attempt):
        type(self).derive_calls += 1
        predictions = []
        for index in range(self.settings.samples):
            pae_path = attempt / ('pae_%d.npy' % index)
            map_path = attempt / ('tokens_%d.json' % index)
            pae = np.load(pae_path) if pae_path.is_file() else None
            tokens = None
            reason = None
            if pae is None:
                reason = 'PAE array is unavailable'
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
            predictions.append({'sample_index': index, 'sample_seed': prepared['sample_seeds'][index],
                                'pae': pae_path.name if pae is not None else None,
                                'token_map': map_path.name if tokens is not None else None,
                                'pairs': pairs, 'derived_null_reason': reason})
        return derived.write_derived(attempt, predictions)

    def verify(self, prepared, attempt):
        assert (attempt / 'pae_0.npy').is_file()

    def healthy(self):
        return True


class DerivedTests(unittest.TestCase):
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
        self.assertEqual([f['id'] for f in report['formulas']],
                         ['ipsae2', 'iptm_d0chn', 'interface_pae_stats'])
        self.assertEqual(len(report['predictions']), 2)
        self.assertIsNone(report['derived_null_reason'])
        for index, prediction in enumerate(report['predictions']):
            self.assertEqual(prediction['sample_index'], index)
            self.assertEqual(prediction['pairs'][0]['chains'], ['A', 'L'])
            self.assertIsNotNone(prediction['pairs'][0]['by_cutoff'][0]['ipsae'])
        receipt = json.loads((self.attempt() / 'receipt.json').read_text())
        self.assertEqual(receipt['derived'], 'metrics/derived.json')

    def test_derived_none_writes_nothing_and_records_why(self):
        self.assertEqual(self.go(Settings(samples=2, derived='none')), 0)
        self.assertFalse((self.attempt() / 'metrics').exists())
        self.assertEqual(FakeEngine.derive_calls, 0)
        receipt = json.loads((self.attempt() / 'receipt.json').read_text())
        self.assertEqual(receipt['derived_null_reason'], 'disabled_by_--derived_none')
        rows = [json.loads(line) for line in
                (self.output / 'predictions.jsonl').read_text().splitlines()]
        self.assertTrue(rows)
        self.assertEqual(rows[0]['derived_null_reason'], 'disabled_by_--derived_none')

    def test_missing_token_map_is_a_recorded_skip(self):
        FakeEngine.token_policy = 'none'
        self.assertEqual(self.go(), 0)
        report = json.loads((self.attempt() / 'metrics' / 'derived.json').read_text())
        self.assertIn('token map', report['derived_null_reason'])
        self.assertEqual(len(report['predictions']), 2)
        self.assertTrue(all(p['pairs'] is None for p in report['predictions']))

    def test_malformed_token_map_is_a_recorded_skip(self):
        FakeEngine.token_policy = 'malformed'
        self.assertEqual(self.go(), 0)
        report = json.loads((self.attempt() / 'metrics' / 'derived.json').read_text())
        self.assertTrue(all(p['pairs'] is None for p in report['predictions']))
        self.assertIn('not valid JSON', report['derived_null_reason'])


if __name__ == '__main__':
    unittest.main()

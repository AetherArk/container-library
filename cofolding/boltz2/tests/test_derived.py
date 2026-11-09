"""CPU-only tests for the derived interface metrics written by the worker.

The fake engine stands in for Boltz2; the PAE and token map are synthetic, but
the score path is the real shared numpy module and the report is the real
``metrics/derived.json`` schema.
"""
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
    [{'index': i, 'kind': 'protein', 'chain': 'A', 'residue_index': i} for i in range(4)]
    + [{'index': 4, 'kind': 'ligand', 'chain': 'L', 'atom_name': 'C1', 'atom_index': 0},
       {'index': 5, 'kind': 'ligand', 'chain': 'L', 'atom_name': 'O1', 'atom_index': 1}]
)
PAE = np.full((6, 6), 4.0, dtype=float)
np.fill_diagonal(PAE, 0.0)
PAE[0, 4] = 9.0
PAE[1, 5] = 21.0


class FakeEngine:
    token_policy = 'good'
    derive_calls = 0

    def __init__(self, assets, checkpoint, settings):
        self.settings = settings
        self.loaded = False

    def prepare(self, source, attempt, seed):
        return source

    def load(self):
        self.loaded = True

    def snapshot(self):
        return {'model_id': 1, 'device': 'cpu', 'parameter_data_ptr': 2}

    def predict(self, prepared, attempt, seed):
        np.save(attempt / 'pae.npy', PAE)
        if self.token_policy == 'none':
            return
        path = attempt / 'tokens.json'
        if self.token_policy == 'malformed':
            path.write_text('{not json')
            return
        path.write_text(json.dumps({'schema': 1, 'n_tokens': len(TOKENS),
                                    'tokens': TOKENS}))

    def derive(self, prepared, attempt):
        type(self).derive_calls += 1
        pae_path = attempt / 'pae.npy'
        tokens = None
        reason = None
        if not pae_path.is_file():
            reason = 'PAE array is unavailable'
        if not (attempt / 'tokens.json').is_file():
            reason = reason or 'token map is unavailable'
        else:
            try:
                tokens = derived.read_token_map(attempt / 'tokens.json')
            except derived.TokenMapError as exc:
                reason = reason or str(exc)
        pairs = None
        if reason is None:
            pairs, reason = derived.score(np.load(pae_path), tokens)
        return derived.write_derived(attempt, [{
            'sample_index': 0, 'sample_seed': None,
            'pae': 'pae.npy' if pae_path.is_file() else None,
            'token_map': 'tokens.json' if tokens is not None else None,
            'pairs': pairs, 'derived_null_reason': reason}])

    def verify(self, prepared, attempt):
        assert (attempt / 'pae.npy').is_file()

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
        self.ckpt.write_bytes(b'checkpoint')
        self.output = self.root / 'output'
        FakeEngine.token_policy = 'good'
        FakeEngine.derive_calls = 0

    def go(self, settings=None):
        return run(self.inputs, self.output, self.assets, self.ckpt, [11],
                   settings or Settings(), False, FakeEngine)

    def attempt(self):
        run_json = json.loads((self.output / 'run.json').read_text())
        return self.output / run_json['jobs'][0]['output']

    def test_derived_report_has_pair_entries_and_is_referenced(self):
        self.assertEqual(self.go(), 0)
        attempt = self.attempt()
        report = json.loads((attempt / 'metrics' / 'derived.json').read_text())
        self.assertEqual(report['schema'], 1)
        self.assertEqual([f['id'] for f in report['formulas']],
                         ['ipsae2', 'iptm_d0chn', 'interface_pae_stats'])
        self.assertEqual(report['cutoffs'], [10.0, 15.0])
        self.assertIsNone(report['derived_null_reason'])
        self.assertEqual(len(report['predictions']), 1)
        pairs = report['predictions'][0]['pairs']
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['chains'], ['A', 'L'])
        self.assertEqual(pairs[0]['types'], ['protein', 'ligand'])
        self.assertEqual(len(pairs[0]['by_cutoff']), 2)
        first = pairs[0]['by_cutoff'][0]
        self.assertAlmostEqual(first['cutoff'], 10.0)
        self.assertIsNotNone(first['ipsae'])
        self.assertIsNotNone(first['iptm_d0chn'])
        self.assertGreater(first['n0chn'], 0)
        self.assertGreater(first['d0chn'], 0)
        self.assertIsNotNone(first['interface_pae']['mean'])
        receipt = json.loads((attempt / 'receipt.json').read_text())
        self.assertEqual(receipt['derived'], 'metrics/derived.json')
        rows = [json.loads(line) for line in
                (self.output / 'predictions.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['derived'], 'metrics/derived.json')
        self.assertIsNone(rows[0]['derived_null_reason'])

    def test_derived_none_writes_nothing_and_records_why(self):
        self.assertEqual(self.go(Settings(derived='none')), 0)
        attempt = self.attempt()
        self.assertFalse((attempt / 'metrics').exists())
        self.assertEqual(FakeEngine.derive_calls, 0)
        receipt = json.loads((attempt / 'receipt.json').read_text())
        self.assertIsNone(receipt['derived'])
        self.assertEqual(receipt['derived_null_reason'], 'disabled_by_--derived_none')
        rows = [json.loads(line) for line in
                (self.output / 'predictions.jsonl').read_text().splitlines()]
        self.assertIsNone(rows[0]['derived'])
        self.assertEqual(rows[0]['derived_null_reason'], 'disabled_by_--derived_none')

    def test_missing_token_map_is_a_recorded_skip(self):
        FakeEngine.token_policy = 'none'
        self.assertEqual(self.go(), 0)
        report = json.loads((self.attempt() / 'metrics' / 'derived.json').read_text())
        self.assertIn('token map', report['derived_null_reason'])
        self.assertIsNone(report['predictions'][0]['pairs'])
        self.assertIn('token map', report['predictions'][0]['derived_null_reason'])

    def test_malformed_token_map_is_a_recorded_skip(self):
        FakeEngine.token_policy = 'malformed'
        self.assertEqual(self.go(), 0)
        report = json.loads((self.attempt() / 'metrics' / 'derived.json').read_text())
        self.assertIsNone(report['predictions'][0]['pairs'])
        self.assertIn('not valid JSON', report['predictions'][0]['derived_null_reason'])


if __name__ == '__main__':
    unittest.main()

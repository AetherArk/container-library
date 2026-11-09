"""CPU-only tests for cofolding.common.derived (numpy only, no torch/GPU)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from cofolding.common import derived  # noqa: E402

TOKENS = [
    {'index': 0, 'kind': 'protein', 'chain': 'A', 'residue_index': 0},
    {'index': 1, 'kind': 'protein', 'chain': 'A', 'residue_index': 1},
    {'index': 2, 'kind': 'ligand', 'chain': 'L', 'atom_name': 'C1', 'atom_index': 0},
]
PAE = np.array([[0.0, 2.0, 5.0],
                [3.0, 0.0, 7.0],
                [4.0, 6.0, 0.0]], dtype=float)


class DerivedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.attempt = Path(self.tmp.name)

    def test_score_reports_pair_values_and_reasons(self):
        pairs, reason = derived.score(PAE, TOKENS)
        self.assertIsNone(reason)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['chains'], ['A', 'L'])
        first = pairs[0]['by_cutoff'][0]
        for key in ('cutoff', 'ipsae', 'ipsae_source', 'iptm_d0chn', 'n0chn', 'd0chn',
                    'n0dom', 'd0dom', 'interface_pae'):
            self.assertIn(key, first)
        self.assertGreater(first['d0chn'], 0)

    def test_short_token_map_is_refused_not_scored(self):
        pairs, reason = derived.score(PAE, TOKENS[:2])
        self.assertIsNone(pairs)
        self.assertIn('tokens were given', reason)

    def test_unknown_kind_is_refused(self):
        bad = [dict(token) for token in TOKENS]
        bad[0]['kind'] = 'water'
        pairs, reason = derived.score(PAE, bad)
        self.assertIsNone(pairs)
        self.assertIn("kind", reason)

    def test_missing_residue_index_is_refused(self):
        bad = [dict(token) for token in TOKENS]
        del bad[1]['residue_index']
        pairs, reason = derived.score(PAE, bad)
        self.assertIsNone(pairs)
        self.assertIn('residue_index', reason)

    def test_write_derived_records_skip_reason(self):
        info = derived.write_derived(self.attempt, [{
            'sample_index': 0, 'sample_seed': None,
            'pae': 'pae.npz', 'token_map': None, 'pairs': None,
            'derived_null_reason': 'token map is unavailable'}])
        report = json.loads((self.attempt / 'metrics' / 'derived.json').read_text())
        self.assertEqual(info['path'], 'metrics/derived.json')
        self.assertEqual(report['schema'], 1)
        self.assertEqual(report['derived_null_reason'], 'token map is unavailable')
        self.assertEqual(report['predictions'][0]['pairs'], None)

    def test_write_derived_with_no_predictions_says_so(self):
        info = derived.write_derived(self.attempt, [])
        report = json.loads((self.attempt / 'metrics' / 'derived.json').read_text())
        self.assertIn('no prediction', info['null_reason'])
        self.assertIn('no prediction', report['derived_null_reason'])

    def test_read_token_map_rejects_malformed_and_out_of_order(self):
        path = self.attempt / 'tokens.json'
        path.write_text('{oops')
        with self.assertRaises(derived.TokenMapError):
            derived.read_token_map(path)
        path.write_text(json.dumps({'n_tokens': 2, 'tokens': [
            {'index': 1, 'kind': 'protein', 'chain': 'A', 'residue_index': 0},
            {'index': 0, 'kind': 'protein', 'chain': 'A', 'residue_index': 1}]}))
        with self.assertRaises(derived.TokenMapError):
            derived.read_token_map(path)
        path.write_text(json.dumps({'n_tokens': 1, 'tokens': [
            {'index': 0, 'kind': 'ligand', 'chain': 'L', 'atom_name': 'C1'}]}))
        with self.assertRaises(derived.TokenMapError):
            derived.read_token_map(path)


if __name__ == '__main__':
    unittest.main()

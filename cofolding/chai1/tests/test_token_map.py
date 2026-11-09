"""CPU-only tests for the Chai-1 PAE token map (synthetic collated-feature arrays)."""
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

from cofolding.common import derived  # noqa: E402
from resident import engine  # noqa: E402


def prepared(entities=None):
    return {'rid': 'tiny', 'entities': entities or [
        {'id': 'A', 'kind': 'protein', 'sequence': 'AGS'},
        {'id': 'L', 'kind': 'ligand', 'smiles': 'CO', 'heavy_atoms': 2}]}


def arrays(exists=(True, True, True, True, True), asym=(1, 1, 1, 2, 2),
           residue=(0, 1, 2, 0, 0), etype=(0, 0, 0, 3, 3),
           atom_token=(0, 1, 2, 3, 4), names=('CA', 'CA', 'CA', 'C1', 'O1'),
           elements=(6, 6, 6, 6, 8)):
    return {'token_exists_mask': np.asarray(exists),
            'token_asym_id': np.asarray(asym),
            'token_residue_index': np.asarray(residue),
            'token_entity_type': np.asarray(etype),
            'atom_token_index': np.asarray(atom_token),
            'atom_exists_mask': np.ones(len(atom_token), dtype=bool),
            'atom_ref_name': list(names),
            'atom_ref_element': np.asarray(elements)}


PAE = np.zeros((5, 5))


class TokenMapTests(unittest.TestCase):
    def test_one_token_per_residue_and_ligand_atom(self):
        mapping = engine.token_map(prepared(), arrays(), PAE)
        self.assertEqual(mapping['n_tokens'], 5)
        self.assertEqual([t['kind'] for t in mapping['tokens']],
                         ['protein', 'protein', 'protein', 'ligand', 'ligand'])
        # Chai-1 writes chains A, B, ... by asym order, so the ligand is chain B.
        self.assertEqual([t['chain'] for t in mapping['tokens']], ['A', 'A', 'A', 'B', 'B'])
        self.assertEqual([t['residue_index'] for t in mapping['tokens'][:3]], [0, 1, 2])
        self.assertEqual([t['atom_name'] for t in mapping['tokens'][3:]], ['C1', 'O1'])
        self.assertEqual([t['atom_index'] for t in mapping['tokens'][3:]], [0, 1])
        self.assertEqual([t['element'] for t in mapping['tokens'][3:]], ['C', 'O'])
        self.assertEqual([c['input_id'] for c in mapping['chains']], ['A', 'L'])

    def test_map_is_accepted_by_the_shared_reader(self):
        mapping = engine.token_map(prepared(), arrays(), PAE)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'tokens.json'
            path.write_text(json.dumps(mapping))
            tokens = derived.read_token_map(path)
        self.assertEqual(len(tokens), 5)
        pairs, reason = derived.score(PAE, tokens)
        self.assertIsNone(reason)
        self.assertEqual(pairs[0]['chains'], ['A', 'B'])

    def test_pae_size_mismatch_is_refused(self):
        with self.assertRaisesRegex(derived.TokenMapError, 'PAE matrix has 6 rows'):
            engine.token_map(prepared(), arrays(), np.zeros((6, 6)))

    def test_missing_entity_is_refused(self):
        with self.assertRaisesRegex(derived.TokenMapError, 'chains but'):
            engine.token_map(prepared(entities=[{'id': 'A', 'kind': 'protein',
                                                 'sequence': 'AGS'}]), arrays(), PAE)

    def test_protein_residue_count_mismatch_is_refused(self):
        bad = prepared(entities=[{'id': 'A', 'kind': 'protein', 'sequence': 'AG'},
                                 {'id': 'L', 'kind': 'ligand', 'heavy_atoms': 2}])
        with self.assertRaisesRegex(derived.TokenMapError, '3 tokens for a 2-residue'):
            engine.token_map(bad, arrays(), PAE)

    def test_ligand_heavy_atom_mismatch_is_refused(self):
        bad = prepared(entities=[{'id': 'A', 'kind': 'protein', 'sequence': 'AGS'},
                                 {'id': 'L', 'kind': 'ligand', 'heavy_atoms': 3}])
        with self.assertRaisesRegex(derived.TokenMapError, '2 tokens but the input SMILES has 3'):
            engine.token_map(bad, arrays(), PAE)

    def test_missing_token_array_is_refused(self):
        broken = arrays()
        del broken['token_asym_id']
        with self.assertRaisesRegex(derived.TokenMapError, 'token_asym_id'):
            engine.token_map(prepared(), broken, PAE)

    def test_non_contiguous_chain_is_refused(self):
        with self.assertRaisesRegex(derived.TokenMapError, 'not contiguous'):
            engine.token_map(prepared(), arrays(asym=(1, 1, 2, 1, 2)), PAE)

    def test_existing_token_mask_must_cover_the_pae(self):
        with self.assertRaisesRegex(derived.TokenMapError, '4 tokens'):
            engine.token_map(prepared(), arrays(exists=(True, True, True, True, False)), PAE)


if __name__ == '__main__':
    unittest.main()

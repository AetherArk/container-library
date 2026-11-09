"""CPU-only tests for the Protenix PAE token map (synthetic tokenized-feature arrays)."""
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

# Protenix encodes elements as a 128-way one-hot over get_all_elems(), i.e. atomic number - 1.
ELEMENTS = ['X'] * 128
ELEMENTS[0] = 'H'
ELEMENTS[5] = 'C'
ELEMENTS[6] = 'N'
ELEMENTS[7] = 'O'


def prepared(entities=None):
    return {'rid': 'tiny', 'entities': entities or [
        {'id': 'A', 'kind': 'protein', 'sequence': 'AGS', 'cif_chain': 'A'},
        {'id': 'L', 'kind': 'ligand', 'smiles': 'CO', 'heavy_atoms': 2, 'cif_chain': 'B'}]}


def element_row(symbol):
    row = np.zeros(128, dtype=np.int64)
    row[ELEMENTS.index(symbol)] = 1
    return row


def name_rows(names):
    rows = np.zeros((len(names), 4, 64), dtype=np.int64)
    for index, name in enumerate(names):
        for slot, character in enumerate(name.ljust(4)[:4]):
            rows[index, slot, max(0, min(63, ord(character) - 32))] = 1
    return rows


def arrays(index=(0, 1, 2, 3, 4), asym=(0, 0, 0, 1, 1), residue=(1, 2, 3, 1, 1),
           atom_token=(0, 1, 2, 3, 4), ligand=(0, 0, 0, 1, 1),
           names=('CA', 'CA', 'CA', 'C1', 'O1'), elements=('C', 'C', 'C', 'C', 'O')):
    return {'token_index': np.asarray(index),
            'residue_index': np.asarray(residue),
            'asym_id': np.asarray(asym),
            'entity_id': np.asarray(asym),
            'atom_to_token_idx': np.asarray(atom_token),
            'is_ligand': np.asarray(ligand),
            'ref_element': np.stack([element_row(symbol) for symbol in elements]),
            'ref_atom_name_chars': name_rows(list(names))}


PAE = np.zeros((5, 5))


class TokenMapTests(unittest.TestCase):
    def test_one_token_per_residue_and_ligand_atom(self):
        mapping = engine.token_map(prepared(), arrays(), PAE, ELEMENTS)
        self.assertEqual(mapping['n_tokens'], 5)
        self.assertEqual([t['kind'] for t in mapping['tokens']],
                         ['protein', 'protein', 'protein', 'ligand', 'ligand'])
        # The container writes its own chain letters, so the ligand is chain B.
        self.assertEqual([t['chain'] for t in mapping['tokens']], ['A', 'A', 'A', 'B', 'B'])
        self.assertEqual([t['residue_index'] for t in mapping['tokens'][:3]], [1, 2, 3])
        self.assertEqual([t['atom_name'] for t in mapping['tokens'][3:]], ['C1', 'O1'])
        self.assertEqual([t['atom_index'] for t in mapping['tokens'][3:]], [0, 1])
        self.assertEqual([t['element'] for t in mapping['tokens'][3:]], ['C', 'O'])
        self.assertEqual([c['input_id'] for c in mapping['chains']], ['A', 'L'])

    def test_map_is_accepted_by_the_shared_reader(self):
        mapping = engine.token_map(prepared(), arrays(), PAE, ELEMENTS)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'tokens.json'
            path.write_text(json.dumps(mapping))
            tokens = derived.read_token_map(path)
        self.assertEqual(len(tokens), 5)
        pairs, reason = derived.score(PAE, tokens)
        self.assertIsNone(reason)
        self.assertEqual(pairs[0]['chains'], ['A', 'B'])

    def test_single_atom_ligand_is_scored_as_an_ion(self):
        document = prepared(entities=[
            {'id': 'A', 'kind': 'protein', 'sequence': 'AGS', 'cif_chain': 'A'},
            {'id': 'Z', 'kind': 'ligand', 'smiles': '[Zn+2]', 'heavy_atoms': 1, 'cif_chain': 'B'}])
        mapping = engine.token_map(document, arrays(index=(0, 1, 2, 3), asym=(0, 0, 0, 1),
                                                    residue=(1, 2, 3, 1), atom_token=(0, 1, 2, 3),
                                                    ligand=(0, 0, 0, 1), names=('CA', 'CA', 'CA', 'ZN'),
                                                    elements=('C', 'C', 'C', 'N')),
                                   np.zeros((4, 4)), ELEMENTS)
        self.assertEqual(mapping['tokens'][3]['kind'], 'ion')
        self.assertEqual(mapping['tokens'][3]['atom_name'], 'ZN')

    def test_pae_size_mismatch_is_refused(self):
        with self.assertRaisesRegex(derived.TokenMapError, 'PAE matrix has 6 rows'):
            engine.token_map(prepared(), arrays(), np.zeros((6, 6)), ELEMENTS)

    def test_missing_entity_is_refused(self):
        with self.assertRaisesRegex(derived.TokenMapError, 'chains but'):
            engine.token_map(prepared(entities=[{'id': 'A', 'kind': 'protein', 'sequence': 'AGS',
                                                 'cif_chain': 'A'}]), arrays(), PAE, ELEMENTS)

    def test_protein_residue_count_mismatch_is_refused(self):
        bad = prepared(entities=[{'id': 'A', 'kind': 'protein', 'sequence': 'AG', 'cif_chain': 'A'},
                                 {'id': 'L', 'kind': 'ligand', 'heavy_atoms': 2, 'cif_chain': 'B'}])
        with self.assertRaisesRegex(derived.TokenMapError, '3 tokens for a 2-residue'):
            engine.token_map(bad, arrays(), PAE, ELEMENTS)

    def test_residue_indices_must_be_one_based_and_complete(self):
        with self.assertRaisesRegex(derived.TokenMapError, 'residue input'):
            engine.token_map(prepared(), arrays(residue=(0, 1, 2, 1, 1)), PAE, ELEMENTS)

    def test_ligand_heavy_atom_mismatch_is_refused(self):
        bad = prepared(entities=[{'id': 'A', 'kind': 'protein', 'sequence': 'AGS', 'cif_chain': 'A'},
                                 {'id': 'L', 'kind': 'ligand', 'heavy_atoms': 3, 'cif_chain': 'B'}])
        with self.assertRaisesRegex(derived.TokenMapError, '2 tokens but the input SMILES has 3'):
            engine.token_map(bad, arrays(), PAE, ELEMENTS)

    def test_missing_token_array_is_refused(self):
        broken = arrays()
        del broken['asym_id']
        with self.assertRaisesRegex(derived.TokenMapError, 'asym_id'):
            engine.token_map(prepared(), broken, PAE, ELEMENTS)

    def test_non_contiguous_chain_is_refused(self):
        with self.assertRaisesRegex(derived.TokenMapError, 'not contiguous'):
            engine.token_map(prepared(), arrays(asym=(0, 0, 1, 0, 1)), PAE, ELEMENTS)

    def test_token_order_must_be_the_feature_order(self):
        with self.assertRaisesRegex(derived.TokenMapError, 'token_index'):
            engine.token_map(prepared(), arrays(index=(0, 1, 3, 2, 4)), PAE, ELEMENTS)

    def test_atom_pointing_outside_the_tokens_is_refused(self):
        with self.assertRaisesRegex(derived.TokenMapError, 'outside the token range'):
            engine.token_map(prepared(), arrays(atom_token=(0, 1, 2, 3, 9)), PAE, ELEMENTS)

    def test_entity_kind_mismatch_is_refused(self):
        bad = prepared(entities=[{'id': 'A', 'kind': 'ligand', 'smiles': 'CO', 'heavy_atoms': 2,
                                  'cif_chain': 'A'},
                                 {'id': 'L', 'kind': 'ligand', 'heavy_atoms': 2, 'cif_chain': 'B'}])
        with self.assertRaisesRegex(derived.TokenMapError, 'polymer in Protenix'):
            engine.token_map(bad, arrays(), PAE, ELEMENTS)


if __name__ == '__main__':
    unittest.main()

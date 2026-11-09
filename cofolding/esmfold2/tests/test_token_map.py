"""CPU-only tests for the ESMFold2 PAE token map (synthetic engine objects)."""
from pathlib import Path
import sys
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

from cofolding.common import derived  # noqa: E402
from resident import engine  # noqa: E402


class ProteinInput:
    def __init__(self, id, sequence):
        self.id, self.sequence = id, sequence


class LigandInput:
    def __init__(self, id, smiles):
        self.id, self.smiles = id, smiles


class Complex:
    def __init__(self, sequence, chain_id, token_to_atoms, atom_names, elements, lookup):
        self.sequence = sequence
        self.chain_id = np.array(chain_id, dtype=np.int64)
        self.token_to_atoms = np.array(token_to_atoms, dtype=np.int32)
        self.atom_names = np.array(atom_names, dtype=object)
        self.atom_elements = np.array(elements, dtype=object)
        self.metadata = types.SimpleNamespace(chain_lookup=lookup)


def prepared():
    return {'input': types.SimpleNamespace(sequences=[ProteinInput('A', 'AGS'),
                                                      LigandInput('L', 'CO')])}


def result(pae_tokens=5, chain_id=(0, 0, 0, 1), token_to_atoms=((0, 2), (2, 4), (4, 5), (5, 7))):
    complex_obj = Complex(['ALA', 'GLY', 'SER', 'LIG'], chain_id, token_to_atoms,
                          ['N', 'CA', 'C', 'O', 'CB', 'C1', 'O1'],
                          ['N', 'C', 'C', 'O', 'C', 'C', 'O'], {0: 'A', 1: 'L'})
    return types.SimpleNamespace(pae=np.zeros((pae_tokens, pae_tokens)),
                                 entity_id=np.array([0, 0, 0, 1, 1]),
                                 complex=complex_obj)


class TokenMapTests(unittest.TestCase):
    def test_one_token_per_residue_and_ligand_atom(self):
        mapping = engine.token_map(prepared(), result())
        self.assertEqual(mapping['n_tokens'], 5)
        self.assertEqual([t['kind'] for t in mapping['tokens']],
                         ['protein', 'protein', 'protein', 'ligand', 'ligand'])
        self.assertEqual([t['chain'] for t in mapping['tokens']], ['A', 'A', 'A', 'L', 'L'])
        self.assertEqual([t['residue_index'] for t in mapping['tokens'][:3]], [0, 1, 2])
        self.assertEqual([t['atom_name'] for t in mapping['tokens'][3:]], ['C1', 'O1'])
        self.assertEqual([t['atom_index'] for t in mapping['tokens'][3:]], [0, 1])
        # The map is accepted by the shared reader and scores without a skip.
        tokens = derived.read_token_map(  # round-trip through JSON is covered elsewhere
            self.write(mapping))
        self.assertEqual(len(tokens), 5)

    def write(self, mapping):
        import json
        import tempfile
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'tokens.json'
        path.write_text(json.dumps(mapping))
        return path

    def test_pae_size_mismatch_is_refused(self):
        with self.assertRaisesRegex(derived.TokenMapError, '6 PAE tokens'):
            engine.token_map(prepared(), result(pae_tokens=6))

    def test_missing_input_chain_is_refused(self):
        bad = {'input': types.SimpleNamespace(sequences=[ProteinInput('A', 'AGS')])}
        with self.assertRaisesRegex(derived.TokenMapError, 'chains but'):
            engine.token_map(bad, result())

    def test_ligand_residue_count_mismatch_is_refused(self):
        with self.assertRaisesRegex(derived.TokenMapError, 'residues for a 3-residue'):
            engine.token_map(prepared(), result(chain_id=(0, 0, 1), token_to_atoms=((0, 2), (2, 4), (4, 5), (5, 7))))


if __name__ == '__main__':
    unittest.main()

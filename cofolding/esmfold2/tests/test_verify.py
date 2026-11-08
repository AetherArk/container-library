"""Independent verifier regressions on synthetic artifacts (no model, no GPU)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resident import verify as verifier  # noqa: E402

RESIDUES = 5
LIGAND_SMILES = 'CCO'
LIGAND_ELEMENTS = ['C', 'C', 'O']
TOKENS = RESIDUES + len(LIGAND_ELEMENTS)
PROTEIN_ATOMS = ['N', 'CA', 'C', 'O']


def write_cif(path, coordinates, chains):
    """Write an mmCIF from explicit chain -> [(residue_name, [elements])] and coordinate rows."""
    import gemmi
    structure = gemmi.Structure()
    model = gemmi.Model('1')
    index = 0
    for chain_name, residues in chains.items():
        chain = gemmi.Chain(chain_name)
        for residue_number, (residue_name, elements) in enumerate(residues, start=1):
            residue = gemmi.Residue()
            residue.name = residue_name
            residue.seqid = gemmi.SeqId(residue_number, ' ')
            for atom_index, element in enumerate(elements):
                atom = gemmi.Atom()
                atom.name = element + str(atom_index)
                atom.element = gemmi.Element(element)
                x, y, z = coordinates[index]
                index += 1
                atom.pos = gemmi.Position(float(x), float(y), float(z))
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    path.write_text(structure.make_mmcif_document().as_string())
    assert index == len(coordinates), (index, len(coordinates))


class Complex:
    def __init__(self, positions):
        self.atom_positions = positions


class Result:
    def __init__(self, positions):
        self.complex = Complex(positions)


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.attempt = Path(self.tmp.name) / 'attempt'
        (self.attempt / 'native').mkdir(parents=True)
        (self.attempt / 'capture').mkdir()
        self.settings = type('S', (), {'samples': 1})()
        self.prepared = {'rid': 'tiny', 'entities': ['A', 'L']}
        (self.attempt / 'chemistry.json').write_text(json.dumps({
            'protein_id': 'A', 'residues': RESIDUES,
            'ligand': {'id': 'L', 'consumed_smiles': LIGAND_SMILES}, 'heme': None}))
        rows = [[float(index), 0.5, 1.25] for index in range(RESIDUES * len(PROTEIN_ATOMS))]
        rows += [[10.0 + index, 2.0, 3.0] for index in range(len(LIGAND_ELEMENTS))]
        self.positions = np.asarray(rows, dtype=np.float32)
        self.chains = {'A': [('ALA', PROTEIN_ATOMS) for _ in range(RESIDUES)],
                       'L': [('LIG', LIGAND_ELEMENTS)]}
        self.write_native()
        self.write_confidence()

    def write_native(self, positions=None, chains=None):
        write_cif(self.attempt / 'native' / 'tiny_sample_1.cif',
                  self.positions if positions is None else positions,
                  self.chains if chains is None else chains)

    def write_confidence(self):
        label = 'tiny_sample_1'
        (self.attempt / 'native' / f'confidence_{label}.json').write_text(json.dumps({
            'sample_index': 1, 'sample_seed': 11000, 'plddt_mean': 70.0, 'ptm': 0.5, 'iptm': 0.4,
            'residues': RESIDUES, 'tokens': TOKENS, 'chains': ['A', 'L'],
            'pae_shape': [TOKENS, TOKENS]}))
        np.savez_compressed(self.attempt / 'native' / f'pae_{label}.npz',
                            pae=np.zeros((TOKENS, TOKENS), dtype=np.float32))
        np.savez_compressed(self.attempt / 'native' / f'plddt_{label}.npz',
                            plddt=np.full(TOKENS, 70.0, dtype=np.float32))
        import torch
        torch.save(Result(self.positions), self.attempt / 'capture' / 'full_result_sample_1.pt')
        (self.attempt / 'capture' / 'full_result_sample_1_inventory.json').write_text(json.dumps({
            '/complex/atom_positions': {'shape': list(self.positions.shape),
                                        'dtype': 'torch.float32', 'finite': True}}))

    def verify(self):
        verifier.verify_job(self.prepared, self.attempt, self.settings)

    def test_valid_artifacts_pass(self):
        self.verify()

    def test_missing_ligand_chain_fails(self):
        self.write_native(chains={'A': self.chains['A']},
                          positions=self.positions[:RESIDUES * len(PROTEIN_ATOMS)])
        with self.assertRaisesRegex(ValueError, 'chain set'):
            self.verify()

    def test_ligand_heavy_atom_loss_fails(self):
        chains = {'A': self.chains['A'], 'L': [('LIG', ['C', 'C'])]}
        self.write_native(chains=chains, positions=self.positions[:RESIDUES * 4 + 2])
        with self.assertRaisesRegex(ValueError, 'ligand heavy-atom count'):
            self.verify()

    def test_truncated_protein_fails(self):
        data = json.loads((self.attempt / 'chemistry.json').read_text())
        data['residues'] = RESIDUES + 3
        (self.attempt / 'chemistry.json').write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'protein residue count'):
            self.verify()

    def test_nonfinite_coordinate_fails(self):
        positions = self.positions.copy()
        positions[0, 0] = np.nan
        self.write_native(positions=positions)
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            self.verify()

    def test_coordinate_mismatch_fails(self):
        import torch
        torch.save(Result(self.positions + 1.0),
                   self.attempt / 'capture' / 'full_result_sample_1.pt')
        with self.assertRaisesRegex(ValueError, 'differ from the written CIF'):
            self.verify()

    def test_missing_confidence_fails(self):
        (self.attempt / 'native' / 'confidence_tiny_sample_1.json').unlink()
        with self.assertRaisesRegex(ValueError, 'confidence file count'):
            self.verify()

    def test_wrong_pae_shape_fails(self):
        label = 'tiny_sample_1'
        np.savez_compressed(self.attempt / 'native' / f'pae_{label}.npz',
                            pae=np.zeros((2, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'PAE must be square'):
            self.verify()

    def test_square_but_truncated_pae_fails(self):
        np.savez_compressed(self.attempt / 'native' / 'pae_tiny_sample_1.npz',
                            pae=np.zeros((RESIDUES, RESIDUES), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'PAE token coverage'):
            self.verify()

    def test_protein_only_plddt_cannot_cover_ligand_tokens(self):
        np.savez_compressed(self.attempt / 'native' / 'plddt_tiny_sample_1.npz',
                            plddt=np.full(RESIDUES, 70.0, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'token pLDDT length'):
            self.verify()

    def test_chain_set_must_match_request(self):
        self.prepared['entities'] = ['A', 'L', 'H']
        with self.assertRaisesRegex(ValueError, 'chain set'):
            self.verify()


if __name__ == '__main__':
    unittest.main()

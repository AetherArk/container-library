"""Input capability mapping: shared YAML entities -> ESMFold2's protein-chain contract."""
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resident.engine import capability, canonical_smiles  # noqa: E402

HEME = ('Cc1c2[n-]3c(c1CCC(=O)O)C=C4C(=C(C5=[N]4[Fe+2]36[N]7=C(C=C8[N-]6C(=C5)C(=C8C)C=C)'
        'C(=C(C7=C2)C)C=C)C)CCC(=O)O')
TRPCAGE = 'NLYIQWLKDGGPSSGRPPPS'


def system(*entities):
    return {'version': 1, 'sequences': list(entities)}


def protein(sequence=TRPCAGE, id='A', **extra):
    return {'protein': dict(id=id, sequence=sequence, **extra)}


def ligand(smiles='CCO', id='L'):
    return {'ligand': dict(id=id, smiles=smiles)}


class CapabilityTests(unittest.TestCase):
    def test_protein_with_one_ligand(self):
        resolved = capability(system(protein(), ligand()), 'x.yaml')
        self.assertEqual([c['id'] for c in resolved['proteins']], ['A'])
        self.assertEqual(resolved['proteins'][0]['sequence'], TRPCAGE)
        self.assertEqual(resolved['ligand']['smiles'], 'CCO')
        self.assertIsNone(resolved['heme'])

    def test_heme_is_recognised_by_iron_and_separated(self):
        resolved = capability(system(protein(), ligand(HEME, 'B'), ligand('CCN', 'L')), 'cyp.yaml')
        self.assertEqual(resolved['heme']['id'], 'B')
        self.assertEqual(resolved['ligand']['id'], 'L')

    def test_heme_alone_is_allowed(self):
        resolved = capability(system(protein(), ligand(HEME, 'B')), 'heme-only.yaml')
        self.assertIsNone(resolved['ligand'])
        self.assertEqual(resolved['heme']['id'], 'B')

    def test_two_protein_chains_accepted(self):
        resolved = capability(system(protein(), protein(id='B'), ligand()), 'zika.yaml')
        self.assertEqual([c['id'] for c in resolved['proteins']], ['A', 'B'])

    def test_duplicate_protein_chain_id_rejected(self):
        with self.assertRaisesRegex(ValueError, 'chain ids must be unique'):
            capability(system(protein(), protein(id='A'), ligand()), 'duplicate.yaml')

    def test_chain_id_colliding_with_ligand_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unique across proteins and ligands'):
            capability(system(protein(), ligand(id='A')), 'collision.yaml')


    def test_second_ligand_rejected(self):
        with self.assertRaisesRegex(ValueError, 'one ligand slot'):
            capability(system(protein(), ligand('CCO', 'L'), ligand('CCN', 'M')), 'two-ligands.yaml')

    def test_two_hemes_rejected(self):
        with self.assertRaisesRegex(ValueError, 'one ligand slot'):
            capability(system(protein(), ligand(HEME, 'B'), ligand(HEME, 'C')), 'two-hemes.yaml')

    def test_apo_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least one ligand'):
            capability(system(protein()), 'apo.yaml')

    def test_msa_file_rejected_but_empty_allowed(self):
        capability(system(protein(msa='empty'), ligand()), 'ok.yaml')
        with self.assertRaisesRegex(ValueError, 'without MSA'):
            capability(system(protein(msa='../msa.a3m'), ligand()), 'msa.yaml')

    def test_affinity_rejected(self):
        document = system(protein(), ligand())
        document['properties'] = [{'affinity': {'binder': 'L'}}]
        with self.assertRaisesRegex(ValueError, 'affinity'):
            capability(document, 'affinity.yaml')

    def test_templates_and_constraints_rejected(self):
        for field in ('templates', 'constraints'):
            document = system(protein(), ligand())
            document[field] = [{'x': 1}]
            with self.assertRaisesRegex(ValueError, field):
                capability(document, field + '.yaml')

    def test_noncanonical_sequence_rejected(self):
        with self.assertRaisesRegex(ValueError, 'canonical uppercase'):
            capability(system(protein('NLYIQX'), ligand()), 'bad.yaml')

    def test_seeds_inside_yaml_rejected(self):
        document = system(protein(), ligand())
        document['seed'] = 3
        with self.assertRaisesRegex(ValueError, 'external --seeds'):
            capability(document, 'seed.yaml')

    def test_modifications_and_ccd_rejected(self):
        with self.assertRaisesRegex(ValueError, 'modified or cyclic'):
            capability(system(protein(modifications=[{'x': 1}]), ligand()), 'mod.yaml')
        with self.assertRaisesRegex(ValueError, 'CCD ligand references'):
            capability(system(protein(), {'ligand': {'id': 'L', 'ccd': 'HEM'}}), 'ccd.yaml')

    def test_chemistry_record_preserves_consumed_state(self):
        molecule, chemistry = canonical_smiles('OCC')
        self.assertEqual(chemistry['consumed_smiles'], 'OCC')
        self.assertEqual(chemistry['canonical_smiles'], 'CCO')
        self.assertEqual(chemistry['heavy_atoms'], 3)
        self.assertEqual(chemistry['iron_atoms'], 0)
        self.assertIn('no neutralization', chemistry['state_policy'])

    def test_unparseable_smiles_rejected(self):
        with self.assertRaisesRegex(ValueError, 'does not parse'):
            capability(system(protein(), ligand('not-a-smiles')), 'bad-ligand.yaml')

    def test_packaged_examples_match_engine_capability(self):
        import yaml
        examples = ROOT / 'examples'
        supported, rejected = [], {}
        for path in sorted(examples.glob('*.yaml')):
            document = yaml.safe_load(path.read_text())
            try:
                resolved = capability(document, path.name)
                supported.append((path.stem, resolved))
            except ValueError as error:
                rejected[path.stem] = str(error)
        names = [name for name, _ in supported]
        self.assertEqual(sorted(names), ['cyp', 'trpcage-ethanol', 'two-chain'], rejected)
        cyp = dict(supported)['cyp']
        self.assertIsNotNone(cyp['heme'])
        self.assertEqual(cyp['heme']['id'], 'B')
        self.assertEqual(cyp['ligand']['id'], 'L')
        self.assertIsNone(dict(supported)['trpcage-ethanol']['heme'])

    def test_shared_test_cases_reject_multi_chain_systems(self):
        """The shared YAML panels are the same inputs the Boltz image uses; Zika must be refused."""
        import yaml
        shared = ROOT.parent / 'test-cases'
        panels = sorted(shared.glob('*/input.yaml')) + sorted(shared.glob('throughput-panel/*.yaml'))
        for path in panels:
            document = yaml.safe_load(path.read_text())
            try:
                capability(document, path.name)
            except ValueError as error:
                self.assertIn('exactly one protein', str(error), path)


if __name__ == '__main__':
    unittest.main()

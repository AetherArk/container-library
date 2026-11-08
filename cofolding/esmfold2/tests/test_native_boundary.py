"""CPU serialization checks; fixtures are not model predictions."""
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resident.engine import cpu, save_packet


@dataclass
class FixtureComplex:
    atom_positions: np.ndarray


@dataclass
class FixtureResult:
    complex: FixtureComplex
    plddt: torch.Tensor


class NativePacketTests(unittest.TestCase):
    def test_dataclass_result_type_and_inventory_survive_roundtrip(self):
        result = FixtureResult(FixtureComplex(np.zeros((2, 3))), torch.ones(2, requires_grad=True))
        saved = cpu(result)
        self.assertIsInstance(saved, FixtureResult)
        self.assertIsInstance(saved.complex, FixtureComplex)
        self.assertFalse(saved.plddt.requires_grad)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_packet(root, 'fixture', result)
            inventory = json.loads((root / 'fixture_inventory.json').read_text())
            self.assertEqual(inventory['/plddt']['shape'], [2])
            self.assertEqual(inventory['/complex/atom_positions']['shape'], [2, 3])
            self.assertTrue(all(v['finite'] for v in inventory.values()))
            loaded = torch.load(root / 'fixture.pt', weights_only=False)
            self.assertIsInstance(loaded, FixtureResult)
            np.testing.assert_array_equal(loaded.complex.atom_positions, result.complex.atom_positions)
            self.assertTrue(torch.equal(loaded.plddt, result.plddt))

    def test_nested_numpy_nonfinite_is_reported(self):
        result = FixtureResult(FixtureComplex(np.array([[np.nan, 0., 0.]])), torch.ones(1))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_packet(root, 'fixture', result)
            inventory = json.loads((root / 'fixture_inventory.json').read_text())
            self.assertFalse(inventory['/complex/atom_positions']['finite'])

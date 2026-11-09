"""CPU-only unit tests for cofolding.common.ipsae (numpy only, no torch/GPU)."""
import sys
import unittest

import numpy as np

from cofolding.common import compute_interface_scores
from cofolding.common.ipsae import d0_from_length, ptm_term

CUTOFFS = (10.0, 15.0)


# ---------------------------------------------------------------------------
# independent, deliberately naive reference implementation (loops, no
# vectorised tricks) used to cross-check the module
# ---------------------------------------------------------------------------
def naive_units(tokens):
    units, seen = [], {}
    for idx, tok in enumerate(tokens):
        if tok["kind"] in ("protein", "dna", "rna"):
            key = ("p", tok["chain"], tok["residue_index"])
        else:
            key = ("n", tok["chain"], idx)
        if key in seen:
            seen[key].append(idx)
        else:
            seen[key] = [idx]
            units.append((tok["chain"], tok["kind"], key))
    out = {}
    for chain, kind, key in units:
        out.setdefault(chain, []).append({"kind": kind, "tokens": seen[key]})
    return out


def naive_d0(n, minimum=1.0):
    if n <= 15:
        return max(minimum, 1.0)
    return max(minimum, 1.24 * (n - 15) ** (1.0 / 3.0) - 1.8)


def naive_direction(pae, frames, scored, cutoff, d0_fixed=None):
    uses_cutoff = any(unit["kind"] in ("protein", "dna", "rna") for unit in scored)
    best_score, best_n0, best_d0, best_ipae = None, None, None, None
    for frame in frames:
        row = pae[frame["rep"]]
        contributions, n0, protein_units = [], 0, 0
        for unit in scored:
            values = [row[t] for t in unit["tokens"]]
            keep = [v for v in values if v <= cutoff] if uses_cutoff else list(values)
            if not keep:
                continue
            n0 += 1
            if unit["kind"] == "protein":
                protein_units += 1
            contributions.append((keep, unit["kind"]))
        if d0_fixed is not None:
            d0 = d0_fixed
        elif protein_units:
            d0 = naive_d0(protein_units)
        else:
            d0 = 2.0
        if n0 == 0:
            score, ipae = 0.0, None
        else:
            per_unit = [sum(1.0 / (1.0 + (v / d0) ** 2) for v in keep) / len(keep)
                        for keep, _ in contributions]
            score = sum(per_unit) / len(per_unit)
            ipae = sum(sum(keep) / len(keep) for keep, _ in contributions) / len(contributions)
        if best_score is None or score > best_score:
            best_score, best_n0, best_d0, best_ipae = score, n0, d0, ipae
    return {"score": best_score, "n0": best_n0, "d0": best_d0, "ipae": best_ipae}


def naive_pair(pae, tokens, chain_a, chain_b, cutoff):
    units = naive_units(tokens)

    def frames(chain):
        for unit in units[chain]:
            if unit["kind"] != "ion":
                yield {"rep": unit["tokens"][0]}

    ab = naive_direction(pae, list(frames(chain_a)), units[chain_b], cutoff)
    ba = naive_direction(pae, list(frames(chain_b)), units[chain_a], cutoff)
    return ab, ba


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def protein_tokens(chain, n, start=1):
    return [{"kind": "protein", "chain": chain, "residue_index": start + i,
             "atom_name": "CA"} for i in range(n)]


def two_chain_pae(n_a, n_b, value_ab=2.0, value_ba=None, background=22.0):
    value_ba = value_ab if value_ba is None else value_ba
    n = n_a + n_b
    pae = np.full((n, n), background)
    pae[:n_a, :n_a] = 1.0
    pae[n_a:, n_a:] = 1.0
    pae[:n_a, n_a:] = value_ab
    pae[n_a:, :n_a] = value_ba
    np.fill_diagonal(pae, 0.0)
    return pae


class ScalarDefinitionTests(unittest.TestCase):
    def test_d0_matches_published_formula(self):
        self.assertEqual(d0_from_length(1), 1.0)
        self.assertEqual(d0_from_length(15), 1.0)
        self.assertEqual(d0_from_length(27), 1.24 * 12 ** (1 / 3) - 1.8)
        self.assertAlmostEqual(d0_from_length(40), 1.8257819953839538, places=12)
        self.assertAlmostEqual(d0_from_length(100, 2.0), 1.24 * 85 ** (1 / 3) - 1.8)
        self.assertEqual(d0_from_length(10, 2.0), 2.0)

    def test_ptm_term(self):
        self.assertAlmostEqual(float(ptm_term(2.0, 2.0)), 0.5)
        self.assertAlmostEqual(float(ptm_term(0.0, 3.0)), 1.0)
        self.assertAlmostEqual(float(ptm_term(np.array([[2.0, 4.0]]), 2.0)[0, 1]), 0.2)


class TwoChainTests(unittest.TestCase):
    def test_hand_computed_clean_interface(self):
        # 4 + 4 residues.  A->B: three residues at 2 A inside the cutoff and one
        # far away; B->A identical.  d0 = 1.0 (n = 4 protein units <= 15).
        n = 8
        pae = np.full((n, n), 30.0)
        pae[:4, :4] = pae[4:, 4:] = 1.0
        pae[:3, 4:7] = 2.0
        pae[4:7, :3] = 2.0
        np.fill_diagonal(pae, 0.0)
        tokens = protein_tokens("A", 4) + protein_tokens("B", 4)
        result = compute_interface_scores(pae, tokens)
        block = result[("A", "B")]["by_cutoff"][10.0]
        tm = 1.0 / (1.0 + (2.0 / 1.0) ** 2)
        # every frame of A sees three B residues at 2 A (mean = tm) except the
        # frames that see none, which score 0
        self.assertAlmostEqual(block["ipsae12"], tm, places=12)
        self.assertAlmostEqual(block["ipsae21"], tm, places=12)
        self.assertAlmostEqual(block["ipsae"], tm, places=12)
        self.assertEqual(block["ipsae_source"], "ipsae12")
        self.assertEqual(block["n0_12"], 3)
        self.assertEqual(block["d0_12"], 1.0)
        self.assertAlmostEqual(block["ipae12"], 2.0)
        self.assertEqual(block["n0chn"], 8)
        self.assertAlmostEqual(block["d0chn"], 1.0)
        self.assertEqual(block["n0dom"], 6)  # 3 + 3 units with any PAE <= 10
        # second cutoff sees the far corner too (30 A is still above 15)
        self.assertEqual(result[("A", "B")]["by_cutoff"][15.0]["n0_12"], 3)

    def test_max_over_frames_not_global_mean(self):
        """A disordered tail must not drag the score down."""
        n_a, n_b = 12, 6
        pae = np.full((n_a + n_b,) * 2, 22.0)
        pae[:n_a, :n_a] = pae[n_a:, n_a:] = 1.0
        pae[:4, n_a:] = 2.0
        pae[n_a:, :4] = 2.0
        pae[4:n_a, n_a:] = 24.0  # tail rows are confidently wrong
        pae[n_a:, 4:n_a] = 24.0
        np.fill_diagonal(pae, 0.0)
        tokens = protein_tokens("A", n_a) + protein_tokens("B", n_b)
        with_tail = compute_interface_scores(pae, tokens)[("A", "B")]["by_cutoff"][10.0]

        trimmed = np.full((4 + n_b,) * 2, 22.0)
        trimmed[:4, :4] = trimmed[4:, 4:] = 1.0
        trimmed[:4, 4:] = 2.0
        trimmed[4:, :4] = 2.0
        np.fill_diagonal(trimmed, 0.0)
        token_trim = protein_tokens("A", 4) + protein_tokens("B", n_b)
        without = compute_interface_scores(trimmed, token_trim)[("A", "B")]["by_cutoff"][10.0]

        self.assertAlmostEqual(with_tail["ipsae12"], without["ipsae12"], places=12)
        self.assertEqual(with_tail["n0_12"], without["n0_12"])
        # a whole-block mean of the TM term (no frame selection) is much lower
        block = pae[:n_a, n_a:]
        whole_block_mean = float(ptm_term(block, 1.0).mean())
        self.assertLess(whole_block_mean, with_tail["ipsae12"])

    def test_same_type_pair_rule_takes_max(self):
        n = 6 + 6
        pae = two_chain_pae(6, 6, value_ab=2.0, value_ba=18.0)
        pae[np.diag_indices(n)] = 0.0
        tokens = protein_tokens("A", 6) + protein_tokens("B", 6)
        block = compute_interface_scores(pae, tokens)[("A", "B")]["by_cutoff"][10.0]
        self.assertGreater(block["ipsae12"], block["ipsae21"])
        self.assertAlmostEqual(block["ipsae"], max(block["ipsae12"], block["ipsae21"]))
        self.assertEqual(block["ipsae_source"], "ipsae12")

    def test_never_symmetrised(self):
        n = 6 + 6
        pae = two_chain_pae(6, 6, value_ab=2.0, value_ba=18.0)
        pae[np.diag_indices(n)] = 0.0
        tokens = protein_tokens("A", 6) + protein_tokens("B", 6)
        forward = compute_interface_scores(pae, tokens)[("A", "B")]["by_cutoff"][10.0]
        reverse = compute_interface_scores(pae.T.copy(), tokens)[("A", "B")]["by_cutoff"][10.0]
        self.assertAlmostEqual(forward["ipsae12"], reverse["ipsae21"], places=12)
        self.assertAlmostEqual(forward["ipsae21"], reverse["ipsae12"], places=12)

    def test_interface_pae_statistics(self):
        n = 4 + 4
        pae = np.full((n, n), 30.0)
        pae[:4, :4] = pae[4:, 4:] = 1.0
        pae[0, 4:] = [2.0, 4.0, 6.0, 12.0]
        np.fill_diagonal(pae, 0.0)
        tokens = protein_tokens("A", 4) + protein_tokens("B", 4)
        pair = compute_interface_scores(pae, tokens)[("A", "B")]
        stats = pair["interface_pae"]
        self.assertEqual(stats["n_pairs"], 16)
        self.assertEqual(stats["n_below"][10.0], 3)
        self.assertEqual(stats["n_below"][15.0], 4)
        self.assertAlmostEqual(stats["fraction_below"][10.0], 3 / 16)
        restricted = pair["by_cutoff"][10.0]["interface_pae"]
        self.assertEqual(restricted["count"], 3)
        self.assertAlmostEqual(restricted["mean"], 4.0)


class UnitTests(unittest.TestCase):
    def test_multi_token_residue_counts_once(self):
        """Three atom tokens of one modified residue are one scored unit."""
        # chain A: 2 protein residues; chain B: one modified residue tokenised as
        # three atom tokens, PAE 2, 6, 20 -> one unit, mean TM over the two that
        # pass the 10 A cutoff
        tokens = [
            {"kind": "protein", "chain": "A", "residue_index": 1, "atom_name": "CA"},
            {"kind": "protein", "chain": "A", "residue_index": 2, "atom_name": "CA"},
            {"kind": "protein", "chain": "B", "residue_index": 1, "atom_name": "N"},
            {"kind": "protein", "chain": "B", "residue_index": 1, "atom_name": "CA"},
            {"kind": "protein", "chain": "B", "residue_index": 1, "atom_name": "C"},
        ]
        pae = np.array([
            [0.0, 30.0, 2.0, 6.0, 20.0],
            [30.0, 0.0, 2.0, 6.0, 20.0],
            [2.0, 2.0, 0.0, 1.0, 1.0],
            [6.0, 6.0, 1.0, 0.0, 1.0],
            [20.0, 20.0, 1.0, 1.0, 0.0],
        ])
        block = compute_interface_scores(pae, tokens)[("A", "B")]["by_cutoff"][10.0]
        self.assertEqual(block["n0_12"], 1)
        tm = (1 / (1 + (2 / 1) ** 2) + 1 / (1 + (6 / 1) ** 2)) / 2
        self.assertAlmostEqual(block["ipsae12"], tm, places=12)
        self.assertAlmostEqual(block["ipae12"], 4.0)

    def test_ligand_fixed_d0_and_no_cutoff(self):
        tokens = protein_tokens("A", 3) + [
            {"kind": "ligand", "chain": "L", "residue_index": None, "atom_name": "C1"},
            {"kind": "ligand", "chain": "L", "residue_index": None, "atom_name": "C2"},
        ]
        pae = np.zeros((5, 5))
        pae[:3, 3:] = [[50.0, 40.0], [40.0, 50.0], [50.0, 40.0]]
        pae[3:, :3] = 2.0
        block = compute_interface_scores(pae, tokens)[("A", "L")]["by_cutoff"][10.0]
        self.assertEqual(block["pae_cutoff12"], None)
        self.assertEqual(block["d0_12"], 2.0)
        self.assertEqual(block["n0_12"], 2)  # both ligand atoms, no cutoff
        expected = (1 / (1 + (50 / 2) ** 2) + 1 / (1 + (40 / 2) ** 2)) / 2
        self.assertAlmostEqual(block["ipsae12"], expected, places=12)
        self.assertEqual(block["ipsae_source"], "ipsae12")  # protein-aligned

    def test_ion_is_never_a_frame_and_ion_ion_is_skipped(self):
        tokens = protein_tokens("A", 3) + [
            {"kind": "ion", "chain": "M", "residue_index": None, "atom_name": "MG"},
            {"kind": "ion", "chain": "N", "residue_index": None, "atom_name": "ZN"},
        ]
        pae = np.zeros((5, 5))
        pae[:3, 3] = 3.0
        pae[3, :3] = 3.0
        result = compute_interface_scores(pae, tokens)
        protein_ion = result[("A", "M")]["by_cutoff"][10.0]
        self.assertIsNone(result[("A", "M")]["skipped"])
        self.assertIsNone(protein_ion["ipsae21"])  # ion cannot align
        self.assertEqual(protein_ion["ipsae12"], 1 / (1 + (3 / 2) ** 2))
        ion_ion = result[("M", "N")]
        self.assertIn("ion-ion", ion_ion["skipped"])
        self.assertEqual(ion_ion["by_cutoff"][10.0]["skipped"], ion_ion["skipped"])

    def test_frame_valid_override(self):
        tokens = protein_tokens("A", 3) + protein_tokens("B", 2)
        for tok in tokens[:2]:
            tok["frame_valid"] = False
        tokens[2]["frame_valid"] = True
        tokens[3]["frame_valid"] = True
        tokens[4]["frame_valid"] = True
        pae = np.zeros((5, 5))
        pae[0, 3:] = [2.0, 8.0]   # invalid frame: would score 0.031
        pae[2, 3:] = [8.0, 8.0]   # valid frame
        block = compute_interface_scores(pae, tokens)[("A", "B")]["by_cutoff"][10.0]
        expected = 1 / (1 + (8 / 1) ** 2)
        self.assertAlmostEqual(block["ipsae12"], expected, places=12)


class MixedTypeRuleTests(unittest.TestCase):
    def test_nucleic_pair_uses_protein_direction(self):
        tokens = protein_tokens("A", 4) + [
            {"kind": "dna", "chain": "D", "residue_index": i + 1, "atom_name": "C1'"}
            for i in range(3)]
        pae = np.zeros((7, 7))
        pae[:4, 4:] = 8.0   # protein-aligned direction: weaker
        pae[4:, :4] = 1.0   # DNA-aligned direction: stronger
        block = compute_interface_scores(pae, tokens)[("A", "D")]["by_cutoff"][10.0]
        self.assertGreater(block["ipsae21"], block["ipsae12"])
        self.assertAlmostEqual(block["ipsae"], block["ipsae12"])
        self.assertEqual(block["ipsae_source"], "ipsae12")
        self.assertEqual(block["d0_12"], 2.0)  # nucleic acid scored, fixed 2 A

    def test_chain_order_follows_molecule_complexity(self):
        tokens = (protein_tokens("P", 2)
                  + [{"kind": "ligand", "chain": "L", "residue_index": None,
                      "atom_name": "C1"}]
                  + [{"kind": "ion", "chain": "M", "residue_index": None,
                      "atom_name": "MG"}])
        pae = np.zeros((4, 4))
        result = compute_interface_scores(pae, tokens)
        self.assertEqual(list(result)[0], ("P", "L"))
        self.assertEqual(list(result)[1], ("P", "M"))
        self.assertEqual(list(result)[2], ("L", "M"))


class NaiveAgreementTests(unittest.TestCase):
    """Vectorised module vs an independent loop implementation."""

    def test_matches_naive_on_random_asymmetric_matrices(self):
        rng = np.random.default_rng(20261109)
        for trial in range(6):
            tokens = (protein_tokens("A", 5 + trial)
                      + protein_tokens("B", 4)
                      + [{"kind": "ligand", "chain": "L", "residue_index": None,
                          "atom_name": f"C{k}"} for k in range(3)])
            n = len(tokens)
            pae = rng.uniform(0.0, 30.0, size=(n, n))
            np.fill_diagonal(pae, 0.0)
            result = compute_interface_scores(pae, tokens)
            for chain_a, chain_b in (("A", "B"), ("A", "L"), ("B", "L")):
                ab, ba = naive_pair(pae, tokens, chain_a, chain_b, 10.0)
                block = result[(chain_a, chain_b)]["by_cutoff"][10.0]
                with self.subTest(trial=trial, pair=(chain_a, chain_b)):
                    if block["ipsae12"] is not None:
                        self.assertAlmostEqual(block["ipsae12"], ab["score"], places=10)
                        self.assertEqual(block["n0_12"], ab["n0"])
                        self.assertAlmostEqual(block["d0_12"], ab["d0"], places=10)
                        if ab["ipae"] is not None:
                            self.assertAlmostEqual(block["ipae12"], ab["ipae"], places=10)
                    self.assertAlmostEqual(block["ipsae21"], ba["score"], places=10)
                    self.assertEqual(block["n0_21"], ba["n0"])


class ErrorTests(unittest.TestCase):
    def test_shape_mismatch(self):
        with self.assertRaises(ValueError) as ctx:
            compute_interface_scores(np.zeros((3, 3)), protein_tokens("A", 2))
        self.assertIn("token", str(ctx.exception))

    def test_non_square(self):
        with self.assertRaises(ValueError):
            compute_interface_scores(np.zeros((3, 4)), protein_tokens("A", 3))

    def test_polymer_without_residue_index(self):
        tokens = protein_tokens("A", 2) + [
            {"kind": "protein", "chain": "B", "residue_index": None, "atom_name": "CA"}]
        with self.assertRaises(ValueError) as ctx:
            compute_interface_scores(np.zeros((3, 3)), tokens)
        message = str(ctx.exception)
        self.assertIn("residue_index", message)
        self.assertIn("token 2", message)

    def test_missing_chain(self):
        tokens = protein_tokens("A", 2) + [
            {"kind": "protein", "chain": "", "residue_index": 1, "atom_name": "CA"}]
        with self.assertRaises(ValueError) as ctx:
            compute_interface_scores(np.zeros((3, 3)), tokens)
        self.assertIn("chain", str(ctx.exception))

    def test_unknown_kind(self):
        tokens = protein_tokens("A", 2) + [
            {"kind": "peptide", "chain": "B", "residue_index": 1}]
        with self.assertRaises(ValueError) as ctx:
            compute_interface_scores(np.zeros((3, 3)), tokens)
        self.assertIn("kind", str(ctx.exception))

    def test_chain_mixing_kinds(self):
        tokens = protein_tokens("A", 2) + [
            {"kind": "dna", "chain": "A", "residue_index": 1, "atom_name": "C1'"}]
        with self.assertRaises(ValueError) as ctx:
            compute_interface_scores(np.zeros((3, 3)), tokens)
        self.assertIn("mixes", str(ctx.exception))

    def test_nonfinite_ligand_pae_skips_the_pair(self):
        """Ligands are scored without a cutoff, so a non-finite value cannot be
        dropped: the pair is reported as skipped with a reason (not scored)."""
        tokens = protein_tokens("A", 2) + [
            {"kind": "ligand", "chain": "L", "residue_index": None, "atom_name": "C1"}]
        pae = np.zeros((3, 3))
        pae[0, 2] = np.nan
        pair = compute_interface_scores(pae, tokens)[("A", "L")]
        self.assertIn("non-finite", pair["skipped"].lower())
        self.assertEqual(pair["by_cutoff"][10.0]["skipped"], pair["skipped"])


class PurityTests(unittest.TestCase):
    def test_no_torch_or_cuda_imported(self):
        # None covers a caller that deliberately poisoned sys.modules["torch"];
        # a real module object here means something imported torch.
        self.assertIsNone(sys.modules.get("torch"))
        from cofolding.common import ipsae
        with open(ipsae.__file__, encoding="utf-8") as fh:
            source = fh.read()
        for forbidden in ("import torch", "from torch", "cupy", "tensorflow",
                          "import gemmi", "import pandas", "import scipy"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()

# ipSAE / ipSAE2 interface scoring: implementation validation

`cofolding/common/ipsae.py` computes per-chain-pair interface confidence from a
predicted aligned error (PAE) matrix, so a cofolding container can derive those
metrics inside the container before the PAE array is dropped. It is pure numpy
and imports nothing that needs a GPU, torch, gemmi, pandas or scipy.

This document records how the implementation was validated against the two
Dunbrack laboratory reference implementations, on synthetic cases and on two
completed predictions from the experiment runs.

Status: **qualified for `ipsae`, `ipsae12`, `ipsae21`, `iPAE`, `n0`, `d0`,
`ipsae_d0chn`, `ipsae_d0dom` and `iptm_d0chn`.** The only column that does not
agree with a reference is `ipsae_d0chn` / `ipsae_d0dom` of the ipSAE v1 code on
the protein-DNA case, and that difference is fully explained by a deliberate
cutoff-boundary rule (see [Deliberate differences](#deliberate-differences),
item 1). No formula was tuned to fit the reference.

## Reference implementations

Both references were cloned into the scratch tree and run from source; neither
was installed into the environment.

```bash
mkdir -p /home/dan/.hermes/cache/scratch/ipsae-ref
cd /home/dan/.hermes/cache/scratch/ipsae-ref
git clone https://github.com/DunbrackLab/IPSAE.git
git clone https://github.com/DunbrackLab/IPSAE2.git
git -C IPSAE rev-parse HEAD    # 6174cf9e71cb1bd660cc805856a18c4871a6dec3
git -C IPSAE2 rev-parse HEAD   # df6e46f2b7a2b49c8751556cb2ee24c336d50db4
```

| repository | commit | version used |
|---|---|---|
| `DunbrackLab/IPSAE` (`ipsae.py`, the original ipSAE) | `6174cf9e71cb1bd660cc805856a18c4871a6dec3` | single-file script |
| `DunbrackLab/IPSAE2` (`src/ipsae`, the directional ipSAE2) | `df6e46f2b7a2b49c8751556cb2ee24c336d50db4` | `__version__ = "0.1.10"` |

`IPSAE/ipsae.py` is the source of the `d0chn` / `d0dom` / `iptm_d0chn`
definitions. `IPSAE2` is the source of the directional `ipsae12` / `ipsae21` /
`ipsae` rule, the type hierarchy, the per-cutoff d0, and ligand/ion handling.

## Module surface

```python
from cofolding.common import compute_interface_scores

result = compute_interface_scores(pae, tokens, cutoffs=(10.0, 15.0))
```

* `pae`: `(N, N)` float array at token resolution. `pae[i, j]` is the predicted
  error of token `j` when the alignment is on token `i`; **rows are alignment
  frames**, and the matrix is never symmetrised.
* `tokens`: list of `N` dicts, in PAE order, with `kind`
  (`protein|dna|rna|ligand|ion`) and `chain`. Polymer tokens also need
  `residue_index` (int) so that tokens are grouped into residues; `atom_name` is
  only used to pick the alignment representative (`CA`, `C3'`, `C1'`). An
  optional `frame_valid` bool (set on every token of a unit, or on none)
  honours a predictor's own frame mask.
* Returns a dict keyed by the ordered `(chain_a, chain_b)` tuple, ordered by
  molecule complexity (protein, nucleic acid, ligand, ion) and then by first
  appearance. Each entry holds the directional scores `ipsae12` / `ipsae21`, the
  pair value `ipsae` (same-type: max; mixed-type: the direction aligning on the
  more complex molecule), `ipsae_d0chn` / `ipsae_d0dom` / `iptm_d0chn` with
  their directional and raw-max variants, the counts and d0 values that
  produced them, per-pair interface PAE statistics (`mean`, `median`,
  `n_pairs`, `n_below` / `fraction_below` per cutoff), and a `skipped` reason
  for any pair that cannot be scored (ion-ion, no usable frame, non-finite PAE
  against an uncut ligand/ion unit).

`n0_12` / `n0_21` / `d0_12` / `d0_21` are the scored-unit count and d0 **at the
maximising alignment frame**, matching the ipSAE2 TSV; `n0chn` / `d0chn` and
`n0dom` / `d0dom` are the original ipSAE length definitions.

## Reproducing the comparison

All scratch scripts live in `/home/dan/.hermes/cache/scratch/ipsae-val/`:

| script | purpose |
|---|---|
| `make_cases.py` | mmCIF/PAE writer (Boltz-style columns; one heavy atom set per residue) |
| `define_cases.py` | builds the seven synthetic cases under `cases/` |
| `compare.py` | runs both references on each case and diffs them against the module → `results.json` |
| `invariants.py` | module-only invariants (tail, asymmetry, mixed-type rule, ions) → `invariants.json` |
| `tokens_from_cif.py` | CIF → module token list (gemmi, scratch only) |
| `real_compare.py` | same comparison on the two real predictions → `real_results_<label>.json` |

```bash
cd /home/dan/.hermes/cache/scratch/ipsae-val
PY=/home/dan/swr/miniconda3/envs/cheminf/bin/python
$PY define_cases.py
$PY compare.py
$PY invariants.py
$PY real_compare.py
```

Per case, `compare.py` runs the references exactly as a user would:

```bash
# ipSAE2 (directional)
cd /home/dan/.hermes/cache/scratch/ipsae-val/cases/clean-interface
PYTHONPATH=/home/dan/.hermes/cache/scratch/ipsae-ref/IPSAE2/src \
  $PY -m ipsae clean-interface.cif pae_clean-interface.npz --all-pairs \
      --output ipsae2.tsv

# ipSAE v1 (d0chn / d0dom / iptm_d0chn), default cutoffs 10 A / 5 A
$PY /home/dan/.hermes/cache/scratch/ipsae-ref/IPSAE/ipsae.py \
  pae_clean-interface.npz clean-interface.cif 10 5
# -> clean-interface_10_05.txt
```

and feeds the module `pae_<case>.npz` plus the token list in
`cases/<case>/meta.json`.

Both references are run with the published defaults (PAE cutoff 10 A, fixed d0
2 A, protein d0 floor 1 A); only the 10 A cutoff therefore has reference
numbers. The module is also exercised at 15 A by the tests and by
`results.json`, but that column set has no reference counterpart.

## Synthetic cases

Each case is a small mmCIF with explicit heavy atoms (N, CA, C, O, CB for
protein; C1', C3', C4', P, O3' for DNA; ring atoms for ligands; one atom for
ions) plus a synthetic token-resolution PAE matrix. Geometry is deliberately
sane so that ipSAE2's three-atom frame detection and its 5 A contact filter both
succeed. Interface PAE values follow a deterministic ramp (base .. base + 4 A)
so that rows differ and the max-over-frames choice actually matters.

| case | tokens | what it is for |
|---|---|---|
| `clean-interface` | 48 | two 24-residue proteins; residues 5-20 of each at 2.5-6.5 A PAE |
| `disordered-tail` | 64 | chain A = 28 ordered residues + 12-residue tail at 24-26 A PAE to B |
| `disordered-tail-truncated` | 52 | the same ordered domain alone: the tail must not change the score |
| `protein-ligand` | 18 | 12-residue protein + 6-atom ligand on its own chain |
| `asymmetric` | 32 | A rows see B at 1.5-5.5 A, B rows see A at 16-20 A |
| `protein-dna` | 24 | protein + DNA where the DNA-aligned direction scores higher |
| `ions` | 10 | protein + Mg + Zn (ion-ion must be skipped) |

Module values at cutoff 10 A (`results.json`) for orientation:

| case | pair | units | ipsae | ipsae12 | ipsae21 | d0chn | d0dom |
|---|---|---|---|---|---|---|---|
| clean-interface | A-B | 24/24 | 0.0677925 | 0.0647831 | 0.0677925 | 2.1773 | 1.8258 |
| disordered-tail | A-B | 40/24 | 0.0792717 | 0.0647831 | 0.0792717 | 2.7375 | 2.3320 |
| disordered-tail-truncated | A-B | 28/24 | 0.0792717 | 0.0647831 | 0.0792717 | 2.3320 | 2.3320 |
| protein-ligand | A-L | 12/6 | 0.2909372 | 0.2909372 | 0.0280649 | 1.0000 | 1.0000 |
| asymmetric | A-B | 16/16 | 0.1317596 | 0.1317596 | 0.0000000 | 1.3884 | 1.3884 |
| protein-dna | A-D | 14/10 | 0.0549539 | 0.0549539 | 0.1895281 | 2.0000 | 2.0000 |
| ions | A-M / A-N | 8/1 | 0.3076923 | 0.3076923 | n/a | 1.0000 | 1.0000 |
| ions | M-N | 1/1 | skipped: `ion-ion pairs are not scored (no frame)` | | | | |

Behaviours confirmed by these cases (`invariants.json`):

* the disordered tail leaves `ipsae12` bit-identical to the truncated model
  (`0.0647830554066` in both) while a whole-block TM mean over the same matrix
  is far lower — i.e. the score is a max over alignment frames, not an average;
* `asymmetric` reproduces `ipsae == max(ipsae12, ipsae21)` and `ipsae21 == 0`
  (all cross values above 10 A) with `ipsae_source == "ipsae12"`;
* `protein-dna` has `ipsae21 = 0.1895 > ipsae12 = 0.0550`, and the mixed-type
  rule still reports `ipsae12` (`ipsae_source == "ipsae12"`);
* the ligand direction uses `d0 = 2.0` with no cutoff (`pae_cutoff12 = None`)
  and scores all 6 ligand atoms; the reverse direction scores the 12 protein
  residues with the 10 A cutoff and `d0 = 1.0`;
* an ion is never an alignment frame (`ipsae21` is `None` on protein-ion pairs)
  and an ion-ion pair is skipped rather than reported as zero.

## Results: module vs ipSAE2

Maximum absolute difference per column over all pairs and both directions, at
cutoff 10 A. `0` means exact agreement of the printed value.

| case | ipsae | ipsae12 | ipsae21 | n0_12 | n0_21 | d0_12 | d0_21 | iPAE12 | iPAE21 |
|---|---|---|---|---|---|---|---|---|---|
| clean-interface | 4.9e-7 | 5.5e-8 | 4.9e-7 | 0 | 0 | 0 | 0 | 3.3e-4 | 0 |
| disordered-tail | 2.9e-7 | 5.5e-8 | 2.9e-7 | 0 | 0 | 0 | 3.4e-4 | 3.3e-4 | 4.3e-4 |
| disordered-tail-truncated | 2.9e-7 | 5.5e-8 | 2.9e-7 | 0 | 0 | 0 | 3.4e-4 | 3.3e-4 | 4.3e-4 |
| protein-ligand | 2.2e-7 | 2.2e-7 | 9.3e-8 | 0 | 0 | 0 | 0 | 3.3e-4 | 3.3e-4 |
| asymmetric | 3.8e-7 | 3.8e-7 | 0 | 0 | 0 | 0 | 0 | 0 | – |
| protein-dna | 1.0e-7 | 1.0e-7 | 1.2e-7 | 0 | 0 | 0 | 0 | 0 | 1.4e-4 |
| ions | 3.1e-7 | 3.1e-7 | – | 0 | – | 0 | – | 0 | – |

These are print-precision differences, not numerical disagreement: the ipSAE2
TSV prints `ipsae` to six decimals (half-ulp 5e-7), `iPAE` and `d0` to three
decimals (half-ulp 5e-4). Every score is inside half an ulp of the printed
reference value; counts are exact. Recomputing the module's `d0` from the
reference's own `n0` gives a difference of exactly `0` everywhere (checked for
`d0chn`, `d0dom12` in `compare.py`).

## Results: module vs ipSAE v1 (`d0chn` / `d0dom` / `iptm_d0chn`)

Only the protein-protein cases can be compared: ipSAE v1 has no ligand or ion
support (it drops non-CA ligand atoms from the token mask, so `protein-ligand`
and `ions` produce no pairs there).

`12` is align-on-`chain_a`, `21` is align-on-`chain_b`; `*_max` is the larger of
the two directions, which is what the ipSAE `max` row prints.

| case | d0chn12 | d0chn21 | d0dom12 | d0dom21 | iptm12 | iptm21 | d0chn_max | d0dom_max | iptm_max | n0chn | n0dom12 | n0dom21 | d0chn | d0dom12 | d0dom21 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean-interface | 4.3e-7 | 3.0e-7 | 4.4e-7 | 1.0e-8 | 4.3e-7 | 4.4e-7 | 3.0e-7 | 1.0e-8 | 4.3e-7 | 0 | 0 | 0 | 2.7e-3 | 4.2e-3 | 4.2e-3 |
| disordered-tail | 2.0e-9 | 8.3e-8 | 1.5e-7 | 3.9e-7 | 2.0e-9 | 3.9e-7 | 8.3e-8 | 3.9e-7 | 2.0e-9 | 0 | 0 | 0 | 2.5e-3 | 2.0e-3 | 2.0e-3 |
| disordered-tail-truncated | 1.5e-7 | 3.9e-7 | 1.5e-7 | 3.9e-7 | 1.5e-7 | 3.9e-7 | 3.9e-7 | 3.9e-7 | 3.9e-7 | 0 | 0 | 0 | 2.0e-3 | 2.0e-3 | 2.0e-3 |
| asymmetric | 1.9e-7 | 0 | 1.9e-7 | 0 | 1.9e-7 | 2.5e-7 | 1.9e-7 | 1.9e-7 | 1.9e-7 | 0 | 0 | 0 | 1.6e-3 | 1.6e-3 | 0 |
| protein-dna | **5.5e-3** | 7.1e-8 | **5.5e-3** | 7.1e-8 | 1.2e-7 | 7.1e-8 | 7.1e-8 | 7.1e-8 | 7.1e-8 | 0 | 0 | 0 | 0 | 0 | 0 |

`n0chn`, `n0dom12/21` agree exactly. The `d0*` columns differ by at most
`4.2e-3` only because the ipSAE v1 table prints d0 with **two** decimals
(half-ulp 5e-3); recomputing d0 from the reference's own `n0` gives exactly 0.

The one real discrepancy is `protein-dna`, `ipsae_d0chn12` and
`ipsae_d0dom12`: module `0.0549539` vs reference `0.060451`. That case
deliberately contains 28 interchain PAE values of exactly `10.0` A (see
`pae_values_exactly_at_cutoff` in `results.json`), and ipSAE v1 uses a strict
`<`, while ipSAE2 and this module use `<=`. On one row the interface values are
`7, 8, 9, 10, 11` twice; with `<=` eight units are selected
(`7,8,9,10` x2) and the 10 A units pull the mean TM down, with `<` only six are
selected (`7,8,9` x2) and the reference scores higher. Recomputing the mean TM
over v1's own six selected units reproduces its `0.060451` exactly, so the
difference is the boundary rule and nothing else. **This column is therefore not
comparable to ipSAE v1 on inputs that contain PAE values equal to the cutoff;
the module follows ipSAE2 there, and the case is left as an intentional
mismatch rather than adjusted.** On the same case every ipSAE2 column agrees to
1.4e-4 or better, and `iptm_d0chn` (no cutoff) agrees to 1.2e-7, which localises
the difference to the boundary rule.

## Results: real predictions

Both completed runs contain a native mmCIF plus a PAE `.npz`, so both were used.

| run | prediction | PAE | chains | pairs compared |
|---|---|---|---|---|
| `/nvme1/dan/folding/runs/rc3-acceptance/.../native/` | `two-chain_sample_1.cif` | `(43, 43)` | A (20 res), B (20 res), L (3-atom ligand) | A-B (protein), A-L, B-L |
| `/nvme1/dan/folding/runs/boltz2-kernels-on/.../native/predictions/trpcage/` | `trpcage_model_0.cif` | `(23, 23)` | A (20 res), L (3-atom ligand) | A-L |

Commands (`real_compare.py` does this for both, including a scratch copy of the
confidence JSON with an empty `pair_chains_iptm` that ipSAE v1 indexes into for
its display-only `ipTM_af` column):

```bash
cd /home/dan/.hermes/cache/scratch/ipsae-val/real/rc3-acceptance-two-chain
PYTHONPATH=/home/dan/.hermes/cache/scratch/ipsae-ref/IPSAE2/src \
  $PY -m ipsae two-chain_sample_1.cif pae_two-chain_sample_1.npz --all-pairs \
      --output ipsae2.tsv
$PY /home/dan/.hermes/cache/scratch/ipsae-ref/IPSAE/ipsae.py \
  pae_two-chain_sample_1.npz two-chain_sample_1.cif 10 5
```

Module vs ipSAE2, maximum absolute difference per column:

| pair | ipsae | ipsae12 | ipsae21 | n0_12 | n0_21 | d0_12 | d0_21 | iPAE12 | iPAE21 |
|---|---|---|---|---|---|---|---|---|---|
| A-B | 3.6e-7 | 3.6e-7 | 1.2e-7 | 0 | 0 | 0 | 0 | 1.6e-4 | 1.9e-4 |
| A-L | 2.6e-7 | 2.6e-7 | 0 | 0 | 0 | 0 | 0 | 4.1e-4 | – |
| B-L | 1.1e-7 | 1.1e-7 | 5.9e-8 | 0 | 0 | 0 | 0 | 2.7e-4 | 6.0e-5 |
| trpcage A-L | 2.5e-7 | 2.5e-7 | 2.3e-7 | 0 | 0 | 0 | 0 | 2.3e-4 | 4.2e-4 |

Module vs ipSAE v1 (the A-B protein pair is the only one v1 can score):

| pair | d0chn12 | d0chn21 | d0dom12 | d0dom21 | iptm12 | iptm21 | d0chn_max | d0dom_max | iptm_max | n0chn | n0dom12 | n0dom21 | d0chn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-B (rc3) | 9.7e-8 | 4.5e-7 | 9.7e-8 | 6.5e-8 | 9.7e-8 | 4.5e-7 | 9.7e-8 | 9.7e-8 | 9.7e-8 | 0 | 0 | 0 | 4.2e-3 |

Reference TSVs for `rc3-acceptance` (verbatim):

```text
chn1	chn2	type1	type2	mol1	mol2	iptm	ipsae	ipsae12	ipsae21	ipsae_source	align_token12	align_token21	iPAE12	iPAE21	n0_12	n0_21	d0_12	d0_21	n_scored_contact12	n_scored_contact21	pae_cutoff12	pae_cutoff21	native_iptm12	native_iptm21
A	B	prot	prot	AA_20	AA_20	NA	0.049101	0.049101	0.047052	ipsae12	TRP:6:CA	TRP:19:CA	4.932	4.892	20	20	1.000	1.000	10	11	10.0	10.0	NA	NA
A	L	prot	lig	AA_20	LIG	NA	0.110221	0.110221	0.000000	ipsae12	TRP:6:CA	LIG:1:C8	5.688	NA	3	0	2.000	1.000	0	0	none	10.0	NA	NA
B	L	prot	lig	AA_20	LIG	NA	0.182163	0.182163	0.017711	ipsae12	ASP:3:CA	LIG:1:O7	4.266	7.739	3	7	2.000	1.000	3	7	none	10.0	NA	NA
```

```text
Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn     pDockQ     pDockQ2    LIS       n0res  n0chn  n0dom   d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model
A    B     10   05   asym  0.049101    0.142368    0.142368    0.000    0.142368      0.0539     0.1892     0.5259      20     40     40    1.00    1.83    1.83     20      20       2       2   two-chain_sample_1
B    A     10   05   asym  0.047052    0.138712    0.126198    0.000    0.138712      0.0539     0.1321     0.3339      20     40     38    1.00    1.83    1.73     18      20       2       2   two-chain_sample_1
A    B     10   05   max   0.049101    0.142368    0.142368    0.000    0.142368      0.0539     0.1892     0.4299      20     40     40    1.00    1.83    1.83     20      20       2       2   two-chain_sample_1
```

The real prediction reproduces the direction asymmetry that makes the
directional definitions necessary: v1's `n0dom` is 40 for A->B and 38 for
B->A on the same pair, and the module returns the same two values (module
`n0dom12 = 40`, `n0dom21 = 38`, `d0dom21 = 1.7264` vs printed `1.73`).

## Deliberate differences

1. **Cutoff boundary.** The module uses `PAE <= cutoff` (ipSAE2 semantics, and
   the `<=` in the published definition). ipSAE v1 uses `<`. The two only
   diverge when a PAE value equals the cutoff exactly; that is the entire
   `protein-dna` `ipsae_d0chn12` / `ipsae_d0dom12` mismatch above (module
   `0.0549539`, reference `0.060451`, difference `5.5e-3`). Real PAE values are
   floats and practically never land exactly on 10.0, but the synthetic case
   does so on purpose to pin the behaviour.
2. **Pair rule applied to the `d0chn` / `d0dom` / `iptm_d0chn` columns.** The
   task defines one pair rule (same type: max; different types: align on the
   more complex molecule). The module applies that rule to every pair-level
   column, so for a mixed-type pair `ipsae_d0chn` is the protein-aligned
   direction, whereas ipSAE v1's `max` row always reports the larger of the two
   directions regardless of type. Both numbers are exposed: `ipsae_d0chn_max`,
   `ipsae_d0dom_max` and `iptm_d0chn_max` reproduce v1's `max` row exactly (see
   the tables), and `ipsae_d0chn12/21` reproduce v1's `asym` rows. That is why
   the v1 comparison above uses `*_max` for pair-level values and `*12/21` for
   directional ones: on `protein-dna` the module's `ipsae_d0chn` is
   `0.0549539` (protein-aligned) while v1's `max` row is `0.4070670`
   (DNA-aligned), and both are correct under their own definition.
3. **No coordinate model inside the module.** ipSAE2 decides which tokens may
   act as alignment frames from geometry (N, CA, C for protein; C1', C3', C4'
   for nucleotides; two distinct neighbours for atom-tokenised ligands) and
   drops pairs with no 5 A heavy-atom contact. The module has no coordinates, so
   it treats every non-ion unit as a usable frame unless the caller supplies
   `frame_valid`, and it reports every chain pair instead of only contacting
   ones. Callers that have the structure can pass the predictor's own frame mask
   (`frame_valid`) to reproduce ipSAE2's frame selection exactly; that is the
   intended container wiring, since the container has the structures available
   but not necessarily gemmi.
4. **No contact filter and no `n_scored_contact`.** Contact counting needs
   coordinates and is out of scope; `n_scored_contact12/21` has no counterpart
   here. The interface PAE statistics replace it for ranking.
5. **Ligand and ion d0 in the `d0chn` family.** ipSAE v1 has no ligand/ion
   support at all. The module generalises `n0chn` to "units in both chains"
   (so a 3-atom ligand contributes 3) and keeps the published min-d0 floors
   (2 A as soon as either chain is nucleic acid, otherwise 1 A). For
   ligand-only scored sides the ipSAE2-style `d0_12` remains the fixed 2 A.
6. **Multi-token residues use one equal-weighted unit.** ipSAE2 has a token-level
   fast path (used when every scored unit is a single token) and a grouped slow
   path. They differ only in how `iPAE` is averaged over a multi-atom modified
   residue. The module always uses the grouped (residue-equal-weight) form,
   which is the slow path and the documented rule ("one scoring unit per
   residue"). For single-token units — all real cases here — the two agree
   exactly, which is why `iPAE` matches.
7. **`n0dom` / `d0dom` are directional.** ipSAE v1 accumulates the row condition
   and the column condition of one direction from the same block, so `n0dom` is
   not symmetric even though the *pair* is. The module computes both directions
   (`n0dom12`, `n0dom21`) and reports the one belonging to the selected
   direction, reproducing v1's 40 / 38 pair on the real prediction.
8. **`ipSAE` itself is not reported.** v1's headline `ipSAE` column is
   `ipsae_d0res` (a per-frame d0 from the number of scored partners of that
   residue), a different quantity from any column here. `ipsae12` equals v1's
   `ipSAE` only when d0 happens to coincide; on the real prediction it does
   (0.049101) because d0 is floored at 1 A in both, but that is not a general
   claim. Use `ipsae_d0chn` / `ipsae_d0dom` for the original ipSAE numbers.
9. **Failure mode for non-finite ligand PAE.** ipSAE2 raises. The module records
   the pair as skipped with the reason in `skipped` (and in each `by_cutoff`
   entry) so that a batch container can keep going, which is the same
   information without aborting the run. Token-placement errors (missing
   `chain`, missing `residue_index`, unknown `kind`, mixed-kind chains,
   PAE/token count mismatch) do raise `ValueError` immediately, naming the token
   index.

## Tests

```bash
cd /home/dan/projects/container-library
/home/dan/swr/miniconda3/envs/cheminf/bin/python -m unittest discover -s cofolding/common/tests
```

Output, verbatim:

```text
......................
----------------------------------------------------------------------
Ran 22 tests in 0.037s

OK
```

(timing varies by a few hundredths of a second between runs). The suite is
numpy-only. `torch` is *installed* in that environment, so the
suite was also run with the module banned to prove nothing in the import path or
the tests reaches for it:

```bash
cd /home/dan/projects/container-library
CUDA_VISIBLE_DEVICES= \
/home/dan/swr/miniconda3/envs/cheminf/bin/python -c "
import sys
sys.modules['torch'] = None          # any 'import torch' now raises ImportError
sys.modules['cupy'] = None
import unittest
suite = unittest.TestLoader().discover('cofolding/common/tests')
result = unittest.TextTestRunner(verbosity=1).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
"
```

```text
......................
----------------------------------------------------------------------
Ran 22 tests in 0.038s

OK
```

Coverage: published d0/TM constants and floors, a hand-computed two-chain
interface, max-over-frames vs a disordered tail, same-type max, mixed-type
protein direction, never-symmetrised (transposing PAE swaps `12`/`21`),
interface statistics, multi-token residue weighting, ligand fixed d0 with no
cutoff, ion frames and skipped ion-ion pairs, `frame_valid` overrides, chain
ordering by complexity, a randomised comparison against an independent
loop-based implementation, every validation error path, and an assertion that
the module source contains no torch/cupy/gemmi/pandas/scipy imports.

## Files added

```text
cofolding/common/__init__.py
cofolding/common/ipsae.py
cofolding/common/tests/__init__.py
cofolding/common/tests/test_ipsae.py
docs/ipsae-validation.md
```

Nothing is wired into either container's worker; the coordinator owns that.
Changes are left uncommitted.

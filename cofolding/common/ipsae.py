"""Pure-numpy ipSAE / ipSAE2 interface scoring from a predicted aligned error matrix.

This module reimplements the published interface scores of the Dunbrack
laboratory (ipSAE, and the directional ipSAE2 variant) without torch, gemmi,
pandas or scipy so that a cofolding container can compute them from the PAE
array before that array is discarded.

Conventions (all quantities are defined on *scored units*, never raw PAE
tokens):

* protein / DNA / RNA: one unit per residue, however many tokens the residue
  occupies.  A residue's contribution is the equal-weighted mean over its
  tokens, so the residue counts once in ``n0``.
* ligand / ion: one unit per token (per heavy atom).
* the alignment direction ``A -> B`` means: row ``i`` of the PAE matrix is the
  frame of a unit ``i`` in ``A``; the TM term is averaged over units ``j`` in
  ``B``.  ``pae[i, j]`` is the predicted error of ``j`` when aligned on ``i``.
* the PAE matrix is never symmetrised, averaged with its transpose, or
  transposed for convenience.

Definitions
-----------
``TM(x, d0) = 1 / (1 + (x / d0) ** 2)``

``d0(n) = max(1.0, 1.24 * (n - 15) ** (1/3) - 1.8)`` with ``n`` the number of
scored units used as the length.  Nucleic-acid d0 is floored at 2.0 A;
ligand/ion-only scored sides use a fixed d0 of 2.0 A.

``ipsae12 = max_{i in A} mean_{j in B, pae[i,j] <= cutoff} TM(pae[i,j], d0)``
(``ipsae21`` is the transpose direction).  The pair value ``ipsae`` is the
larger of the two directions for same-type pairs; for different types it is
the direction aligning on the more complex molecule, ordered
protein > nucleic acid > ligand > ion.

``ipsae_d0chn`` / ``ipsae_d0dom`` / ``iptm_d0chn`` follow the original ipSAE
definitions: ``d0chn`` from the summed length of both chains (no PAE cutoff
for ``iptm_d0chn``), ``d0dom`` from the number of units in either chain that
have at least one interchain PAE below the cutoff.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

#: Molecule-kind complexity order used for mixed-type pair rules.
KIND_RANK = {"protein": 0, "dna": 1, "rna": 1, "ligand": 2, "ion": 3}
POLYMER_KINDS = ("protein", "dna", "rna")
NUCLEIC_KINDS = ("dna", "rna")
KNOWN_KINDS = tuple(KIND_RANK)

PROTEIN_MIN_D0 = 1.0
NUCLEIC_MIN_D0 = 2.0
FIXED_D0 = 2.0


# ---------------------------------------------------------------------------
# scalar definitions
# ---------------------------------------------------------------------------
def ptm_term(pae_values, d0):
    """TM-style interface kernel ``1 / (1 + (pae / d0) ** 2)``.

    ``d0`` may be a scalar or an array broadcastable against ``pae_values``.
    """
    d0 = np.asarray(d0, dtype=float)
    return 1.0 / (1.0 + (np.asarray(pae_values, dtype=float) / d0) ** 2)


def d0_from_length(n, minimum=PROTEIN_MIN_D0):
    """``max(minimum, 1.24 * (n - 15) ** (1/3) - 1.8)``; 1.0 A for n <= 15."""
    length = float(n)
    if length > 15.0:
        value = 1.24 * np.cbrt(length - 15.0) - 1.8
    else:
        value = PROTEIN_MIN_D0
    return float(max(float(minimum), float(value)))


def _d0_array(counts, minimum=PROTEIN_MIN_D0):
    counts = np.asarray(counts, dtype=float)
    return np.maximum(float(minimum), 1.24 * np.cbrt(counts - 15.0) - 1.8)


# ---------------------------------------------------------------------------
# units: the residue / ligand-atom level the scores are defined on
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Unit:
    kind: str
    chain: str
    token_idxs: Tuple[int, ...]
    rep: int
    frame_valid: bool
    label: str


_REP_ATOMS = {"protein": ("CA",), "dna": ("C3'", "C1'"), "rna": ("C3'", "C1'")}


def _token_kind(token, index):
    kind = token.get("kind")
    if kind is None:
        raise ValueError(
            f"token {index} has no 'kind'; expected one of {KNOWN_KINDS}"
        )
    if kind not in KNOWN_KINDS:
        raise ValueError(
            f"token {index} has kind {kind!r}; expected one of {KNOWN_KINDS}"
        )
    return kind


def _token_chain(token, index):
    chain = token.get("chain")
    if not isinstance(chain, str) or not chain:
        raise ValueError(
            f"token {index} lacks a usable 'chain' (got {chain!r}); every token "
            "must name the chain it belongs to so its molecule can be placed"
        )
    return chain


def _build_units(tokens):
    """Group tokens into scored units and validate that they can be placed."""
    if not isinstance(tokens, (list, tuple)):
        raise TypeError("tokens must be a list of dicts")
    groups: "OrderedDict[tuple, List[int]]" = OrderedDict()
    kinds: Dict[tuple, str] = {}
    for index, token in enumerate(tokens):
        if not isinstance(token, dict):
            raise TypeError(f"token {index} must be a dict, got {type(token).__name__}")
        kind = _token_kind(token, index)
        chain = _token_chain(token, index)
        if kind in POLYMER_KINDS:
            residue_index = token.get("residue_index")
            if residue_index is None:
                raise ValueError(
                    f"token {index} (kind={kind!r}, chain={chain!r}) lacks "
                    "'residue_index'; polymer tokens must carry a residue index so "
                    "tokens can be grouped into residues instead of being treated "
                    "as independent units"
                )
            if isinstance(residue_index, bool) or not isinstance(residue_index, int):
                raise ValueError(
                    f"token {index} (chain={chain!r}) has residue_index "
                    f"{residue_index!r}; expected a plain int"
                )
            key = ("polymer", chain, residue_index)
        else:
            key = ("nonpolymer", chain, index)
        if key in kinds and kinds[key] != kind:
            raise ValueError(
                f"unit {key} mixes kinds {kinds[key]!r} and {kind!r}"
            )
        kinds[key] = kind
        groups.setdefault(key, []).append(index)

    units: List[Unit] = []
    for key, idxs in groups.items():
        kind = kinds[key]
        chain = key[1]
        wanted = _REP_ATOMS.get(kind, ())
        rep = next(
            (i for i in idxs if tokens[i].get("atom_name") in wanted), idxs[0]
        )
        flags = [tokens[i].get("frame_valid") for i in idxs]
        if any(flag is not None for flag in flags):
            if any(flag is None for flag in flags):
                raise ValueError(
                    f"unit {key} has 'frame_valid' on only some of its tokens; "
                    "either set it on every token of the unit or on none"
                )
            frame_valid = all(bool(flag) for flag in flags)
        else:
            # Without coordinates or an explicit mask, every polymer residue and
            # every multi-atom ligand is assumed able to define a frame.  Ions
            # never can (a single point has no orientation).
            frame_valid = kind != "ion"
        if kind in POLYMER_KINDS:
            label = f"{chain}:{key[2]}"
        else:
            atom = tokens[rep].get("atom_name")
            label = f"{chain}:{atom}" if atom else f"{chain}:#{rep}"
        units.append(
            Unit(kind=kind, chain=chain, token_idxs=tuple(idxs), rep=rep,
                 frame_valid=frame_valid, label=label)
        )
    return units


def _chain_units(units):
    chains: "OrderedDict[str, List[Unit]]" = OrderedDict()
    for unit in units:
        chains.setdefault(unit.chain, []).append(unit)
    return chains


def _chain_types(chains):
    types = {}
    for chain, units in chains.items():
        kinds = {unit.kind for unit in units}
        polymers = kinds & set(POLYMER_KINDS)
        if len(polymers) > 1:
            raise ValueError(
                f"chain {chain!r} mixes polymer kinds {sorted(polymers)}; a chain "
                "must be a single protein, DNA or RNA molecule"
            )
        if len(kinds) > 1 and polymers:
            raise ValueError(
                f"chain {chain!r} mixes polymer and non-polymer units "
                f"({sorted(kinds)}); give each ligand or ion its own chain"
            )
        types[chain] = units[0].kind
    return types


def _protein_units(units):
    return np.array([unit.kind == "protein" for unit in units], dtype=bool)


# ---------------------------------------------------------------------------
# one direction
# ---------------------------------------------------------------------------
def _score_direction(pae, frame_units, scored_units, cutoff, d0_mode="units",
                     pair_d0=None):
    """Max over alignment frames of the mean TM term over scored units.

    ``d0_mode='units'`` is the ipSAE2 rule: d0 from the number of qualifying
    protein residues in the scored side (fixed 2.0 A when that side has no
    protein).  ``d0_mode='pair'`` uses the scalar ``pair_d0`` for every frame
    (ipSAE ``d0chn`` / ``d0dom``).  Returns ``None`` when no frame is usable.
    """
    rows = [unit.rep for unit in frame_units if unit.frame_valid]
    if not rows:
        return None
    rows = np.asarray(rows, dtype=int)
    lengths = np.asarray([len(unit.token_idxs) for unit in scored_units], dtype=int)
    if lengths.size == 0:
        return None
    cols = np.concatenate([np.asarray(unit.token_idxs, dtype=int) for unit in scored_units])
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(int)

    # Shape (frames, scored tokens).  Index only: the matrix is never
    # symmetrised, so row/column roles stay exactly as the predictor wrote them.
    sub = np.asarray(pae[np.ix_(rows, cols)], dtype=float)

    protein_mask = _protein_units(scored_units)
    uses_cutoff = bool(np.isin([u.kind for u in scored_units], POLYMER_KINDS).any())
    if uses_cutoff and cutoff is not None:
        masked = np.isfinite(sub) & (sub <= float(cutoff))
    elif uses_cutoff:
        masked = np.isfinite(sub)
    else:
        if not np.isfinite(sub).all():
            raise ValueError(
                "non-finite PAE between a terminal frame and a ligand/ion unit: "
                "ligand and ion units are scored without a PAE cutoff, so every "
                "interchain value must be finite"
            )
        masked = np.ones(sub.shape, dtype=bool)

    if d0_mode == "pair":
        d0s = np.full(len(rows), float(pair_d0), dtype=float)
    else:
        unit_hits = _per_unit(masked.astype(float), starts) > 0
        if protein_mask.any():
            protein_counts = unit_hits[:, protein_mask].sum(axis=1)
            d0s = _d0_array(protein_counts)
        else:
            d0s = np.full(len(rows), FIXED_D0, dtype=float)

    tm = ptm_term(sub, d0s[:, None])
    masked_tm = np.where(masked, tm, 0.0)
    sums = _per_unit(masked_tm, starts)
    counts = _per_unit(masked.astype(float), starts)
    selected = counts > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_scores = np.divide(sums, counts, out=np.zeros_like(sums), where=selected)
    n0 = selected.sum(axis=1)
    total = unit_scores.sum(axis=1)
    means = np.divide(total, n0, out=np.zeros(len(rows), dtype=float),
                      where=n0 > 0)
    best = int(np.argmax(means))

    n0_best = int(n0[best])
    ipae = None
    if n0_best:
        masked_pae = np.where(masked[best], sub[best], 0.0)[None, :]
        pae_sums = _per_unit(masked_pae, starts)
        pae_means = np.divide(pae_sums, counts[best], out=np.zeros_like(pae_sums),
                              where=selected[best])
        ipae = float(pae_means[0][selected[best]].mean())
    return {
        "score": float(means[best]),
        "ipae": ipae,
        "n0": n0_best,
        "d0": float(d0s[best]),
        "best_index": best,
    }


def _per_unit(values, starts):
    """Sum columns into the contiguous per-unit groups described by ``starts``."""
    return np.add.reduceat(values, starts, axis=1)


# ---------------------------------------------------------------------------
# pair-level helpers
# ---------------------------------------------------------------------------
def _n0dom(pae, units_a, units_b, cutoff):
    """Units in either chain with any interchain PAE at/below the cutoff.

    Evaluated for one direction (rows = ``units_a``, columns = ``units_b``),
    exactly as the reference does: the row condition and the column condition
    are read from the same block, so the value is direction-dependent because
    the PAE matrix is not symmetric.
    """
    rows = np.asarray([u.rep for u in units_a], dtype=int)
    cols = np.asarray([u.rep for u in units_b], dtype=int)
    if rows.size == 0 or cols.size == 0:
        return 0
    block = np.asarray(pae[np.ix_(rows, cols)], dtype=float)
    ok = np.isfinite(block) & (block <= float(cutoff))
    if not ok.any():
        return 0
    return int(ok.any(axis=1).sum() + ok.any(axis=0).sum())


def _pair_d0(n_units, nucleic):
    return d0_from_length(n_units, NUCLEIC_MIN_D0 if nucleic else PROTEIN_MIN_D0)


def _interface_stats(pae, units_a, units_b, cutoffs):
    """PAE statistics over all interchain unit pairs, plus per-cutoff counts."""
    rows = np.asarray([u.rep for u in units_a], dtype=int)
    cols = np.asarray([u.rep for u in units_b], dtype=int)
    block = np.asarray(pae[np.ix_(rows, cols)], dtype=float).ravel()
    finite = block[np.isfinite(block)]
    stats = {
        "mean": float(finite.mean()) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "n_pairs": int(finite.size),
        "n_below": {},
        "fraction_below": {},
    }
    for cutoff in cutoffs:
        count = int(np.count_nonzero(finite <= float(cutoff)))
        stats["n_below"][float(cutoff)] = count
        stats["fraction_below"][float(cutoff)] = (
            float(count) / finite.size if finite.size else None
        )
    return stats


def _pair_rule(type_a, type_b, a, b):
    """Choose the reported pair value from the two directional results."""
    scores = {name: res["score"] for name, res in (("ipsae12", a), ("ipsae21", b))
              if res is not None}
    if not scores:
        return None, None
    if KIND_RANK[type_a] == KIND_RANK[type_b]:
        source = max(scores, key=lambda k: (scores[k], k == "ipsae12"))
        return scores[source], source
    # Different molecule kinds: align on the more complex molecule.  Chains are
    # ordered by KIND_RANK before pairing, so that is normally ipsae12.
    source = "ipsae12" if KIND_RANK[type_a] < KIND_RANK[type_b] else "ipsae21"
    if source not in scores:
        return None, None
    return scores[source], source


def _max_direction(a, b):
    """Larger of the two directional scores regardless of molecule type."""
    scores = [res["score"] for res in (a, b) if res is not None]
    return max(scores) if scores else None


def _field(result, key):
    """Read one field from a possibly-undefined direction result."""
    return None if result is None else result[key]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def compute_interface_scores(pae, tokens, cutoffs=(10.0, 15.0)):
    """ipSAE / ipSAE2 interface scores for every chain pair of one prediction.

    Parameters
    ----------
    pae : array-like, shape (N, N)
        Predicted aligned error in Angstrom, token level.  ``pae[i, j]`` is the
        expected error of token ``j`` when the alignment is on token ``i``;
        rows are alignment frames.
    tokens : list of dict
        ``N`` dicts describing the tokens in PAE order.  Required keys:
        ``kind`` (``'protein' | 'dna' | 'rna' | 'ligand' | 'ion'``) and
        ``chain``.  Polymer tokens also require ``residue_index`` (int) so that
        tokens can be grouped into residues.  ``atom_name`` is optional and is
        used only to pick the alignment representative (CA / C3' / C1').
        An optional ``frame_valid`` bool (on every token of a unit) honours a
        predictor's own frame mask; without it every non-ion unit is treated as
        a usable frame.
    cutoffs : sequence of float
        PAE cutoffs in Angstrom applied to polymer scored units.

    Returns
    -------
    dict
        Keyed by the ordered ``(chain_a, chain_b)`` tuple, ordered by molecule
        complexity (protein, nucleic acid, ligand, ion) then by first
        appearance.  Each value is ``{"chains", "types", "n_units",
        "interface_pae", "by_cutoff", "skipped"}``.

        ``interface_pae`` holds ``mean``, ``median``, ``n_pairs`` over every
        interchain unit pair, plus ``n_below`` / ``fraction_below`` keyed by
        cutoff.

        ``by_cutoff`` is keyed by the cutoff and holds:

        * ``ipsae12``, ``ipsae21`` - the two directional ipSAE/ipSAE2 scores
          (``12`` = align on ``chain_a``, score ``chain_b``);
        * ``ipsae``, ``ipsae_source`` - the pair value from the type rule;
        * ``ipsae_d0chn``, ``ipsae_d0dom``, ``iptm_d0chn`` and
          ``ipsae_d0chn12/21``, ``ipsae_d0dom12/21``, ``iptm_d0chn12/21`` -
          the ipSAE ``d0chn``/``d0dom`` family, under the same pair rule plus
          the plain directional values;
        * ``ipsae_d0chn_max``, ``ipsae_d0dom_max``, ``iptm_d0chn_max`` - the
          larger of the two directions regardless of type, which is what the
          ipSAE ``max`` row reports for mixed-type pairs too;
        * ``n0_12``, ``n0_21``, ``d0_12``, ``d0_21``, ``ipae12``, ``ipae21`` -
          scored-unit count, d0 and mean PAE at the maximizing frame of each
          direction (``None`` when that direction has no usable frame);
        * ``n0chn``, ``d0chn``, ``n0dom``, ``d0dom`` (and ``n0dom12/21``,
          ``d0dom12/21``) - the ipSAE length definitions, reported for the
          direction that produced the pair value;
        * ``pae_cutoff12``, ``pae_cutoff21`` - the cutoff actually applied to
          the scored side, or ``None`` for ligands/ions;
        * ``interface_pae`` - ``mean``, ``median`` and ``count`` of the
          interchain PAE values at or below this cutoff.

        ``skipped`` is ``None`` for a scored pair, otherwise a reason string
        (ion-ion pair, no usable frame in either chain, non-finite PAE between a
        frame and an uncut ligand/ion unit).  A skipped pair still appears in
        the result, with ``{"skipped": reason}`` in each ``by_cutoff`` entry.
    """
    pae = np.asarray(pae, dtype=float)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"pae must be a square 2-D array, got shape {pae.shape}")
    if len(tokens) != pae.shape[0]:
        raise ValueError(
            f"pae is {pae.shape[0]}x{pae.shape[1]} but {len(tokens)} tokens were "
            "given; PAE rows must be in token order"
        )
    cutoffs = tuple(float(c) for c in cutoffs)
    if not cutoffs:
        raise ValueError("at least one PAE cutoff is required")

    units = _build_units(tokens)
    chains = _chain_units(units)
    types = _chain_types(chains)
    order = sorted(
        chains,
        key=lambda c: (KIND_RANK[types[c]], list(chains).index(c)),
    )

    results: Dict[Tuple[str, str], dict] = {}
    for pos, chain_a in enumerate(order):
        for chain_b in order[pos + 1:]:
            type_a, type_b = types[chain_a], types[chain_b]
            units_a, units_b = chains[chain_a], chains[chain_b]
            record = {
                "chains": (chain_a, chain_b),
                "types": (type_a, type_b),
                "n_units": (len(units_a), len(units_b)),
                "interface_pae": _interface_stats(pae, units_a, units_b, cutoffs),
                "by_cutoff": {},
                "skipped": None,
            }
            if type_a == "ion" and type_b == "ion":
                record["skipped"] = "ion-ion pairs are not scored (no frame)"
                record["by_cutoff"] = {c: {"skipped": record["skipped"]} for c in cutoffs}
                results[(chain_a, chain_b)] = record
                continue
            if not any(u.frame_valid for u in units_a) and not any(
                    u.frame_valid for u in units_b):
                record["skipped"] = (
                    "no usable alignment frame in either chain; no PAE-based "
                    "score can be defined for this pair"
                )
                record["by_cutoff"] = {c: {"skipped": record["skipped"]} for c in cutoffs}
                results[(chain_a, chain_b)] = record
                continue

            nucleic = type_a in NUCLEIC_KINDS or type_b in NUCLEIC_KINDS
            n0chn = len(units_a) + len(units_b)
            d0chn = _pair_d0(n0chn, nucleic)

            try:
                for cutoff in cutoffs:
                    n0dom_12 = _n0dom(pae, units_a, units_b, cutoff)
                    n0dom_21 = _n0dom(pae, units_b, units_a, cutoff)
                    d0dom_12 = _pair_d0(n0dom_12, nucleic)
                    d0dom_21 = _pair_d0(n0dom_21, nucleic)

                    a = _score_direction(pae, units_a, units_b, cutoff)
                    b = _score_direction(pae, units_b, units_a, cutoff)
                    a_chn = _score_direction(pae, units_a, units_b, cutoff,
                                             d0_mode="pair", pair_d0=d0chn)
                    b_chn = _score_direction(pae, units_b, units_a, cutoff,
                                             d0_mode="pair", pair_d0=d0chn)
                    a_dom = _score_direction(pae, units_a, units_b, cutoff,
                                             d0_mode="pair", pair_d0=d0dom_12)
                    b_dom = _score_direction(pae, units_b, units_a, cutoff,
                                             d0_mode="pair", pair_d0=d0dom_21)
                    a_iptm = _score_direction(pae, units_a, units_b, None,
                                              d0_mode="pair", pair_d0=d0chn)
                    b_iptm = _score_direction(pae, units_b, units_a, None,
                                              d0_mode="pair", pair_d0=d0chn)

                    value, source = _pair_rule(type_a, type_b, a, b)
                    chn_max = _max_direction(a_chn, b_chn)
                    dom_max = _max_direction(a_dom, b_dom)
                    iptm_max = _max_direction(a_iptm, b_iptm)
                    chn_value, chn_source = _pair_rule(type_a, type_b, a_chn, b_chn)
                    dom_value, dom_source = _pair_rule(type_a, type_b, a_dom, b_dom)
                    iptm_value, iptm_source = _pair_rule(type_a, type_b, a_iptm, b_iptm)
                    # Report the d0dom that produced the winning direction.
                    if dom_source == "ipsae21":
                        n0dom, d0dom = n0dom_21, d0dom_21
                    else:
                        n0dom, d0dom = n0dom_12, d0dom_12

                    rows_a = np.asarray([u.rep for u in units_a], dtype=int)
                    rows_b = np.asarray([u.rep for u in units_b], dtype=int)
                    block = np.asarray(pae[np.ix_(rows_a, rows_b)], dtype=float).ravel()
                    block = block[np.isfinite(block) & (block <= cutoff)]

                    record["by_cutoff"][cutoff] = {
                        "ipsae12": _field(a, "score"),
                        "ipsae21": _field(b, "score"),
                        "ipsae": value,
                        "ipsae_source": source,
                        "ipsae_d0chn": chn_value,
                        "ipsae_d0chn_source": chn_source,
                        "ipsae_d0chn_max": chn_max,
                        "ipsae_d0chn12": _field(a_chn, "score"),
                        "ipsae_d0chn21": _field(b_chn, "score"),
                        "ipsae_d0dom": dom_value,
                        "ipsae_d0dom_source": dom_source,
                        "ipsae_d0dom_max": dom_max,
                        "ipsae_d0dom12": _field(a_dom, "score"),
                        "ipsae_d0dom21": _field(b_dom, "score"),
                        "iptm_d0chn": iptm_value,
                        "iptm_d0chn_max": iptm_max,
                        "iptm_d0chn12": _field(a_iptm, "score"),
                        "iptm_d0chn21": _field(b_iptm, "score"),
                        "n0_12": _field(a, "n0"),
                        "n0_21": _field(b, "n0"),
                        "d0_12": _field(a, "d0"),
                        "d0_21": _field(b, "d0"),
                        "ipae12": _field(a, "ipae"),
                        "ipae21": _field(b, "ipae"),
                        "n0chn": n0chn,
                        "d0chn": d0chn,
                        "n0dom": n0dom,
                        "d0dom": d0dom,
                        "n0dom12": n0dom_12,
                        "n0dom21": n0dom_21,
                        "d0dom12": d0dom_12,
                        "d0dom21": d0dom_21,
                        "pae_cutoff12": cutoff if type_b in POLYMER_KINDS else None,
                        "pae_cutoff21": cutoff if type_a in POLYMER_KINDS else None,
                        "interface_pae": {
                            "mean": float(block.mean()) if block.size else None,
                            "median": float(np.median(block)) if block.size else None,
                            "count": int(block.size),
                        },
                    }
            except ValueError as exc:
                record["skipped"] = str(exc)
                record["by_cutoff"] = {c: {"skipped": str(exc)} for c in cutoffs}
            results[(chain_a, chain_b)] = record
    return results

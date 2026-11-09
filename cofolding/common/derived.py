"""Assemble the per-prediction derived interface report (numpy only).

The containers call :func:`write_derived` after a prediction and before the
large PAE tensors may be dropped.  This module owns the report schema and the
pair-entry shape; each container owns the engine-specific token map, because
only the engine can say what its own PAE rows mean.

Nothing here imports torch, gemmi, rdkit or pandas.
"""
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from . import ipsae

SCHEMA = 1
CUTOFFS = (10.0, 15.0)

#: Formula identities written into every report, with the version this module
#: implements.  ``ipsae2`` is the directional ipSAE2 pair value; ``iptm_d0chn``
#: is the ipSAE ``d0chn`` no-cutoff variant; ``interface_pae_stats`` is the raw
#: interchain PAE summary.
FORMULAS = (
    {"id": "ipsae2", "version": "1",
     "definition": "max over alignment frames of the mean TM(pae, d0) over the "
                   "scored side, d0 from the qualifying protein units; pair rule "
                   "by molecule complexity"},
    {"id": "iptm_d0chn", "version": "1",
     "definition": "same kernel with d0 from the summed length of both chains and "
                   "no PAE cutoff"},
    {"id": "interface_pae_stats", "version": "1",
     "definition": "mean/median/count of interchain PAE, plus counts at each cutoff"},
)


class TokenMapError(ValueError):
    """A token cannot be placed; the caller records the reason, never a guess."""


def _plain(value):
    """Convert numpy scalars/arrays into JSON-native values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def atomic_json(path, obj):
    """Write JSON atomically; the container-local writer is equivalent."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(obj, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_token_map(path):
    """Read and structurally validate a token map; raise TokenMapError."""
    try:
        data = json.loads(Path(path).read_text())
    except OSError as exc:
        raise TokenMapError('token map is unreadable: %s (%s)' % (path, exc))
    except ValueError as exc:
        raise TokenMapError('token map is not valid JSON: %s (%s)' % (path, exc))
    if not isinstance(data, dict):
        raise TokenMapError('token map must be a JSON object: %s' % (path,))
    tokens = data.get('tokens')
    if not isinstance(tokens, list) or not tokens:
        raise TokenMapError('token map has no non-empty "tokens" list: %s' % (path,))
    for index, token in enumerate(tokens):
        if not isinstance(token, dict):
            raise TokenMapError('token %d is not an object: %s' % (index, path))
        if token.get('index') != index:
            raise TokenMapError(
                'token %d records index %r; entries must be in PAE order: %s'
                % (index, token.get('index'), path))
        if token.get('kind') not in ipsae.KNOWN_KINDS:
            raise TokenMapError(
                'token %d has kind %r, expected one of %s: %s'
                % (index, token.get('kind'), list(ipsae.KNOWN_KINDS), path))
        if not isinstance(token.get('chain'), str) or not token['chain']:
            raise TokenMapError('token %d has no usable chain: %s' % (index, path))
        if token['kind'] in ipsae.POLYMER_KINDS:
            if not isinstance(token.get('residue_index'), int) or isinstance(
                    token.get('residue_index'), bool):
                raise TokenMapError(
                    'token %d (%s) has no integer residue_index: %s'
                    % (index, token['kind'], path))
        else:
            if 'atom_index' not in token:
                raise TokenMapError(
                    'token %d (%s) has no atom_index: %s'
                    % (index, token['kind'], path))
    n_tokens = data.get('n_tokens')
    if isinstance(n_tokens, int) and n_tokens != len(tokens):
        raise TokenMapError(
            'token map declares %d tokens but lists %d: %s'
            % (n_tokens, len(tokens), path))
    return tokens


def build_pair_entries(scores, cutoffs):
    """Turn one ``compute_interface_scores`` result into report entries."""
    pairs = []
    for (chain_a, chain_b), record in scores.items():
        entry = {
            "chains": [chain_a, chain_b],
            "types": list(record["types"]),
            "n_units": list(record["n_units"]),
            "skipped": record.get("skipped"),
            "interface_pae": _plain(record.get("interface_pae")),
            "by_cutoff": [],
        }
        for cutoff, values in record.get("by_cutoff", {}).items():
            if "skipped" in values:
                entry["by_cutoff"].append({"cutoff": float(cutoff),
                                           "skipped": values["skipped"]})
                continue
            entry["by_cutoff"].append(_plain({
                "cutoff": float(cutoff),
                "ipsae": values.get("ipsae"),
                "ipsae_source": values.get("ipsae_source"),
                "ipsae12": values.get("ipsae12"),
                "ipsae21": values.get("ipsae21"),
                "ipae12": values.get("ipae12"),
                "ipae21": values.get("ipae21"),
                "pae_cutoff12": values.get("pae_cutoff12"),
                "pae_cutoff21": values.get("pae_cutoff21"),
                "ipsae_d0chn": values.get("ipsae_d0chn"),
                "ipsae_d0dom": values.get("ipsae_d0dom"),
                "iptm_d0chn": values.get("iptm_d0chn"),
                "n0_12": values.get("n0_12"),
                "n0_21": values.get("n0_21"),
                "d0_12": values.get("d0_12"),
                "d0_21": values.get("d0_21"),
                "n0chn": values.get("n0chn"),
                "d0chn": values.get("d0chn"),
                "n0dom": values.get("n0dom"),
                "d0dom": values.get("d0dom"),
                "interface_pae": values.get("interface_pae"),
            }))
        pairs.append(entry)
    return pairs


def score(pae, tokens, cutoffs=CUTOFFS):
    """Score one prediction; return ``(pair_entries, null_reason)``."""
    if pae is None:
        return None, "PAE array is unavailable"
    if tokens is None:
        return None, "token map is unavailable"
    try:
        scores = ipsae.compute_interface_scores(pae, tokens, cutoffs)
    except (ValueError, TypeError) as exc:
        return None, "interface scores unavailable: %s" % (exc,)
    return build_pair_entries(scores, cutoffs), None


def write_derived(attempt, predictions, cutoffs=CUTOFFS):
    """Write ``<attempt>/metrics/derived.json`` and return its receipt record.

    ``predictions`` is a list of dicts already carrying ``sample_index``,
    ``pae`` (attempt-relative path or null), ``token_map`` (attempt-relative
    path or null), ``pairs`` (or null) and ``derived_null_reason``.  A missing
    or unusable PAE/token map is recorded verbatim, never replaced by a number.
    """
    attempt = Path(attempt)
    metrics = attempt / 'metrics'
    metrics.mkdir(exist_ok=True)
    reasons = []
    entries = []
    for item in predictions:
        entry = {
            "sample_index": item.get("sample_index"),
            "sample_seed": item.get("sample_seed"),
            "pae": item.get("pae"),
            "token_map": item.get("token_map"),
            "pairs": item.get("pairs"),
            "derived_null_reason": item.get("derived_null_reason"),
        }
        if entry["derived_null_reason"]:
            reasons.append(entry["derived_null_reason"])
        entries.append(entry)
    if not entries:
        reasons.append("no prediction was available to derive interface scores from")
    payload = {
        "schema": SCHEMA,
        "formulas": list(FORMULAS),
        "cutoffs": [float(c) for c in cutoffs],
        "predictions": entries,
        "derived_null_reason": "; ".join(dict.fromkeys(reasons)) if reasons else None,
    }
    atomic_json(metrics / 'derived.json', payload)
    return {"path": "metrics/derived.json",
            "null_reason": payload["derived_null_reason"]}

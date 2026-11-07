"""Current-job saved-byte checks, without legacy CLI/event assumptions."""
from pathlib import Path
import sys

# Extracted helpers use sibling imports; ship the complete helper directory.
_HELPERS = Path(__file__).resolve().parents[1] / 'batch_native'
sys.path.insert(0, str(_HELPERS))
from verify import SCALARS, packet, eq, check_input, check_cif
from graph import require

import json
import numpy as np
import torch
from boltz.data.types import StructureV2


def verify_job(prepared, attempt, settings):
    rid = prepared['rid']
    b, f, p = [packet(attempt / 'capture' / (label + '.pt'))
               for label in ('consumed_batch', 'full_forward', 'full_prediction')]
    require(p['exception'] is False and p['coords'].shape[0] == settings.samples, 'samples/native exception')
    require({'pdistogram', 's', 'z', 'pde_logits', 'pae_logits', 'plddt_logits', 'resolved_logits'} <= set(f), 'full forward fields')
    require({'pae', 'pde', 'plddt', 'ptm', 'iptm', 'pair_chains_iptm', 's', 'z'} <= set(p), 'full prediction fields')
    eq(p['coords'], f['sample_atom_coords'], 'forward coords')
    eq(p['masks'], b['atom_pad_mask'], 'atom mask')
    require([r.id for r in b['record']] == [rid], 'consumed record identity')
    data = prepared['data']
    # Reuse full source chemistry/MSA checks for the established narrow route.
    # Other native structure YAMLs still receive tensor, atom and writer checks.
    simple = not data.get('templates') and not data.get('constraints')
    proteins = 0
    for entity in data.get('sequences', []):
        kind = next(iter(entity))
        entry = entity[kind]
        proteins += kind == 'protein'
        simple &= (kind in ('protein', 'ligand') and isinstance(entry.get('id'), str)
                   and not entry.get('modifications') and not entry.get('cyclic')
                   and (kind != 'ligand' or 'smiles' in entry))
    simple &= proteins == 1
    if simple:
        structure, elements = check_input(data, prepared['processed'], rid, b)
    else:
        structure = StructureV2.load(prepared['processed'] / 'structures' / (rid + '.npz'))
        structure = structure.remove_invalid_chains()
        elements = {}
        require(int(b['atom_pad_mask'].sum()) == len(structure.atoms), 'full atom coverage')
    rank = torch.argsort(p['confidence_score'], descending=True).tolist()
    require(sorted(rank) == list(range(settings.samples)), 'rank bijection')
    pred = prepared['native'] / 'predictions' / rid
    extension = 'pdb' if settings.output_format == 'pdb' else 'cif'
    expected = {'embeddings_' + rid + '.npz'}
    for r, i in enumerate(rank):
        base = f'{rid}_model_{r}'
        expected |= {base + '.' + extension, 'confidence_' + base + '.json',
                     *{key + '_' + base + '.npz' for key in ('pae', 'pde', 'plddt')}}
        check_cif(pred / (base + '.' + extension), structure,
                  p['coords'][i][p['masks'][0].bool()].numpy(), elements)
        confidence = json.loads((pred / ('confidence_' + base + '.json')).read_text())
        for field in SCALARS:
            eq(confidence[field], p[field][i].numpy(), 'confidence ' + field)
        for a, values in p['pair_chains_iptm'].items():
            for c, value in values.items():
                eq(confidence['pair_chains_iptm'][str(a)][str(c)], value[i].numpy(), 'chain confidence')
            eq(confidence['chains_ptm'][str(a)], values[a][i].numpy(), 'chain ptm')
        for field in ('pae', 'pde', 'plddt'):
            with np.load(pred / (field + '_' + base + '.npz'), allow_pickle=False) as arrays:
                eq(arrays[field], p[field][i].numpy(), field + ' pairing')
    require({p.name for p in pred.iterdir()} == expected, 'exact native output inventory')
    with np.load(pred / ('embeddings_' + rid + '.npz'), allow_pickle=False) as arrays:
        require(set(arrays.files) == {'s', 'z'}, 'embeddings fields')
        for key in arrays.files:
            eq(arrays[key], p[key].numpy(), 'embeddings preservation')

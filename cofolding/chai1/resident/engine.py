"""Pinned Chai-1 0.6.1 native preparation/model/writer boundary (GPU runtime).

Accepts the shared native YAML entity format used by the other cofold containers and maps it onto
Chai-1's FASTA input. Protein chains and SMILES ligands are supported; RNA/DNA chains, per-chain
MSA, templates, constraints, covalent bonds and glycan strings are outside the qualified scope.
Every rejection names the rule that rejected the system, so an unsupported system never becomes
an unexplained failure.

Chai-1 is invoked through two library entry points that are part of its public API:

* ``chai_lab.chai1.make_all_atom_feature_context`` builds the tokenized feature context; the
  context's token arrays are the authoritative token-level identity used for the derived-metric
  token map (never the PAE shape).
* ``chai_lab.chai1.run_folding_on_context`` folds one diffusion sample from that context with an
  explicit seed, so one native call produces exactly one candidate.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import string
import time

from .errors import native_boundary
import inspect

import numpy as np
import torch
import yaml
from cofolding.common import derived

from .identity import directory_identity
from .worker import atomic_json

CANONICAL = set('ACDEFGHIKLMNPQRSTVWY')

#: keyword arguments the worker always passes explicitly to the native fold call
FOLD_OVERRIDES = ('output_dir', 'num_trunk_recycles', 'num_diffn_timesteps', 'num_diffn_samples',
                  'seed', 'device', 'low_memory')
VERSION_PACKAGES = ('chai_lab', 'torch', 'numpy', 'rdkit', 'gemmi', 'biopython')

#: Files chai_lab looks for under CHAI_DOWNLOADS_DIR at inference time.
CHAI_COMPONENTS = ('feature_embedding.pt', 'bond_loss_input_proj.pt', 'token_embedder.pt',
                   'trunk.pt', 'diffusion_module.pt', 'confidence_head.pt')
CHAI_CONFORMERS = 'conformers_v1.apkl'
CHAI_ESM = 'esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt'


def fold_record(fn):
    """Effective per-call arguments: what we override, and the native defaults we inherit."""
    parameters = inspect.signature(fn).parameters
    defaults = {}
    for name, parameter in parameters.items():
        if name in ('self', 'model', 'input', 'feature_context'):
            continue
        defaults[name] = None if parameter.default is inspect.Parameter.empty else parameter.default
    return {'explicit': list(FOLD_OVERRIDES),
            'native_defaults': {k: v for k, v in defaults.items() if k not in FOLD_OVERRIDES}}


def chain_letter(asym_id):
    """Chai-1's one-indexed asym id -> written CIF chain id (``A``..``Z``, ``a``..``z``, then pairs)."""
    vocab = [*string.ascii_uppercase, *string.ascii_lowercase]
    vocab = vocab + [x + y for x in vocab for y in vocab]
    if not isinstance(asym_id, int) or isinstance(asym_id, bool) or not 1 <= asym_id <= len(vocab):
        raise ValueError('asym id must be a one-indexed integer within the CIF chain vocabulary')
    return vocab[asym_id - 1]


def canonical_smiles(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('ligand SMILES does not parse; the ligand is never dropped')
    return molecule, {
        'consumed_smiles': smiles, 'canonical_smiles': Chem.MolToSmiles(molecule),
        'heavy_atoms': molecule.GetNumHeavyAtoms(), 'formal_charge': Chem.GetFormalCharge(molecule),
        'formula': rdMolDescriptors.CalcMolFormula(molecule),
        'rings': len(Chem.GetSSSR(molecule)),
        'iron_atoms': sum(1 for atom in molecule.GetAtoms() if atom.GetSymbol() == 'Fe'),
        'state_policy': 'Exact explicit input state; no neutralization/protonation/tautomer edits'}


def capability(data, label):
    """Map the shared YAML entity format onto Chai-1; reject what it cannot express."""
    if not isinstance(data, dict):
        raise ValueError(label + ': native YAML must be a mapping')
    if 'seeds' in data or 'seed' in data:
        raise ValueError(label + ': seeds belong in the external --seeds list')
    if data.get('properties'):
        raise ValueError(label + ': affinity/property prediction is not supported by Chai-1')
    for field in ('templates', 'constraints'):
        if data.get(field):
            raise ValueError(label + ': Chai-1 runs without ' + field)
    proteins, ligands = [], []
    for entity in data.get('sequences') or []:
        if not isinstance(entity, dict) or len(entity) != 1:
            raise ValueError(label + ': each entity must have exactly one type key')
        kind, entry = next(iter(entity.items()))
        if not isinstance(entry, dict) or not isinstance(entry.get('id'), str):
            raise ValueError(label + ': entity needs a string id')
        if kind == 'protein':
            if entry.get('modifications') or entry.get('cyclic'):
                raise ValueError(label + ': modified or cyclic polymers are not supported')
            if entry.get('msa') not in (None, 'empty'):
                raise ValueError(label + ': Chai-1 runs without MSA; use msa: empty or omit it')
            sequence = entry.get('sequence')
            if not isinstance(sequence, str) or not sequence or set(sequence) - CANONICAL:
                raise ValueError(label + ': protein sequence must be canonical uppercase amino acids')
            proteins.append(entry)
        elif kind == 'ligand':
            if entry.get('ccd'):
                raise ValueError(label + ': CCD ligand references are not supported; pass SMILES')
            if not isinstance(entry.get('smiles'), str) or not entry['smiles']:
                raise ValueError(label + ': ligand needs a SMILES string')
            ligands.append(entry)
        else:
            raise ValueError(label + ': unsupported entity type ' + kind)
    if not proteins:
        raise ValueError(label + ': at least one protein chain is required')
    ids = [entry['id'] for entry in proteins]
    if len(set(ids)) != len(ids):
        raise ValueError(label + ': protein chain ids must be unique; got ' + ', '.join(ids))
    taken = set(ids) | {entry['id'] for entry in ligands}
    if len(taken) != len(ids) + len(ligands):
        raise ValueError(label + ': chain ids must be unique across proteins and ligands')
    for entry in ligands:
        canonical_smiles(entry['smiles'])
    return {'proteins': proteins, 'ligands': ligands}


def fasta_text(prepared):
    """The exact FASTA handed to chai_lab; order fixes the asym ids and written chain ids."""
    lines = []
    for entry in prepared['entities']:
        kind = 'protein' if entry['kind'] == 'protein' else 'ligand'
        lines.append('>' + kind + '|' + entry['id'])
        lines.append(entry['sequence'] if entry['kind'] == 'protein' else entry['smiles'])
    return '\n'.join(lines) + '\n'


# -- token map -------------------------------------------------------------------------
def _element_symbol(atomic_number):
    import gemmi
    try:
        return gemmi.Element(int(atomic_number)).name
    except Exception:
        return str(int(atomic_number))


def token_map(prepared, arrays, pae):
    """One entry per PAE token, derived from Chai-1's tokenized structure context.

    ``arrays`` holds the collated-feature token/atom arrays copied from
    ``feature_context.structure_context``: ``token_exists_mask``, ``token_asym_id``,
    ``token_residue_index``, ``token_entity_type``, ``atom_token_index``,
    ``atom_exists_mask``, ``atom_ref_name`` and ``atom_ref_element``.
    Chai-1 tokenizes proteins one token per residue and ligands one token per heavy atom, so the
    PAE rows are atom-expanded exactly as in ESMFold2.  Any mismatch is refused, never guessed.
    """
    if pae is None:
        raise derived.TokenMapError('result carries no PAE matrix')
    shape = np.asarray(pae).shape
    if len(shape) < 2 or shape[-1] != shape[-2]:
        raise derived.TokenMapError('PAE matrix is not square: %s' % (shape,))
    n_pae = int(shape[-1])

    def column(name):
        value = arrays.get(name)
        if value is None:
            raise derived.TokenMapError('token arrays are missing ' + name)
        return np.asarray(value)

    exists = column('token_exists_mask').astype(bool).tolist()
    asym = column('token_asym_id').astype(int).tolist()
    residue = column('token_residue_index').astype(int).tolist()
    entity_type = column('token_entity_type').astype(int).tolist()
    atom_token = column('atom_token_index').astype(int).tolist()
    atom_exists = column('atom_exists_mask').astype(bool).tolist()
    atom_names = list(arrays.get('atom_ref_name') or [])
    atom_elements = column('atom_ref_element').astype(int).tolist()
    if not (len(exists) == len(asym) == len(residue) == len(entity_type)):
        raise derived.TokenMapError('token-level arrays disagree on token count')
    if len(atom_token) != len(atom_exists) or len(atom_names) != len(atom_token) \
            or len(atom_elements) != len(atom_token):
        raise derived.TokenMapError('atom-level arrays disagree on atom count')

    # Tokens present in the collated feature are the rows the confidence head scored.
    present = [index for index, flag in enumerate(exists) if flag]
    if len(present) != n_pae:
        raise derived.TokenMapError(
            'collated features place %d tokens but the PAE matrix has %d rows'
            % (len(present), n_pae))

    entities = prepared.get('entities') or []
    order = []
    for index in present:
        chain = asym[index]
        if not order or order[-1] != chain:
            if chain in order:
                raise derived.TokenMapError('asym id %r is not contiguous in the token order' % (chain,))
            order.append(chain)
    if len(order) != len(entities):
        raise derived.TokenMapError(
            'Chai-1 tokenized %d chains but %d entities were constructed'
            % (len(order), len(entities)))

    tokens = []
    chains = []
    for position, chain_number in enumerate(order):
        indexs = [index for index in present if asym[index] == chain_number]
        chain_id = chain_letter(chain_number)
        entity = entities[position]
        kind_code = entity_type[indexs[0]]
        if any(entity_type[index] != kind_code for index in indexs):
            raise derived.TokenMapError('chain %r mixes entity types' % (chain_id,))
        if kind_code == 0:
            if entity['kind'] != 'protein':
                raise derived.TokenMapError(
                    'chain %r is a protein in Chai-1 but %r in the input'
                    % (chain_id, entity['kind']))
            residues = [residue[index] for index in indexs]
            if sorted(residues) != list(range(len(entity['sequence']))):
                raise derived.TokenMapError(
                    'chain %r has %d tokens for a %d-residue input'
                    % (chain_id, len(indexs), len(entity['sequence'])))
            for index in indexs:
                tokens.append({'index': len(tokens), 'kind': 'protein', 'chain': chain_id,
                               'residue_index': int(residue[index])})
            chains.append({'id': chain_id, 'kind': 'protein', 'units': len(indexs),
                           'input_id': entity['id']})
            continue
        if kind_code != 3:
            raise derived.TokenMapError(
                'chain %r has Chai-1 entity type %d; only protein (0) and ligand (3) are supported'
                % (chain_id, kind_code))
        if entity['kind'] != 'ligand':
            raise derived.TokenMapError(
                'chain %r is a ligand in Chai-1 but %r in the input'
                % (chain_id, entity['kind']))
        expected = entity.get('heavy_atoms')
        if expected is not None and len(indexs) != int(expected):
            raise derived.TokenMapError(
                'ligand chain %r has %d tokens but the input SMILES has %d heavy atoms'
                % (chain_id, len(indexs), int(expected)))
        for atom_index, index in enumerate(indexs):
            atoms = [atom for atom, token in enumerate(atom_token)
                     if token == index and atom_exists[atom]]
            if len(atoms) != 1:
                raise derived.TokenMapError(
                    'ligand token %d on chain %r covers %d atoms, expected 1'
                    % (index, chain_id, len(atoms)))
            atom = atoms[0]
            tokens.append({'index': len(tokens), 'kind': 'ion' if len(indexs) == 1 else 'ligand',
                           'chain': chain_id, 'atom_name': str(atom_names[atom]),
                           'atom_index': atom_index,
                           'element': _element_symbol(atom_elements[atom])})
        chains.append({'id': chain_id, 'kind': 'ion' if len(indexs) == 1 else 'ligand',
                       'units': len(indexs), 'input_id': entity['id']})

    if len(tokens) != n_pae:
        raise derived.TokenMapError(
            'constructed inputs place %d tokens but the PAE matrix has %d rows'
            % (len(tokens), n_pae))
    return {'schema': 1, 'engine': 'chai1', 'n_tokens': n_pae,
            'pae_index': 'row/column index of the atom-expanded token in the PAE matrix',
            'token_layout': 'one token per protein residue, one per ligand heavy atom',
            'chains': chains, 'tokens': tokens}


def save_token_map(native, rid, sample_index, token_map_obj):
    name = 'tokens_' + rid + '_sample_' + str(sample_index) + '.json'
    atomic_json(Path(native) / name, token_map_obj)
    return name


class Chai1Engine:
    def __init__(self, assets, checkpoint, settings):
        self.assets, self.checkpoint, self.settings = Path(assets), Path(checkpoint), settings
        self.model = None
        self.checkpoint_load_count = 0
        self.cpu_to_gpu_transfer_count = 0
        self.native_component_loads = 0
        self.benchmark_metrics = None
        self.checkpoint_files = {}
        # Optional portable job-timeout hook, set by the worker and called between
        # samples; never a signal handler, never inside a native CUDA call.
        self.sample_deadline = None
        torch.set_grad_enabled(False)

    # -- preparation -------------------------------------------------------------------
    @native_boundary
    def prepare(self, source, attempt, seed):
        label = source.name
        data = yaml.safe_load(source.read_text())
        resolved = capability(data, label)
        proteins, ligands = resolved['proteins'], resolved['ligands']
        entities = ([{'id': entry['id'], 'kind': 'protein', 'sequence': entry['sequence']}
                     for entry in proteins]
                    + [{'id': entry['id'], 'kind': 'ligand', 'smiles': entry['smiles'],
                        'heavy_atoms': canonical_smiles(entry['smiles'])[1]['heavy_atoms']}
                       for entry in ligands])
        chains = [{'id': entity['id'], 'cif_chain': chain_letter(position + 1),
                   'kind': entity['kind'],
                   'residues': len(entity['sequence']) if entity['kind'] == 'protein' else 0,
                   'atoms': entity.get('heavy_atoms', 0)} for position, entity in enumerate(entities)]
        chemistry = {'protein_id': entities[0]['id'], 'residues': sum(c['residues'] for c in chains),
                     'chains': chains,
                     'ligands': [dict(canonical_smiles(entry['smiles'])[1], id=entry['id'])
                                 for entry in ligands],
                     'ligand': (dict(canonical_smiles(ligands[0]['smiles'])[1], id=ligands[0]['id'])
                                if ligands else None),
                     'heme': None}
        (attempt / 'source').mkdir()
        shutil.copyfile(source, attempt / 'source' / label)
        inputs = attempt / 'inputs'
        inputs.mkdir()
        shutil.copyfile(source, inputs / label)
        fasta = inputs / (source.stem + '.fasta')
        fasta.write_text(fasta_text({'entities': entities}))
        atomic_json(attempt / 'chemistry.json', chemistry)
        sample_seeds = [seed * 1000 + index for index in range(self.settings.samples)]
        atomic_json(attempt / 'protocol.json', {
            'engine': 'chai1', 'revision': os.environ.get('CHAI1_REVISION'),
            'settings': {'samples': self.settings.samples,
                         'num_trunk_recycles': self.settings.recycles,
                         'num_diffn_timesteps': self.settings.steps,
                         'low_memory': self.settings.low_memory,
                         'max_parallel_samples': self.settings.max_parallel_samples,
                         'output_format': self.settings.output_format,
                         'derived': self.settings.derived},
            'base_seed': seed, 'sample_seeds': sample_seeds,
            'seed_formula': 'base_seed*1000 + zero_based_sample_index',
            'num_diffusion_samples_per_call': 1, 'msa': False, 'templates': False,
            'fold_call': getattr(self, 'fold_call', None),
            'precision': ('native fp32 TorchScript exports; the traced ESM-2 3B embedder is fp16 '
                          'and is moved to the GPU for the embedding pass'),
            'flow': ('exact sequence + exact SMILES -> native FASTA -> tokenized feature context '
                     '-> frozen released Chai-1 TorchScript modules -> unmodified native outputs'),
            'entities': ([chain['id'] for chain in chains])})
        atomic_json(attempt / 'checkpoint-lock.json', {
            'model': str(self.checkpoint), 'revision': os.environ.get('CHAI1_REVISION'),
            'files': self.checkpoint_files or directory_identity(self.checkpoint)['files']})
        return {'source': Path(source), 'attempt': Path(attempt), 'rid': source.stem,
                'fasta': fasta, 'entities': entities, 'chemistry': chemistry,
                'sample_seeds': sample_seeds, 'data': data}

    # -- model -------------------------------------------------------------------------
    def _native_files(self):
        """Resolve the pinned artifact paths under the mounted asset directory."""
        return ([self.assets / 'models_v2' / name for name in CHAI_COMPONENTS]
                + [self.assets / CHAI_CONFORMERS, self.assets / CHAI_ESM])

    @native_boundary
    def check_checkpoint(self):
        """Fail with a readable message before chai_lab can attempt a download."""
        if not self.assets.is_dir():
            raise ValueError('asset directory does not exist: ' + str(self.assets))
        missing = [str(path) for path in self._native_files() if not path.is_file()]
        if missing:
            raise ValueError('asset directory is missing pinned Chai-1 files (mount a populated '
                             '--assets directory): ' + ', '.join(missing))

    def load(self):
        if self.model is not None:
            raise RuntimeError('second checkpoint load refused')
        if not torch.cuda.is_available():
            raise RuntimeError('resident native inference requires CUDA; CPU tests use FakeEngine')
        self.check_checkpoint()
        # chai_lab resolves its downloads directory at import time; point it at the
        # read-only mount before the first import so no network path is ever taken.
        os.environ['CHAI_DOWNLOADS_DIR'] = str(self.assets)
        from chai_lab import chai1
        started = time.perf_counter()
        if not self.checkpoint_files:
            self.checkpoint_files = directory_identity(self.checkpoint)['files']
        self.chai1 = chai1
        self.fold_call = fold_record(chai1.run_folding_on_context)
        self.model = chai1
        self.checkpoint_load_count += 1
        self.cpu_to_gpu_transfer_count += 1
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started

    # -- prediction --------------------------------------------------------------------
    @native_boundary
    def predict(self, prepared, attempt, seed):
        native = attempt / 'native'
        native.mkdir(exist_ok=True)
        samples = self.settings.samples
        metrics = {'scope': 'one native call per sample', 'forward_seconds': [], 'sample_seeds': [],
                   'native_component_loads': 0, 'output_write_seconds': 0.0,
                   'model_load_seconds': getattr(self, 'load_seconds', None)}
        torch.cuda.reset_peak_memory_stats()
        properties = torch.cuda.get_device_properties(0)
        metrics['cuda_device'] = {'logical_index': 0, 'name': properties.name,
                                  'uuid': str(properties.uuid) if getattr(properties, 'uuid', None) else None,
                                  'total_memory_bytes': properties.total_memory}
        metrics['allocator_baseline_bytes'] = {'allocated': torch.cuda.memory_allocated(),
                                               'reserved': torch.cuda.memory_reserved()}
        self.benchmark_metrics = metrics
        self.token_map_files = {}
        self.token_map_errors = {}
        rid = prepared['rid']
        model = getattr(self, 'chai1', None)
        if model is None:
            raise RuntimeError('predict called before load')
        device = torch.device('cuda:0')
        # The feature context is built once per job: the ESM embedding pass is the
        # expensive preparation, and every sample reuses the same tokenized inputs.
        feature_context = model.make_all_atom_feature_context(
            fasta_file=prepared['fasta'], output_dir=attempt / 'inputs' / 'features',
            use_esm_embeddings=True, use_msa_server=False, msa_directory=None,
            constraint_path=None, use_templates_server=False, esm_device=device)
        arrays = self._arrays(feature_context)
        for index, sample_seed in enumerate(prepared['sample_seeds']):
            check = self.sample_deadline
            if check is not None:
                check()  # portable job-timeout point, between samples
            number = index + 1
            raw = native / ('chai_sample_' + str(number))
            if raw.exists():
                raise RuntimeError('native sample directory already exists: ' + str(raw))
            torch.manual_seed(int(sample_seed))
            torch.cuda.manual_seed_all(int(sample_seed))
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                candidates = model.run_folding_on_context(
                    feature_context, output_dir=raw,
                    num_trunk_recycles=self.settings.recycles,
                    num_diffn_timesteps=self.settings.steps,
                    num_diffn_samples=1, seed=sample_seed, device=device,
                    low_memory=self.settings.low_memory)
            torch.cuda.synchronize()
            metrics['forward_seconds'].append(time.perf_counter() - started)
            metrics['sample_seeds'].append(sample_seed)
            # chai_lab loads six TorchScript components inside every fold call.
            metrics['native_component_loads'] += len(CHAI_COMPONENTS)
            self.native_component_loads += len(CHAI_COMPONENTS)
            label = rid + '_sample_' + str(number)
            written = time.perf_counter()
            cif = Path(candidates.cif_paths[0])
            shutil.copyfile(cif, native / (label + '.cif'))
            pae = np.asarray(candidates.pae[0].detach().cpu() if torch.is_tensor(candidates.pae)
                             else candidates.pae[0], dtype=np.float32)
            plddt = np.asarray(candidates.plddt[0].detach().cpu() if torch.is_tensor(candidates.plddt)
                               else candidates.plddt[0], dtype=np.float32)
            pde = np.asarray(candidates.pde[0].detach().cpu() if torch.is_tensor(candidates.pde)
                             else candidates.pde[0], dtype=np.float32)
            np.savez_compressed(native / ('pae_' + label + '.npz'), pae=pae)
            np.savez_compressed(native / ('plddt_' + label + '.npz'), plddt=plddt)
            np.savez_compressed(native / ('pde_' + label + '.npz'), pde=pde)
            ranking = candidates.ranking_data[0]
            ptm = ranking.ptm_scores.complex_ptm
            iptm = ranking.ptm_scores.interface_ptm
            atomic_json(native / ('confidence_' + label + '.json'), {
                'sample_index': number, 'sample_seed': sample_seed,
                'plddt_mean': float(plddt.mean()),
                'ptm': float(ptm.item() if torch.is_tensor(ptm) else ptm),
                'iptm': float(iptm.item() if torch.is_tensor(iptm) else iptm),
                'residues': prepared['chemistry']['residues'],
                'tokens': int(plddt.shape[0]),
                'plddt_scope': 'native_atom_expanded_token',
                'chains': [chain['cif_chain'] for chain in prepared['chemistry']['chains']],
                'pae_shape': list(pae.shape)})
            # Token map: derived from the collated token/atom arrays and this PAE,
            # never from the PAE shape alone.  A placement failure is recorded, not guessed.
            try:
                name = save_token_map(native, rid, number, token_map(prepared, arrays, pae))
                self.token_map_files[index] = name
            except derived.TokenMapError as exc:
                self.token_map_errors[index] = str(exc)
            metrics['output_write_seconds'] += time.perf_counter() - written
        metrics['predict_step_seconds'] = sum(metrics['forward_seconds'])
        metrics['allocator_peak_bytes'] = {'allocated': torch.cuda.max_memory_allocated(),
                                           'reserved': torch.cuda.max_memory_reserved()}
        atomic_json(attempt / 'benchmark_metrics.json', metrics)

    @staticmethod
    def _arrays(feature_context):
        """Copy chai_lab's tokenized structure context into plain numpy arrays."""
        context = feature_context.structure_context

        def numpy(value):
            if hasattr(value, 'detach'):
                value = value.detach().cpu()
            return np.asarray(value)
        return {'token_exists_mask': numpy(context.token_exists_mask).reshape(-1),
                'token_asym_id': numpy(context.token_asym_id).reshape(-1),
                'token_residue_index': numpy(context.token_residue_index).reshape(-1),
                'token_entity_type': numpy(context.token_entity_type).reshape(-1),
                'atom_token_index': numpy(context.atom_token_index).reshape(-1),
                'atom_exists_mask': numpy(context.atom_exists_mask).reshape(-1),
                'atom_ref_name': list(context.atom_ref_name),
                'atom_ref_element': numpy(context.atom_ref_element).reshape(-1)}

    # -- derived interface metrics -----------------------------------------------------
    @native_boundary
    def derive(self, prepared, attempt):
        """Score every written sample's PAE against its token map."""
        attempt = Path(attempt)
        native = attempt / 'native'
        rid = prepared['rid']
        predictions = []
        for index, sample_seed in enumerate(prepared['sample_seeds']):
            number = index + 1
            pae_path = native / ('pae_' + rid + '_sample_' + str(number) + '.npz')
            map_name = getattr(self, 'token_map_files', {}).get(index)
            reason = getattr(self, 'token_map_errors', {}).get(index)
            pae = None
            tokens = None
            try:
                with np.load(pae_path, allow_pickle=False) as arrays:
                    pae = np.asarray(arrays['pae'], dtype=float)
            except (OSError, ValueError, KeyError) as exc:
                reason = reason or 'PAE array is unreadable: %s (%s)' % (pae_path, exc)
            if reason is None:
                if not map_name:
                    reason = 'token map was not written for this sample'
                else:
                    try:
                        tokens = derived.read_token_map(native / map_name)
                    except derived.TokenMapError as exc:
                        reason = str(exc)
            pairs = None
            if reason is None:
                pairs, reason = derived.score(pae, tokens)
            predictions.append({
                'sample_index': index,
                'sample_seed': sample_seed,
                'pae': ('native/' + pae_path.name) if pae is not None else None,
                'token_map': ('native/' + map_name) if (tokens is not None and map_name) else None,
                'pairs': pairs,
                'derived_null_reason': reason,
            })
        return derived.write_derived(attempt, predictions)

    # -- verification ------------------------------------------------------------------
    @native_boundary
    def verify(self, prepared, attempt):
        from .verify import verify_job
        verify_job(prepared, attempt, self.settings)

    def failure_metrics(self, attempt):
        """Written by the shared native boundary; keep it cheap and non-raising."""
        try:
            payload = {'checkpoint_load_count': self.checkpoint_load_count,
                       'cpu_to_gpu_transfer_count': self.cpu_to_gpu_transfer_count,
                       'native_component_loads': self.native_component_loads,
                       'benchmark_metrics': self.benchmark_metrics}
            if torch.cuda.is_available():
                payload['allocator'] = {'allocated': torch.cuda.memory_allocated(),
                                        'reserved': torch.cuda.memory_reserved(),
                                        'peak_allocated': torch.cuda.max_memory_allocated(),
                                        'peak_reserved': torch.cuda.max_memory_reserved()}
            atomic_json(Path(attempt) / 'benchmark_failure.json', payload)
        except Exception:
            pass

    def snapshot(self):
        if self.model is None:
            return {'loaded': False}
        return {'loaded': True, 'device': 'cuda:0', 'engine': 'chai_lab',
                'checkpoint_load_count': self.checkpoint_load_count}

    def healthy(self):
        gc_collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return bool((torch.ones(1, device='cuda') + 1).item() == 2)
        except Exception:
            return False


def gc_collect():
    import gc
    gc.collect()

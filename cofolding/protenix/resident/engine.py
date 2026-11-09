"""Pinned Protenix 2.0.0 native preparation/model/writer boundary (GPU runtime).

Accepts the shared native YAML entity format used by the other cofold containers and maps it onto
Protenix's AF3-style input JSON: one entity per chain (`count: 1`), protein chains as
``proteinChain``, ligands as SMILES ``ligand`` entries with an explicit one-letter chain id, so
the written mmCIF keeps the chain letters this container records.

Protenix is driven through its own public inference pieces, never through a re-implementation:

* ``configs.configs_{base,data,inference,model_type}`` + ``protenix.config.config.parse_configs``
  build the exact configuration ``runner/inference.py`` builds.
* ``protenix.data.inference.infer_dataloader.InferenceDataset`` featurizes one input JSON into the
  token/atom feature dictionary that the model and the confidence head consume.  Its token arrays
  are the authoritative token identity used for the derived-metric token map.
* ``Protenix(configs)`` with the released checkpoint is the model; one forward with
  ``sample_diffusion.N_sample = 1`` produces exactly one structure per native call.
* ``runner.dumper.DataDumper`` writes the native mmCIF and the per-sample confidence JSON, so the
  files this container copies are byte-for-byte the upstream writer's output.

The per-token PAE that the derived interface scores need is emitted by the confidence head inside
``pred_dict["full_data"][i]["token_pair_pae"]`` (and rounded into the dumper's
``<sample>_summary_confidence_sample_<rank>.json``), never as a separate file: see SPEC.md §5.1.
"""
import os
from pathlib import Path
import shutil
import string
import time

import numpy as np
import torch
import yaml
from cofolding.common import derived

from .errors import native_boundary
from .identity import directory_identity
from .worker import atomic_json

CANONICAL = set('ACDEFGHIKLMNPQRSTVWY')

#: the model name Protenix 2.0.0 resolves to its released v2 weights
MODEL_NAME = 'protenix-v2'
CHECKPOINT_NAME = MODEL_NAME + '.pt'
VERSION_PACKAGES = ('protenix', 'torch', 'numpy', 'rdkit', 'biotite', 'gemmi')

#: Files the pinned Protenix data pipeline resolves under PROTENIX_ROOT_DIR.
PROTENIX_COMMON = ('components.cif', 'components.cif.rdkit_mol.pkl',
                   'clusters-by-entity-40.txt', 'obsolete_release_date.csv')

#: Configuration keys this container always passes explicitly to parse_configs.
CONFIG_OVERRIDES = ('model_name', 'input_json_path', 'dump_dir', 'seeds', 'use_msa',
                    'use_template', 'use_seeds_in_json', 'need_atom_confidence',
                    'sample_diffusion.N_sample', 'sample_diffusion.N_step', 'model.N_cycle',
                    'load_checkpoint_dir')

#: Inherited configuration that changes numerics or writes, recorded for the reader.
INHERITED_CONFIG = {
    'dtype': 'bf16',
    'enable_tf32': True,
    'enable_efficient_fusion': True,
    'enable_diffusion_shared_vars_cache': True,
    'triangle_attention': 'cuequivariance',
    'triangle_multiplicative': 'cuequivariance',
    'sample_diffusion.N_sample': 5,
    'sample_diffusion.N_step': 200,
    'model.N_cycle': 10,
    'num_workers': 0,
    'use_rna_msa': False,
    'sorted_by_ranking_score': True,
    'need_atom_confidence': False,
}


def call_record(settings):
    """The exact configuration overrides, split from the inherited defaults they override."""
    return {'explicit': list(CONFIG_OVERRIDES),
            'native_defaults': {key: value for key, value in INHERITED_CONFIG.items()
                                if key not in CONFIG_OVERRIDES},
            'inherited': dict(INHERITED_CONFIG),
            'entry_points': ['protenix.config.config.parse_configs',
                             'protenix.data.inference.infer_dataloader.InferenceDataset',
                             'protenix.model.protenix.Protenix.__call__',
                             'runner.dumper.DataDumper.dump'],
            'overrides': {'sample_diffusion.N_sample': 1,
                          'sample_diffusion.N_step': settings.steps,
                          'model.N_cycle': settings.recycles,
                          'use_msa': False,
                          'use_template': False,
                          'use_seeds_in_json': False,
                          'need_atom_confidence': True}}


def chain_letter(position):
    """Zero-based entity position -> written mmCIF chain id (``A``..``Z``, ``a``..``z``, then pairs)."""
    vocab = [*string.ascii_uppercase, *string.ascii_lowercase]
    vocab = vocab + [x + y for x in vocab for y in vocab]
    if not isinstance(position, int) or isinstance(position, bool) or not 0 <= position < len(vocab):
        raise ValueError('chain position must be within the CIF chain vocabulary')
    return vocab[position]


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
    """Map the shared YAML entity format onto Protenix; reject what it cannot express."""
    if not isinstance(data, dict):
        raise ValueError(label + ': native YAML must be a mapping')
    if 'seeds' in data or 'seed' in data:
        raise ValueError(label + ': seeds belong in the external --seeds list')
    if data.get('properties'):
        raise ValueError(label + ': affinity/property prediction is not supported by Protenix')
    for field in ('templates', 'constraints'):
        if data.get(field):
            raise ValueError(label + ': Protenix runs without ' + field)
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
                raise ValueError(label + ': Protenix runs without MSA; use msa: empty or omit it')
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
        elif kind in ('dna', 'rna'):
            raise ValueError(label + ': unsupported entity type ' + kind)
        elif kind == 'ion':
            raise ValueError(label + ': ion entities are not supported; pass the ion as a SMILES ligand')
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


def input_json(prepared):
    """The exact AF3-style JSON handed to Protenix; entity order and ids are frozen here.

    ``id`` is written explicitly so the upstream parser uses this container's chain letters
    instead of walking its own free-letter allocator; ``count`` is always 1 because one YAML
    entity is one chain, which also makes ``entity_id`` equal to the entity position.
    """
    sequences = []
    for entity in prepared['entities']:
        if entity['kind'] == 'protein':
            sequences.append({'proteinChain': {'sequence': entity['sequence'], 'count': 1,
                                               'id': [entity['cif_chain']]}})
        else:
            sequences.append({'ligand': {'ligand': entity['smiles'], 'count': 1,
                                         'id': [entity['cif_chain']]}})
    return [{'name': prepared['rid'], 'sequences': sequences}]


# -- token map -------------------------------------------------------------------------
def _atom_name(chars_row):
    """Decode AF3 ``ref_atom_name_chars`` (ord(c)-32, four slots, space padded)."""
    letters = []
    for code in np.asarray(chars_row).argmax(axis=-1):
        letters.append(chr(int(code) + 32))
    return ''.join(letters).rstrip()


def token_map(prepared, arrays, pae, elements=None):
    """One entry per PAE token, derived from Protenix's own tokenized feature arrays.

    ``arrays`` holds the token/atom columns of ``InferenceDataset``'s ``input_feature_dict``:
    ``token_index``, ``residue_index``, ``asym_id``, ``entity_id``, ``atom_to_token_idx``,
    ``is_ligand``, ``ref_element`` and ``ref_atom_name_chars``.

    Protenix tokenizes proteins one token per residue and non-standard residues / ligands one
    token per atom (``protenix/data/tokenizer.py``).  The PAE rows are therefore the tokens in
    feature order, and every mismatch — PAE size, entity order, residue count, ligand heavy-atom
    count, chain contiguity — is refused, never guessed.
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

    index = column('token_index').astype(int).tolist()
    asym = column('asym_id').astype(int).tolist()
    residue = column('residue_index').astype(int).tolist()
    atom_token = column('atom_to_token_idx').astype(int).tolist()
    is_ligand = column('is_ligand').astype(bool).tolist()
    ref_element = column('ref_element')
    ref_chars = arrays.get('ref_atom_name_chars')
    if ref_chars is None:
        raise derived.TokenMapError('token arrays are missing ref_atom_name_chars')
    ref_chars = np.asarray(ref_chars)
    if not (len(index) == len(asym) == len(residue)):
        raise derived.TokenMapError('token-level arrays disagree on token count')
    if index != list(range(len(index))):
        raise derived.TokenMapError('token_index is not the feature order')
    if ref_element.ndim != 2 or ref_chars.ndim != 3 or len(atom_token) != len(is_ligand) \
            or len(atom_token) != ref_element.shape[0] or len(atom_token) != ref_chars.shape[0]:
        raise derived.TokenMapError('atom-level arrays disagree on atom count')
    if len(index) != n_pae:
        raise derived.TokenMapError(
            'Protenix tokenized %d tokens but the PAE matrix has %d rows' % (len(index), n_pae))
    if atom_token and (min(atom_token) < 0 or max(atom_token) >= len(index)):
        raise derived.TokenMapError('atom_to_token_idx points outside the token range')

    # Chain order is the order asym_id first appears; every chain must be contiguous.
    order = []
    for value in asym:
        if not order or order[-1] != value:
            if value in order:
                raise derived.TokenMapError('asym id %r is not contiguous in the token order' % (value,))
            order.append(value)

    entities = prepared.get('entities') or []
    if len(order) != len(entities):
        raise derived.TokenMapError(
            'Protenix tokenized %d chains but %d entities were constructed'
            % (len(order), len(entities)))

    def symbol(row):
        if elements is None:
            return None
        position = int(np.asarray(row).argmax())
        if position >= len(elements):
            raise derived.TokenMapError('element index %d has no symbol' % position)
        return elements[position]

    atoms = list(range(len(atom_token)))
    tokens, chains = [], []
    for position, asym_id in enumerate(order):
        members = [token for token in index if asym[token] == asym_id]
        entity = entities[position]
        chain_id = entity['cif_chain']
        owners = [atom for atom in atoms if atom_token[atom] in members]
        if not owners:
            raise derived.TokenMapError('chain %r has no atoms' % (chain_id,))
        ligand = any(is_ligand[atom] for atom in owners)
        if not ligand:
            if entity['kind'] != 'protein':
                raise derived.TokenMapError(
                    'chain %r is a polymer in Protenix but %r in the input'
                    % (chain_id, entity['kind']))
            residues = [residue[token] for token in members]
            length = len(entity['sequence'])
            if len(residues) != length or sorted(residues) != list(range(1, length + 1)):
                raise derived.TokenMapError(
                    'chain %r has %d tokens for a %d-residue input'
                    % (chain_id, len(members), length))
            for token in members:
                tokens.append({'index': len(tokens), 'kind': 'protein', 'chain': chain_id,
                               'residue_index': int(residue[token])})
            chains.append({'id': chain_id, 'kind': 'protein', 'units': len(members),
                           'input_id': entity['id']})
            continue
        if entity['kind'] != 'ligand':
            raise derived.TokenMapError(
                'chain %r is a ligand in Protenix but %r in the input'
                % (chain_id, entity['kind']))
        expected = entity.get('heavy_atoms')
        if expected is not None and len(members) != int(expected):
            raise derived.TokenMapError(
                'ligand chain %r has %d tokens but the input SMILES has %d heavy atoms'
                % (chain_id, len(members), int(expected)))
        for atom_index, token in enumerate(members):
            own = [atom for atom in owners if atom_token[atom] == token]
            if len(own) != 1:
                raise derived.TokenMapError(
                    'ligand token %d on chain %r covers %d atoms, expected 1'
                    % (token, chain_id, len(own)))
            atom = own[0]
            tokens.append({'index': len(tokens), 'kind': 'ion' if len(members) == 1 else 'ligand',
                           'chain': chain_id, 'atom_name': _atom_name(ref_chars[atom]),
                           'atom_index': atom_index, 'element': symbol(ref_element[atom])})
        chains.append({'id': chain_id, 'kind': 'ion' if len(members) == 1 else 'ligand',
                       'units': len(members), 'input_id': entity['id']})

    if len(tokens) != n_pae:
        raise derived.TokenMapError(
            'constructed inputs place %d tokens but the PAE matrix has %d rows'
            % (len(tokens), n_pae))
    return {'schema': 1, 'engine': 'protenix', 'n_tokens': n_pae,
            'pae_index': 'row/column index of the token in the PAE matrix',
            'token_layout': 'one token per protein residue, one per ligand atom',
            'chains': chains, 'tokens': tokens}


def save_token_map(native, rid, sample_index, token_map_obj):
    name = 'tokens_' + rid + '_sample_' + str(sample_index) + '.json'
    atomic_json(Path(native) / name, token_map_obj)
    return name


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            # numpy has no bfloat16; the writer does the same conversion.
            value = value.float()
    return np.asarray(value)


class ProtenixEngine:
    def __init__(self, assets, checkpoint, settings):
        self.assets, self.checkpoint, self.settings = Path(assets), Path(checkpoint), settings
        self.model = None
        self.configs = None
        self.dumper = None
        self.checkpoint_load_count = 0
        self.cpu_to_gpu_transfer_count = 0
        self.benchmark_metrics = None
        self.checkpoint_files = {}
        self.token_map_files = {}
        self.token_map_errors = {}
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
        for position, entity in enumerate(entities):
            entity['cif_chain'] = chain_letter(position)
        chains = [{'id': entity['id'], 'cif_chain': entity['cif_chain'], 'kind': entity['kind'],
                   'residues': len(entity['sequence']) if entity['kind'] == 'protein' else 0,
                   'atoms': entity.get('heavy_atoms', 0)} for entity in entities]
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
        native_input = inputs / (source.stem + '.json')
        atomic_json(native_input, input_json({'rid': source.stem, 'entities': entities}))
        atomic_json(attempt / 'chemistry.json', chemistry)
        sample_seeds = [seed * 1000 + position for position in range(self.settings.samples)]
        atomic_json(attempt / 'protocol.json', {
            'engine': 'protenix', 'model_name': MODEL_NAME,
            'revision': os.environ.get('PROTENIX_REVISION'),
            'settings': {'samples': self.settings.samples,
                         'model.N_cycle': self.settings.recycles,
                         'sample_diffusion.N_step': self.settings.steps,
                         'max_parallel_samples': self.settings.max_parallel_samples,
                         'output_format': self.settings.output_format,
                         'derived': self.settings.derived},
            'base_seed': seed, 'sample_seeds': sample_seeds,
            'seed_formula': 'base_seed*1000 + zero_based_sample_index',
            'num_diffusion_samples_per_call': 1, 'msa': False, 'templates': False,
            'call': call_record(self.settings),
            'precision': ('native bf16 autocast for the diffusion and confidence heads, with '
                          'the released checkpoint loaded in fp32 parameters'),
            'flow': ('exact sequence + exact SMILES -> AF3-style JSON -> Protenix feature '
                     'dictionary -> released protenix-v2 checkpoint -> unmodified native outputs'),
            'entities': [chain['id'] for chain in chains]})
        atomic_json(attempt / 'checkpoint-lock.json', {
            'model': str(self.checkpoint), 'revision': os.environ.get('PROTENIX_REVISION'),
            'files': self.checkpoint_files or directory_identity(self.checkpoint)['files']})
        return {'source': Path(source), 'attempt': Path(attempt), 'rid': source.stem,
                'json': native_input, 'entities': entities, 'chemistry': chemistry,
                'sample_seeds': sample_seeds, 'data': data}

    # -- model -------------------------------------------------------------------------
    def _native_files(self):
        """Resolve the pinned artifact paths under the mounted PROTENIX_ROOT_DIR."""
        return ([self.assets / 'checkpoint' / CHECKPOINT_NAME]
                + [self.assets / 'common' / name for name in PROTENIX_COMMON])

    @native_boundary
    def check_checkpoint(self):
        """Fail with a readable message before Protenix can attempt a download."""
        if not self.assets.is_dir():
            raise ValueError('asset directory does not exist: ' + str(self.assets))
        missing = [str(path) for path in self._native_files() if not path.is_file()]
        if missing:
            raise ValueError('asset directory is missing pinned Protenix files (mount a populated '
                             '--assets directory): ' + ', '.join(missing))
        if self.checkpoint.resolve() != self.assets.resolve():
            # The launcher always mounts assets at /models and the checkpoint at /checkpoint,
            # so the two paths differ inside the container even when the caller passed the same
            # host directory. Accept that when the checkpoint mount carries the same artifacts,
            # and refuse only when it genuinely does not. Names and sizes are compared first so
            # the common case costs a directory listing rather than a hash of every file.
            expected = sorted((path.relative_to(self.assets).as_posix(),
                               path.stat().st_size) for path in self._native_files())
            found = []
            missing = []
            for relative, size in expected:
                candidate = self.checkpoint / relative
                if candidate.is_file() and candidate.stat().st_size == size:
                    found.append((relative, size))
                else:
                    missing.append(relative)
            if missing:
                raise ValueError(
                    'Protenix resolves weights and data cache from one PROTENIX_ROOT_DIR; '
                    '--assets and --checkpoint must name the same content, but the checkpoint '
                    'mount lacks: ' + ', '.join(missing))

    @staticmethod
    def build_configs(settings, input_json_path, dump_dir):
        """The configuration ``runner/inference.py`` builds, assembled twice like upstream."""
        from configs.configs_base import configs as configs_base
        from configs.configs_data import data_configs
        from configs.configs_inference import inference_configs
        from configs.configs_model_type import model_configs
        from protenix.config.config import parse_configs
        from typing import Mapping

        args = ('--model_name %s --input_json_path %s --dump_dir %s --seeds 1 '
                '--use_msa false --use_template false --use_seeds_in_json false '
                '--need_atom_confidence true --sample_diffusion.N_sample 1 '
                '--sample_diffusion.N_step %d --model.N_cycle %d --load_checkpoint_dir %s'
                % (MODEL_NAME, input_json_path, dump_dir, settings.steps, settings.recycles,
                   Path(os.environ['PROTENIX_ROOT_DIR']) / 'checkpoint'))
        base = {**configs_base, 'data': data_configs, **inference_configs}
        # First pass only resolves ``model_name``, exactly as upstream does.
        parse_configs(configs=base, arg_str=args, fill_required_with_null=True)
        # Second pass with the model-specific block merged in, exactly as upstream does.
        merged = {**configs_base, 'data': data_configs, **inference_configs}

        def deep_update(target, update):
            for key, value in update.items():
                if isinstance(value, Mapping) and key in target and isinstance(target[key], Mapping):
                    deep_update(target[key], value)
                else:
                    target[key] = value
            return target
        deep_update(merged, model_configs[MODEL_NAME])
        return parse_configs(configs=merged, arg_str=args, fill_required_with_null=True)

    def load(self):
        if self.model is not None:
            raise RuntimeError('second checkpoint load refused')
        if not torch.cuda.is_available():
            raise RuntimeError('resident native inference requires CUDA; CPU tests use FakeEngine')
        self.check_checkpoint()
        # configs.configs_data / configs_inference resolve every cache path from
        # PROTENIX_ROOT_DIR at import time; point it at the read-only mount first.
        os.environ['PROTENIX_ROOT_DIR'] = str(self.assets)
        started = time.perf_counter()
        from runner.inference import update_gpu_compatible_configs
        from protenix.model.protenix import Protenix

        configs = self.build_configs(self.settings, '/nonexistent/input.json', '/nonexistent/output')
        configs = update_gpu_compatible_configs(configs)
        self.device = torch.device('cuda:0')
        self.model = Protenix(configs).to(self.device)
        checkpoint = torch.load(self.assets / 'checkpoint' / CHECKPOINT_NAME,
                                map_location=self.device, weights_only=False)
        state = checkpoint['model']
        if next(iter(state)).startswith('module.'):
            state = {key[len('module.'):]: value for key, value in state.items()}
        self.model.load_state_dict(state_dict=state, strict=configs.load_strict)
        self.model.eval()
        if not self.checkpoint_files:
            self.checkpoint_files = directory_identity(self.checkpoint)['files']
        self.configs = configs
        self.checkpoint_load_count += 1
        self.cpu_to_gpu_transfer_count += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started

    # -- prediction --------------------------------------------------------------------
    @staticmethod
    def _arrays(feature_dict):
        """Copy Protenix's tokenized feature columns into plain numpy arrays."""
        return {'token_index': _numpy(feature_dict['token_index']).reshape(-1),
                'residue_index': _numpy(feature_dict['residue_index']).reshape(-1),
                'asym_id': _numpy(feature_dict['asym_id']).reshape(-1),
                'entity_id': _numpy(feature_dict['entity_id']).reshape(-1),
                'atom_to_token_idx': _numpy(feature_dict['atom_to_token_idx']).reshape(-1),
                'is_ligand': _numpy(feature_dict['is_ligand']).reshape(-1),
                'ref_element': _numpy(feature_dict['ref_element']),
                'ref_atom_name_chars': _numpy(feature_dict['ref_atom_name_chars'])}

    @native_boundary
    def predict(self, prepared, attempt, seed):
        native = attempt / 'native'
        native.mkdir(exist_ok=True)
        samples = self.settings.samples
        metrics = {'scope': 'one native forward per sample', 'forward_seconds': [],
                   'sample_seeds': [], 'featurization_seconds': None,
                   'output_write_seconds': 0.0,
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
        if self.model is None:
            raise RuntimeError('predict called before load')
        from runner.inference import update_gpu_compatible_configs, update_inference_configs
        from runner.dumper import DataDumper
        from protenix.data.inference.infer_dataloader import InferenceDataset
        from protenix.data.constants import get_all_elems
        from protenix.utils.seed import seed_everything
        from protenix.utils.torch_utils import to_device

        raw_root = native / 'protenix'
        # Rebuild the per-job configuration rather than mutating the loaded model's: the
        # configuration is cheap and this keeps model construction and job settings independent.
        configs = update_gpu_compatible_configs(
            self.build_configs(self.settings, str(prepared['json']), str(raw_root)))
        # Feature construction generates the SMILES reference conformer with RDKit; seeding
        # first keeps that conformer reproducible for a recorded base seed.
        seed_everything(int(seed), False)
        started = time.perf_counter()
        dataset = InferenceDataset(configs=configs)
        data, atom_array, error = dataset[0]
        metrics['featurization_seconds'] = time.perf_counter() - started
        if error:
            raise ValueError('Protenix featurization failed: ' + error.splitlines()[0])
        features = to_device(data['input_feature_dict'], self.device)
        configs = update_inference_configs(configs, int(data['N_token']))
        self.model.configs = configs
        arrays = self._arrays(data['input_feature_dict'])
        elements = get_all_elems()
        entity_poly_type = {key: value for key, value in data['entity_poly_type'].items()
                            if value != 'non-polymer'}
        for position, sample_seed in enumerate(prepared['sample_seeds']):
            check = self.sample_deadline
            if check is not None:
                check()  # portable job-timeout point, between samples
            number = position + 1
            raw = raw_root / (rid + '.seed_' + str(sample_seed))
            if raw.exists():
                raise RuntimeError('native sample directory already exists: ' + str(raw))
            seed_everything(int(sample_seed), False)
            torch.cuda.synchronize()
            started = time.perf_counter()
            precision = {'fp32': torch.float32, 'bf16': torch.bfloat16,
                         'fp16': torch.float16}[configs.dtype]
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=precision):
                prediction, _, _ = self.model(
                    input_feature_dict=features, label_full_dict=None, label_dict=None,
                    mode='inference', mc_dropout_apply_rate=configs.mc_dropout_apply_rate)
            torch.cuda.synchronize()
            metrics['forward_seconds'].append(time.perf_counter() - started)
            metrics['sample_seeds'].append(sample_seed)
            label = rid + '_sample_' + str(number)
            written = time.perf_counter()
            # Copy the unrounded confidence-head tensors BEFORE the writer runs: the dumper's
            # full-data cleaner rounds token_pair_pae/pde in place.
            full = prediction['full_data'][0]
            pae = _numpy(full['token_pair_pae']).astype(np.float32)
            pde = _numpy(full['token_pair_pde']).astype(np.float32)
            plddt = (_numpy(full['atom_plddt']).astype(np.float32) * 100.0)
            summary = prediction['summary_confidence'][0]
            # need_atom_confidence=True makes the engine's own writer emit
            # <rid>_full_data_sample_0.json, the only native file that carries the PAE.
            dumper = DataDumper(base_dir=str(raw),
                                need_atom_confidence=bool(configs.need_atom_confidence),
                                sorted_by_ranking_score=bool(configs.sorted_by_ranking_score))
            dumper.dump(dataset_name='', pdb_id=rid, seed=int(sample_seed), pred_dict=prediction,
                        atom_array=atom_array, entity_poly_type=entity_poly_type)
            produced = raw / rid / ('seed_' + str(sample_seed)) / 'predictions'
            cif = produced / (rid + '_sample_0.cif')
            if not cif.is_file():
                raise RuntimeError('Protenix writer did not produce ' + str(cif))
            shutil.copyfile(cif, native / (label + '.cif'))
            np.savez_compressed(native / ('pae_' + label + '.npz'), pae=pae)
            np.savez_compressed(native / ('pde_' + label + '.npz'), pde=pde)
            np.savez_compressed(native / ('plddt_' + label + '.npz'), plddt=plddt)

            def scalar(key):
                value = summary.get(key)
                if value is None:
                    return None
                return float(value.item()) if torch.is_tensor(value) else float(value)

            def series(key):
                value = summary.get(key)
                if value is None:
                    return None
                return (_numpy(value).astype(float).tolist() if torch.is_tensor(value)
                        else list(value))

            atomic_json(native / ('confidence_' + label + '.json'), {
                'sample_index': number, 'sample_seed': sample_seed,
                'plddt_mean': float(plddt.mean()),
                'plddt_scope': 'atom', 'plddt_units': '0-100 (native atom_plddt * 100)',
                'ptm': scalar('ptm'), 'iptm': scalar('iptm'),
                'ranking_score': scalar('ranking_score'), 'has_clash': scalar('has_clash'),
                'num_recycles': scalar('num_recycles'),
                'chain_ptm': series('chain_ptm'), 'chain_pair_iptm': series('chain_pair_iptm'),
                'residues': prepared['chemistry']['residues'],
                'tokens': int(pae.shape[0]), 'atoms': int(plddt.shape[0]),
                'chains': [chain['cif_chain'] for chain in prepared['chemistry']['chains']],
                'pae_shape': list(pae.shape), 'pde_shape': list(pde.shape)})
            # Token map: derived from the tokenized features and this PAE, never from the
            # PAE shape alone.  A placement failure is recorded, not guessed.
            try:
                name = save_token_map(native, rid, number,
                                      token_map(prepared, arrays, pae, elements))
                self.token_map_files[position] = name
            except derived.TokenMapError as exc:
                self.token_map_errors[position] = str(exc)
            metrics['output_write_seconds'] += time.perf_counter() - written
        metrics['predict_step_seconds'] = sum(metrics['forward_seconds'])
        metrics['allocator_peak_bytes'] = {'allocated': torch.cuda.max_memory_allocated(),
                                           'reserved': torch.cuda.max_memory_reserved()}
        atomic_json(attempt / 'benchmark_metrics.json', metrics)

    # -- derived interface metrics -----------------------------------------------------
    @native_boundary
    def derive(self, prepared, attempt):
        """Score every written sample's PAE against its token map."""
        attempt = Path(attempt)
        native = attempt / 'native'
        rid = prepared['rid']
        predictions = []
        for position, sample_seed in enumerate(prepared['sample_seeds']):
            number = position + 1
            pae_path = native / ('pae_' + rid + '_sample_' + str(number) + '.npz')
            map_name = getattr(self, 'token_map_files', {}).get(position)
            reason = getattr(self, 'token_map_errors', {}).get(position)
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
                'sample_index': position,
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
        return {'loaded': True, 'device': 'cuda:0', 'engine': 'protenix',
                'model_name': MODEL_NAME, 'checkpoint_load_count': self.checkpoint_load_count}

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

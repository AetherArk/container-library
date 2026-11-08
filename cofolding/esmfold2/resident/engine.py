"""Pinned ESMFold2 3.4.1.post1 native preparation/model/writer boundary (GPU runtime).

Accepts the shared native YAML entity format used by the Boltz2 container and maps it onto the
engine's single-protein, single-ligand, optional-heme input. Every rejection names the engine
capability that is missing, so an unsupported system never becomes an unexplained failure.
"""
from dataclasses import fields, is_dataclass, replace
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from .errors import native_boundary
import inspect

import numpy as np
import torch
import yaml

from .identity import directory_identity
from .worker import atomic_json

CANONICAL = set('ACDEFGHIKLMNPQRSTVWY')

#: keyword arguments the worker always passes explicitly to ``builder.fold``
FOLD_OVERRIDES = ('num_loops', 'num_sampling_steps', 'num_diffusion_samples', 'seed',
                  'include_embeddings', 'complex_id')


def fold_record(fn):
    """Effective per-call arguments: what we override, and the native defaults we inherit."""
    parameters = inspect.signature(fn).parameters
    defaults = {}
    for name, parameter in parameters.items():
        if name in ('self', 'model', 'input'):
            continue
        defaults[name] = None if parameter.default is inspect.Parameter.empty else parameter.default
    return {'explicit': list(FOLD_OVERRIDES),
            'native_defaults': {k: v for k, v in defaults.items() if k not in FOLD_OVERRIDES}}
VERSION_PACKAGES = ('esm', 'torch', 'transformers', 'triton', 'transformer-engine',
                    'transformer-engine-torch', 'cuequivariance-torch',
                    'cuequivariance-ops-torch-cu13', 'gemmi', 'rdkit')


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
    """Map the shared YAML entity format onto this engine; reject what it cannot express."""
    if not isinstance(data, dict):
        raise ValueError(label + ': native YAML must be a mapping')
    if 'seeds' in data or 'seed' in data:
        raise ValueError(label + ': seeds belong in the external --seeds list')
    if data.get('properties'):
        raise ValueError(label + ': affinity/property prediction is not supported by ESMFold2')
    for field in ('templates', 'constraints'):
        if data.get(field):
            raise ValueError(label + ': ESMFold2 runs without ' + field)
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
                raise ValueError(label + ': ESMFold2 runs without MSA; use msa: empty or omit it')
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
    if len(proteins) != 1:
        raise ValueError(label + ': ESMFold2 takes exactly one protein chain, got ' + str(len(proteins)))
    heme, plain = [], []
    for entry in ligands:
        (heme if canonical_smiles(entry['smiles'])[1]['iron_atoms'] else plain).append(entry)
    if len(heme) > 1 or len(plain) > 1:
        raise ValueError(label + ': ESMFold2 has one ligand slot and one heme slot; got '
                         + str(len(plain)) + ' ligand(s) and ' + str(len(heme)) + ' heme(s)')
    if not plain and not heme:
        raise ValueError(label + ': at least one ligand is required; apo folding is not this route')
    return {'protein': proteins[0], 'ligand': plain[0] if plain else None,
            'heme': heme[0] if heme else None}


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: cpu(getattr(value, field.name))
                                 for field in fields(value) if field.init})
    if isinstance(value, dict):
        return {str(key): cpu(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(child) for child in value)
    return value


def save_packet(directory, label, value):
    obj = cpu(value)
    torch.save(obj, directory / (label + '.pt'))
    items = {}

    def walk(item, path=''):
        if torch.is_tensor(item):
            items[path] = {'shape': list(item.shape), 'dtype': str(item.dtype),
                           'finite': bool(torch.isfinite(item).all())
                                     if item.is_floating_point() or item.is_complex() else True}
        elif isinstance(item, np.ndarray):
            items[path] = {'shape': list(item.shape), 'dtype': str(item.dtype),
                           'finite': bool(np.isfinite(item).all()) if np.issubdtype(item.dtype, np.number) else True}
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                walk(getattr(item, field.name), path + '/' + field.name)
        elif isinstance(item, dict):
            for key, child in item.items():
                walk(child, path + '/' + str(key))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, path + '/' + str(index))
    walk(obj)
    atomic_json(directory / (label + '_inventory.json'), items)
    return obj


def portable(value, arrays, path='result'):
    """JSON structure plus a flat numpy bundle; the pickle stays authoritative."""
    import dataclasses
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        arrays[path] = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
        return {'array': path, 'dtype': str(value.dtype), 'shape': list(value.shape)}
    if isinstance(value, np.ndarray):
        arrays[path] = value.astype(str) if value.dtype == object else value
        return {'array': path, 'dtype': str(value.dtype), 'shape': list(value.shape)}
    if dataclasses.is_dataclass(value):
        return {field.name: portable(getattr(value, field.name), arrays, path + '.' + field.name)
                for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): portable(child, arrays, path + '.' + str(key)) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(child, arrays, path + '.' + str(index)) for index, child in enumerate(value)]
    if isinstance(value, np.generic):
        return value.item()
    return value


class ESMFold2Engine:
    def __init__(self, assets, checkpoint, settings):
        self.assets, self.checkpoint, self.settings = Path(assets), Path(checkpoint), settings
        self.model = None
        self.builder = None
        self.checkpoint_load_count = 0
        self.cpu_to_gpu_transfer_count = 0
        self.benchmark_metrics = None
        self.checkpoint_files = {}
        torch.set_grad_enabled(False)

    # -- preparation -------------------------------------------------------------------
    @native_boundary
    def prepare(self, source, attempt, seed):
        label = source.name
        data = yaml.safe_load(source.read_text())
        resolved = capability(data, label)
        protein = resolved['protein']
        chemistry = {'protein_id': protein['id'], 'residues': len(protein['sequence']),
                     'ligand': None, 'heme': None}
        if resolved['ligand'] is not None:
            chemistry['ligand'] = dict(canonical_smiles(resolved['ligand']['smiles'])[1],
                                       id=resolved['ligand']['id'])
        if resolved['heme'] is not None:
            chemistry['heme'] = dict(canonical_smiles(resolved['heme']['smiles'])[1],
                                     id=resolved['heme']['id'])
        (attempt / 'source').mkdir()
        shutil.copyfile(source, attempt / 'source' / label)
        inputs = attempt / 'inputs'
        inputs.mkdir()
        shutil.copyfile(source, inputs / label)
        atomic_json(attempt / 'chemistry.json', chemistry)
        sample_seeds = [seed * 1000 + index for index in range(self.settings.samples)]
        atomic_json(attempt / 'protocol.json', {
            'engine': 'esmfold2', 'revision': os.environ.get('ESMFOLD2_REVISION'),
            'settings': {'samples': self.settings.samples, 'num_loops': self.settings.loops,
                         'num_sampling_steps': self.settings.steps, 'backend': self.settings.backend,
                         'max_parallel_samples': self.settings.max_parallel_samples,
                         'output_format': self.settings.output_format},
            'base_seed': seed, 'sample_seeds': sample_seeds,
            'seed_formula': 'base_seed*1000 + zero_based_sample_index',
            'num_diffusion_samples_per_call': 1, 'msa': False, 'templates': False,
            'fold_call': getattr(self, 'fold_call', None),
            'precision': 'FP32 folding weights; bf16 ESM-C loaded on CPU then moved to GPU',
            'flow': ('exact sequence + exact SMILES -> native input preparation -> frozen released '
                     'ESMFold2 and confidence heads -> unmodified native outputs'),
            'entities': [entry['id'] for entry in (protein, resolved['ligand'], resolved['heme'])
                         if entry is not None]})
        atomic_json(attempt / 'checkpoint-lock.json', {
            'model': str(self.checkpoint), 'revision': os.environ.get('ESMFOLD2_REVISION'),
            'files': self.checkpoint_files or directory_identity(self.checkpoint)['files']})
        from esm.models.esmfold2 import LigandInput, ProteinInput, StructurePredictionInput
        sequences = [ProteinInput(id='A', sequence=protein['sequence'], msa=None)]
        entities = ['A']
        if resolved['ligand'] is not None:
            sequences.append(LigandInput(id='L', smiles=resolved['ligand']['smiles']))
            entities.append('L')
        if resolved['heme'] is not None:
            sequences.append(LigandInput(id='H', smiles=resolved['heme']['smiles']))
            entities.append('H')
        return {'source': Path(source), 'attempt': Path(attempt), 'rid': source.stem,
                'input': StructurePredictionInput(sequences=sequences), 'chemistry': chemistry,
                'entities': entities, 'sample_seeds': sample_seeds, 'data': data}

    # -- model -------------------------------------------------------------------------
    @native_boundary
    def check_checkpoint(self):
        """Fail with a readable message when the mounted model directory is unusable."""
        config = self.checkpoint / 'config.json'
        if not config.exists():
            raise ValueError('checkpoint directory has no config.json: ' + str(self.checkpoint))
        if config.is_symlink() and not config.exists():
            raise ValueError('checkpoint symlink is dangling inside the mount: %s -> %s'
                             % (config, os.readlink(config)))
        try:
            config.read_bytes()
        except OSError as error:
            raise ValueError('checkpoint config.json is unreadable: %s (%s)' % (config, error))
        if not (self.checkpoint / 'ccd.pkl').exists():
            raise ValueError('checkpoint directory has no ccd.pkl: ' + str(self.checkpoint))

    def load(self):
        if self.model is not None:
            raise RuntimeError('second checkpoint load refused')
        if not torch.cuda.is_available():
            raise RuntimeError('resident native inference requires CUDA; CPU tests use FakeEngine')
        self.check_checkpoint()
        from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model
        started = time.perf_counter()
        if not self.checkpoint_files:
            self.checkpoint_files = directory_identity(self.checkpoint)['files']
        model = EsmFold2Model.from_pretrained(str(self.checkpoint), device='cpu',
                                              esmc_precision='bf16')
        model = model.to('cuda').eval()
        model.set_kernel_backend(None if self.settings.backend == 'reference' else self.settings.backend)
        self.model = model
        self.builder = ESMFold2InputBuilder(ccd_cache=self.checkpoint)
        self.fold_call = fold_record(ESMFold2InputBuilder.fold)
        self.checkpoint_load_count += 1
        self.cpu_to_gpu_transfer_count += 1
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started

    # -- prediction --------------------------------------------------------------------
    @native_boundary
    def predict(self, prepared, attempt, seed):
        capture = attempt / 'capture'
        capture.mkdir(exist_ok=True)
        native = attempt / 'native'
        native.mkdir(exist_ok=True)
        samples = self.settings.samples
        metrics = {'scope': 'one native call per sample', 'forward_seconds': [], 'sample_seeds': [],
                   'capture_hook_write_seconds': 0.0, 'output_write_seconds': 0.0,
                   'model_load_seconds': getattr(self, 'load_seconds', None)}
        torch.cuda.reset_peak_memory_stats()
        properties = torch.cuda.get_device_properties(0)
        metrics['cuda_device'] = {'logical_index': 0, 'name': properties.name,
                                  'uuid': str(properties.uuid) if getattr(properties, 'uuid', None) else None,
                                  'total_memory_bytes': properties.total_memory}
        metrics['allocator_baseline_bytes'] = {'allocated': torch.cuda.memory_allocated(),
                                               'reserved': torch.cuda.memory_reserved()}
        self.benchmark_metrics = metrics
        rid = prepared['rid']
        for index, sample_seed in enumerate(prepared['sample_seeds']):
            observed = {}

            def prehook(_module, args, kwargs):
                started = time.perf_counter()
                if 'prepared_features' not in observed:
                    observed['prepared_features'] = cpu(kwargs)
                metrics['capture_hook_write_seconds'] += time.perf_counter() - started

            def hook(_module, _args, result):
                started = time.perf_counter()
                if 'raw' not in observed:
                    observed['raw'] = cpu(result)
                metrics['capture_hook_write_seconds'] += time.perf_counter() - started

            model = self.model
            if model is None or self.builder is None:
                raise RuntimeError('predict called before load')
            # The native fold() applies LM dropout at inference when lm_dropout > 0, so a fixed
            # seed only reproduces a structure if the process RNG is seeded as well.
            torch.manual_seed(int(sample_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(sample_seed))
            before = model.register_forward_pre_hook(prehook, with_kwargs=True)
            after = model.register_forward_hook(hook)
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    result = self.builder.fold(
                        model, prepared['input'],
                        num_loops=self.settings.loops,
                        num_sampling_steps=self.settings.steps,
                        num_diffusion_samples=1, seed=sample_seed,
                        include_embeddings=True, complex_id=rid)
                torch.cuda.synchronize()
            finally:
                before.remove()
                after.remove()
            seconds = time.perf_counter() - started
            metrics['forward_seconds'].append(seconds)
            metrics['sample_seeds'].append(sample_seed)
            label = rid + '_sample_' + str(index + 1)
            written = time.perf_counter()
            (native / (label + '.cif')).write_text(result.complex.to_mmcif())
            arrays = {}
            structure = portable(result, arrays)
            np.savez_compressed(native / ('pae_' + label + '.npz'),
                                pae=np.asarray(result.pae, dtype=np.float32))
            np.savez_compressed(native / ('plddt_' + label + '.npz'),
                                plddt=np.asarray(result.plddt, dtype=np.float32))
            embeddings = {}
            for key in ('output_embedding_sequence', 'output_embedding_pair_pooled'):
                value = getattr(result, key, None)
                if value is not None:
                    embeddings[key] = np.asarray(value, dtype=np.float32)
            np.savez_compressed(native / ('embeddings_' + label + '.npz'), **embeddings)
            atomic_json(native / ('confidence_' + label + '.json'), {
                'sample_index': index + 1, 'sample_seed': sample_seed,
                'plddt_mean': float(np.asarray(result.plddt).mean()),
                'ptm': float(result.ptm), 'iptm': float(result.iptm),
                'residues': prepared['chemistry']['residues'],
                'tokens': int(np.asarray(result.plddt).shape[0]),
                'plddt_scope': 'native_atom_expanded_token',
                'chains': prepared['entities'],
                'pae_shape': list(np.asarray(result.pae).shape)})
            save_packet(capture, 'prepared_features_sample_' + str(index + 1),
                        observed.get('prepared_features'))
            save_packet(capture, 'raw_model_output_sample_' + str(index + 1), observed.get('raw'))
            save_packet(capture, 'full_result_sample_' + str(index + 1), result)
            np.savez_compressed(capture / ('full_result_sample_' + str(index + 1) + '.npz'), **arrays)
            atomic_json(capture / ('full_result_sample_' + str(index + 1) + '.json'), structure)
            metrics['output_write_seconds'] += time.perf_counter() - written
        metrics['predict_step_seconds'] = sum(metrics['forward_seconds'])
        metrics['allocator_peak_bytes'] = {'allocated': torch.cuda.max_memory_allocated(),
                                           'reserved': torch.cuda.max_memory_reserved()}
        atomic_json(attempt / 'benchmark_metrics.json', metrics)

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
        digest = hashlib.sha256()
        parameters = 0
        for name, parameter in self.model.named_parameters():
            digest.update(name.encode())
            digest.update(str(parameter.device).encode())
            digest.update(str(parameter.data_ptr()).encode())
            parameters += parameter.numel()
        return {'loaded': True, 'device': str(next(self.model.parameters()).device),
                'parameters': parameters, 'layout_sha256': digest.hexdigest(),
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

"""Pinned Boltz 2.2.1 native preparation/model/writer boundary (GPU runtime)."""
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from .benchmark import timed_forward
from .errors import native_boundary

import numpy as np
import torch
import yaml
from pytorch_lightning import seed_everything
import boltz.main as native
from boltz.data import const
from boltz.data.module import inferencev2 as inference
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzWriter
from boltz.model.models.boltz2 import Boltz2

from .worker import atomic_json


class StrictPredictionDataset(inference.PredictionDataset):
    """Same structure-only native operations; never substitute item zero on failure."""
    def __getitem__(self, idx):
        record = self.manifest.records[idx]
        data = inference.load_input(record=record, target_dir=self.target_dir,
                                    msa_dir=self.msa_dir, constraints_dir=self.constraints_dir,
                                    template_dir=self.template_dir, extra_mols_dir=self.extra_mols_dir,
                                    affinity=False)
        tokenized = self.tokenizer.tokenize(data)
        molecules = dict(self.canonicals)
        molecules.update(data.extra_mols)
        missing = set(tokenized.tokens['res_name'].tolist()) - set(molecules)
        molecules.update(inference.load_molecules(self.mol_dir, missing))
        options = record.inference_options
        features = self.featurizer.process(
            tokenized, molecules=molecules, random=np.random.default_rng(42),
            training=False, max_atoms=None, max_tokens=None, max_seqs=const.max_msa_seqs,
            pad_to_max_seqs=False, single_sequence_prop=0.0, compute_frames=True,
            inference_pocket_constraints=options.pocket_constraints if options else None,
            inference_contact_constraints=options.contact_constraints if options else None,
            compute_constraint_features=True, override_method=self.override_method,
            compute_affinity=False)
        features['record'] = record
        return features


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu(child) for key, child in value.items()}
    if isinstance(value, list):
        return [cpu(child) for child in value]
    if isinstance(value, tuple):
        return tuple(cpu(child) for child in value)
    return value


def cpu_packet(value):
    """Detached CPU copy plus the tensor inventory, without touching disk."""
    obj = cpu(value)
    items = {}
    def walk(v, path=''):
        if torch.is_tensor(v):
            items[path] = {'shape': list(v.shape), 'dtype': str(v.dtype),
                           'finite': bool(torch.isfinite(v).all()) if v.is_floating_point() or v.is_complex() else True}
        elif isinstance(v, dict):
            for key, child in v.items():
                walk(child, path + '/' + str(key))
        elif isinstance(v, (list, tuple)):
            for i, child in enumerate(v):
                walk(child, path + '/' + str(i))
    walk(obj)
    return obj, items


def save_packet(directory, label, value):
    obj, items = cpu_packet(value)
    torch.save(obj, directory / (label + '.pt'))
    atomic_json(directory / (label + '_inventory.json'), items)
    return obj


def save_summary(directory, saved):
    """Small numeric summary of a prediction, used by verification when capture is off."""
    # Same scalar set the verifier compares, plus the arrays written next to the structures.
    keep = ('coords', 'masks', 'pae', 'pde', 'plddt', 'pair_chains_iptm', 's', 'z',
            'confidence_score', 'ptm', 'iptm', 'ligand_iptm', 'protein_iptm',
            'complex_plddt', 'complex_iplddt', 'complex_pde', 'complex_ipde')
    fields = {key: saved[key] for key in keep if key in saved}
    torch.save(fields, directory / 'prediction_summary.pt')


class NativeEngine:
    def __init__(self, assets, checkpoint, settings):
        self.assets, self.checkpoint, self.settings = Path(assets), Path(checkpoint), settings
        self.model = None
        self.cpu_to_gpu_transfer_count = 0
        self.checkpoint_load_count = 0
        self.benchmark_metrics = None
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision('highest')
        for key in ('CUEQ_DEFAULT_CONFIG', 'CUEQ_DISABLE_AOT_TUNING'):
            os.environ.setdefault(key, '1')
        if not (self.assets / 'mols').is_dir():
            raise ValueError('offline assets must contain extracted mols/')

    @native_boundary
    def prepare(self, source, attempt, seed):
        seed_everything(seed, workers=True)
        data = yaml.safe_load(source.read_text())
        if not isinstance(data, dict):
            raise ValueError('native YAML must be a mapping')
        if 'seeds' in data or 'seed' in data:
            raise ValueError('seeds belong in external --seeds, not native YAML')
        if data.get('properties'):
            raise ValueError('affinity/property prediction is not supported by the structure-only resident worker')
        # Native parser interprets relative file references against cwd. Resolve
        # those paths to the source directory without changing scientific fields.
        def resolve(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in ('msa', 'cif', 'pdb', 'path') and isinstance(child, str) and child not in ('empty', 'auto'):
                        p = Path(child)
                        if not p.is_absolute():
                            value[key] = str((source.parent / p).resolve())
                    else:
                        resolve(child)
            elif isinstance(value, list):
                for child in value:
                    resolve(child)
        resolve(data)
        (attempt / 'source').mkdir()
        shutil.copyfile(source, attempt / 'source' / source.name)
        input_dir = attempt / 'inputs'
        input_dir.mkdir()
        staged = input_dir / source.name
        staged.write_text(yaml.safe_dump(data, sort_keys=False))
        native_dir = attempt / 'native'
        native.process_inputs(data=[staged], out_dir=native_dir, ccd_path=self.assets / 'ccd.pkl',
                              mol_dir=self.assets / 'mols', msa_server_url='', msa_pairing_strategy='greedy',
                              use_msa_server=False, boltz2=True, preprocessing_threads=1)
        processed = native_dir / 'processed'
        manifest = Manifest.load(processed / 'manifest.json')
        if len(manifest.records) != 1 or manifest.records[0].id != source.stem:
            raise ValueError('native preparation failed or returned the wrong input record')
        if manifest.records[0].affinity:
            raise ValueError('affinity requires a separate model; refused by structure-only worker')
        module = inference.Boltz2InferenceDataModule(
            manifest=manifest, target_dir=processed / 'structures', msa_dir=processed / 'msa',
            mol_dir=self.assets / 'mols', num_workers=0, constraints_dir=processed / 'constraints',
            template_dir=processed / 'templates', extra_mols_dir=processed / 'mols')
        dataset = StrictPredictionDataset(
            manifest=manifest, target_dir=module.target_dir, msa_dir=module.msa_dir,
            mol_dir=module.mol_dir, constraints_dir=module.constraints_dir,
            template_dir=module.template_dir, extra_mols_dir=module.extra_mols_dir)
        batch = inference.collate([dataset[0]])
        return {'batch': batch, 'module': module, 'data': data, 'rid': source.stem,
                'processed': processed, 'native': native_dir}

    def load(self):
        if self.model is not None:
            raise RuntimeError('second checkpoint load refused')
        if not torch.cuda.is_available():
            raise RuntimeError('resident native inference requires CUDA; CPU tests use FakeEngine only')
        diffusion = native.Boltz2DiffusionParams()
        # Pinned main.py uses 1.5 for Boltz2, not Boltz1's 1.638.
        if diffusion.step_scale != 1.5:
            raise RuntimeError('unexpected upstream Boltz2 diffusion default')
        steering = native.BoltzSteeringParams()
        steering.fk_steering = self.settings.use_potentials
        steering.physical_guidance_update = self.settings.use_potentials
        self.load_args = {
            'strict': True, 'predict_args': {
                'recycling_steps': self.settings.recycles, 'sampling_steps': self.settings.steps,
                'diffusion_samples': self.settings.samples,
                'max_parallel_samples': self.settings.max_parallel_samples,
                'write_confidence_summary': True, 'write_full_pae': True, 'write_full_pde': True},
            'map_location': 'cpu', 'diffusion_process_args': asdict(diffusion), 'ema': False,
            'use_kernels': not self.settings.no_kernels,
            'pairformer_args': asdict(native.PairformerArgsV2()),
            'msa_args': asdict(native.MSAModuleArgs()), 'steering_args': asdict(steering)}
        self.model = Boltz2.load_from_checkpoint(self.checkpoint, **self.load_args)
        self.checkpoint_load_count += 1
        self.model.eval().to(torch.device('cuda'))
        self.cpu_to_gpu_transfer_count += 1
        torch.cuda.synchronize()

    def snapshot(self):
        params = [(name, str(p.device), p.data_ptr()) for name, p in self.model.named_parameters()]
        devices = {}
        for _, device, _ in params:
            devices[device] = devices.get(device, 0) + 1
        return {'model_id': id(self.model), 'device': str(self.model.device),
                'parameter_data_ptr': params[0][2], 'named_parameter_count': len(params),
                'parameter_devices': devices,
                'named_parameter_storage_sha256': hashlib.sha256(json.dumps(params).encode()).hexdigest(),
                'cpu_to_gpu_transfer_count': self.cpu_to_gpu_transfer_count}

    @native_boundary
    def predict(self, prepared, attempt, seed):
        assert self.model is not None
        metrics = self.benchmark_metrics = {
            'forward_seconds': [], 'forward_null_reason': 'forward_not_completed',
            'predict_step_seconds': None, 'predict_step_null_reason': 'predict_step_not_completed',
            'capture_hook_write_seconds': 0.0, 'output_write_seconds': 0.0,
            'scope': 'input_seed_batch; forward excludes post-hook capture writes',
            'torch_cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version()}
        metrics['torch_cuda_version_null_reason'] = None if metrics['torch_cuda_version'] is not None else 'not_a_cuda_build'
        metrics['cudnn_version_null_reason'] = None if metrics['cudnn_version'] is not None else 'torch_cudnn_version_unavailable'
        try:
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            metrics['cuda_device'] = {'logical_index': device, 'name': properties.name,
                                      'uuid': str(properties.uuid) if getattr(properties, 'uuid', None) else None,
                                      'uuid_null_reason': None if getattr(properties, 'uuid', None) else 'torch_property_unavailable',
                                      'total_memory_bytes': properties.total_memory}
        except Exception:
            metrics['cuda_device'] = None
            metrics['cuda_device_null_reason'] = 'torch_device_query_unavailable'
        torch.cuda.synchronize()
        metrics['allocator_baseline_bytes'] = {
            'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()}
        torch.cuda.reset_peak_memory_stats()
        metrics['allocator_peak_reset'] = True
        capture = attempt / 'capture' if self.settings.capture == 'full' else None
        if capture is not None:
            capture.mkdir()
        begin = time.perf_counter()
        atomic_json(attempt / 'resolved_model_args.json', self.load_args)
        # Full per-parameter device/pointer evidence once per job; compact hashes
        # before/after every job in run.json bind ALL named parameters, not just one.
        atomic_json(attempt / 'model_parameters.json', {
            name: {'device': str(p.device), 'data_ptr': p.data_ptr()}
            for name, p in self.model.named_parameters()})
        metrics['output_write_seconds'] += time.perf_counter() - begin
        torch.cuda.synchronize()
        begin = time.perf_counter()
        batch = prepared['module'].transfer_batch_to_device(prepared['batch'], self.model.device, 0)
        torch.cuda.synchronize()
        metrics['batch_transfer_seconds'] = time.perf_counter() - begin
        begin = time.perf_counter()
        if capture is not None:
            save_packet(capture, 'consumed_batch', batch)
        metrics['output_write_seconds'] += time.perf_counter() - begin
        def capture_forward(module, args, result):
            begin = time.perf_counter()
            try:
                save_packet(capture, 'full_forward', result)
            finally:
                metrics['capture_hook_write_seconds'] += time.perf_counter() - begin
        original_forward = self.model.forward
        had_forward_override = 'forward' in self.model.__dict__
        self.model.forward = timed_forward(original_forward, torch.cuda.synchronize, metrics)
        hook = self.model.register_forward_hook(capture_forward) if capture is not None else None
        seed_everything(seed, workers=True)
        try:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            try:
                with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    prediction = self.model.predict_step(batch, 0)
            finally:
                torch.cuda.synchronize()
                metrics['predict_step_seconds'] = time.perf_counter() - begin
                metrics['predict_step_null_reason'] = None
                if metrics['forward_seconds']:
                    metrics['forward_null_reason'] = None
            begin = time.perf_counter()
            if capture is not None:
                saved = save_packet(capture, 'full_prediction', prediction)
            else:
                saved, _ = cpu_packet(prediction)
                save_summary(attempt, saved)
            metrics['output_write_seconds'] += time.perf_counter() - begin
            if saved['exception']:
                raise RuntimeError('native predict_step reported a recoverable prediction exception')
            writer = BoltzWriter(data_dir=prepared['processed'] / 'structures',
                                 output_dir=prepared['native'] / 'predictions',
                                 output_format=self.settings.output_format, boltz2=True, write_embeddings=True)
            begin = time.perf_counter()
            try:
                writer.write_on_batch_end(None, None, saved, [0], batch, 0, 0)
                torch.cuda.synchronize()
            finally:
                metrics['output_write_seconds'] += time.perf_counter() - begin
        finally:
            if hook is not None:
                hook.remove()
            if had_forward_override:
                self.model.forward = original_forward
            else:
                del self.model.forward
            metrics['allocator_peak_bytes'] = {
                'allocated': torch.cuda.max_memory_allocated(), 'reserved': torch.cuda.max_memory_reserved()}
            begin = time.perf_counter()
            atomic_json(attempt / 'model_parameters_after.json', {
                name: {'device': str(p.device), 'data_ptr': p.data_ptr()}
                for name, p in self.model.named_parameters()})
            metrics['output_write_seconds'] += time.perf_counter() - begin
            atomic_json(attempt / 'benchmark_metrics.json', metrics)
            # Never .cpu() the model or create a per-job Trainer.
            prepared['batch'] = None
            del batch
            gc.collect()

    @native_boundary
    def verify(self, prepared, attempt):
        from .verify import verify_job
        verify_job(prepared, attempt, self.settings)

    def healthy(self):
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force a tiny kernel and host read, not just an asynchronous enqueue.
            return bool((torch.ones(1, device='cuda') + 1).item() == 2)
        except Exception:
            return False

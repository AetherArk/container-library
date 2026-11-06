"""Small dependency-free native launch boundary (no model imports)."""
from pathlib import Path
import os


def offline_assets(cache, checkpoint=Path('/checkpoint')):
    """Replace native's unconditional downloader for structure-only requests.

    Native main.py downloads affinity weights before parsing inputs, but only
    consumes them in the affinity branch. This route rejects affinity inputs and
    supplies an explicit structure checkpoint, so neither default checkpoint nor
    the CCD archive is needed. Native processing reads the extracted mols/ tree.
    """
    if not (Path(cache)/'mols').is_dir():
        raise ValueError('Missing offline native cache asset: mols/')
    if not Path(checkpoint).is_file():
        raise ValueError('Missing structure checkpoint')


def native_args(stage, seed, settings):
    args = ['predict', '/run/inputs', '--out_dir', str(stage/'native'),
            '--cache', '/models', '--checkpoint', '/checkpoint', '--model',
            'boltz2', '--seed', str(seed), '--num_workers', str(settings.get('num_workers', 0)),
            '--preprocessing-threads', str(settings.get('preprocessing_threads', settings['threads'] or os.cpu_count() or 1)),
            '--write_full_pae', '--write_full_pde',
            '--output_format', settings.get('output_format', 'mmcif')]
    if settings.get('write_embeddings', True):
        args += ['--write_embeddings']
    if settings.get('use_potentials', False):
        args += ['--use_potentials']
    for key in ('recycling_steps', 'sampling_steps', 'diffusion_samples', 'max_parallel_samples'):
        args += ['--'+key, str(settings[key])]
    if settings['no_kernels']:
        args += ['--no_kernels']
    return args

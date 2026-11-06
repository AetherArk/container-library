#!/usr/bin/env python3
"""Replay a frozen historical case through batch.py; compare every native byte."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import batch

ROOT = Path(__file__).resolve().parent
CASES = ROOT.parent/'test-cases'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON number: '+value)
    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=invalid)


def keys(obj, required, optional=()):
    require(type(obj) is dict and set(required) <= set(obj) <= set(required)|set(optional),
            'Unexpected or missing fields')


def text(value, pattern=None):
    require(type(value) is str and bool(value), 'Nonempty string required')
    if pattern:
        require(re.fullmatch(pattern, value) is not None, 'Invalid identifier or hash')


def integer(value, minimum=0, maximum=None):
    require(type(value) is int and value >= minimum and (maximum is None or value <= maximum),
            'Invalid integer')


def native_names(record, samples, output_format='mmcif', write_embeddings=True):
    names = {'embeddings_'+record+'.npz'} if write_embeddings else set()
    extension = 'pdb' if output_format == 'pdb' else 'cif'
    for rank in range(samples):
        base = f'{record}_model_{rank}'
        names |= {base+'.'+extension, 'confidence_'+base+'.json',
                  *{key+'_'+base+'.npz' for key in ('pae', 'pde', 'plddt')}}
    return names


def load_case(directory):
    case = read_json(directory/'input.json')
    settings = read_json(directory/'boltz2/settings.json')
    expected = read_json(directory/'boltz2/expected.json')
    keys(case, ('id', 'protein', 'ligand', 'msa', 'seed', 'samples'), ('cofactors',))
    text(case['id'], r'[A-Za-z0-9_-]+')
    keys(case['protein'], ('id', 'sequence'))
    text(case['protein']['sequence'], r'[ACDEFGHIKLMNPQRSTVWY]+')
    keys(case['ligand'], ('id', 'smiles'))
    require(type(case.get('cofactors', [])) is list, 'cofactors must be a list')
    entities = [case['protein'], *case.get('cofactors', []), case['ligand']]
    ids = set()
    for index, entity in enumerate(entities):
        if index:
            keys(entity, ('id', 'smiles'))
            text(entity['smiles'])
        text(entity['id'], r'[A-Za-z][A-Za-z0-9]*')
        require(entity['id'] not in ids, 'Duplicate entity ID')
        ids.add(entity['id'])
    integer(case['seed'], maximum=4294967295)
    integer(case['samples'], 1)
    msa_bytes = None
    if case['msa'] != 'empty':
        keys(case['msa'], ('file', 'sha256'))
        text(case['msa']['file']); text(case['msa']['sha256'], r'[0-9a-f]{64}')
        relative = Path(case['msa']['file'])
        require(not relative.is_absolute() and '..' not in relative.parts and relative.name.endswith('.a3m.gz'),
                'MSA must be a contained compressed A3M')
        source = (directory/relative).resolve()
        require(source.is_relative_to(directory.resolve()), 'MSA escapes case directory')
        with gzip.open(source, 'rb') as stream:
            msa_bytes = bytes(stream.read())
        require(hashlib.sha256(msa_bytes).hexdigest() == case['msa']['sha256'], 'Decompressed MSA hash mismatch')
    keys(settings, ('record_id', 'recycling_steps', 'sampling_steps', 'max_parallel_samples',
                    'no_kernels', 'threads', 'preprocessing_threads'),
         ('output_format', 'use_potentials', 'write_embeddings', 'num_workers'))
    settings = {'output_format': 'mmcif', 'use_potentials': False, 'write_embeddings': True, 'num_workers': 0, **settings}
    require(settings['output_format'] in ('pdb', 'mmcif'), 'output_format must be pdb or mmcif')
    for key in ('use_potentials', 'write_embeddings'):
        require(type(settings[key]) is bool, key+' must be boolean')
    text(settings['record_id'], r'[A-Za-z0-9_-]+')
    for key in ('sampling_steps', 'max_parallel_samples', 'preprocessing_threads'):
        integer(settings[key], 1)
    integer(settings['threads'])
    integer(settings['num_workers'])
    integer(settings['recycling_steps'])
    require(type(settings['no_kernels']) is bool, 'no_kernels must be boolean')
    keys(expected, ('historical_run', 'image_id', 'checkpoint_sha256', 'native_files', 'rank_to_original_sample'))
    text(expected['historical_run']); text(expected['image_id'], r'sha256:[0-9a-f]{64}')
    text(expected['checkpoint_sha256'], r'[0-9a-f]{64}')
    require(type(expected['native_files']) is dict and
            set(expected['native_files']) == native_names(settings['record_id'], case['samples'],
                                                         settings['output_format'], settings['write_embeddings']),
            'Historical inventory must contain every native prediction file')
    for spec in expected['native_files'].values():
        keys(spec, ('sha256', 'bytes'))
        text(spec['sha256'], r'[0-9a-f]{64}'); integer(spec['bytes'], 1)
    rank = expected['rank_to_original_sample']
    require(rank is None or (type(rank) is list and all(type(i) is int for i in rank) and
            sorted(rank) == list(range(case['samples']))), 'Historical ranks must be null or a sample permutation')
    return case, settings, expected, msa_bytes


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialize(output, case, settings, msa_bytes):
    """Never replace an existing input, including on resume."""
    protein = dict(case['protein'], msa='empty')
    files = {}
    if msa_bytes is not None:
        files['msa/msa.a3m'] = msa_bytes
        protein['msa'] = '../msa/msa.a3m'
    data = {'version': 1, 'sequences': [{'protein': protein},
            *[{'ligand': entity} for entity in case.get('cofactors', [])], {'ligand': case['ligand']}]}
    files['inputs/'+settings['record_id']+'.yaml'] = (json.dumps(data, indent=2)+'\n').encode()
    for dirname in ('inputs', 'msa'):
        directory = output/dirname
        require(not directory.is_symlink(), 'Staged directory cannot be a symlink')
        if directory.exists():
            require({str(p.relative_to(output)) for p in directory.rglob('*') if p.is_file()} ==
                    {name for name in files if name.startswith(dirname+'/')}, 'Incompatible existing input inventory')
    for name, payload in files.items():
        target = output/name
        require(not target.is_symlink(), 'Staged input cannot be a symlink')
        if target.exists():
            require(target.read_bytes() == payload, 'Incompatible existing input: '+name)
    for name, payload in files.items():
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(payload)
    (output/'msa').mkdir(exist_ok=True)


def batch_command(args, case, settings):
    cmd = [sys.executable, str(ROOT/'batch.py'), '--input', str(args.output/'inputs'),
           '--output', str(args.output/'prediction'), '--cache', str(args.cache),
           '--checkpoint', str(args.checkpoint), '--gpu', args.gpu, '--image', args.image,
           '--seeds', str(case['seed']), '--diffusion_samples', str(case['samples'])]
    for key in ('recycling_steps', 'sampling_steps', 'max_parallel_samples', 'threads', 'preprocessing_threads'):
        cmd += ['--'+key.replace('_', '-') if key == 'preprocessing_threads' else '--'+key, str(settings[key])]
    if settings['no_kernels']:
        cmd.append('--no_kernels')
    cmd += ['--output-format', settings.get('output_format', 'mmcif')]
    cmd += ['--num-workers', str(settings.get('num_workers', 0))]
    if settings.get('use_potentials', False):
        cmd.append('--use-potentials')
    if not settings.get('write_embeddings', True):
        cmd.append('--no-write-embeddings')
    if args.resume:
        cmd.append('--resume')
    if args.dry_run:
        cmd.append('--dry-run')
    return cmd


def compare(output, case, settings, expected):
    stage = output/'prediction'/f"seed-{case['seed']}"
    pred = stage/'native/boltz_results_inputs/predictions'/settings['record_id']
    actual = {str(p.relative_to(pred)): p for p in pred.rglob('*') if p.is_file()}
    frozen = expected['native_files']
    missing = sorted(set(frozen)-set(actual)); extra = sorted(set(actual)-set(frozen))
    differing = sorted(name for name in set(frozen)&set(actual)
                       if actual[name].is_symlink() or actual[name].stat().st_size != frozen[name]['bytes']
                       or batch.sha(actual[name]) != frozen[name]['sha256'])
    receipt = read_json(stage/'verified.json') if (stage/'verified.json').is_file() else {}
    ranks = receipt.get('rank_to_original_sample', {}).get(settings['record_id'])
    valid_ranks = (type(ranks) is list and all(type(i) is int for i in ranks)
                   and sorted(ranks) == list(range(case['samples'])))
    ranks_checked = expected['rank_to_original_sample'] is not None
    ranks_match = valid_ranks and (not ranks_checked or ranks == expected['rank_to_original_sample'])
    accepted = (receipt.get('status') == 'PASS' and type(receipt.get('models')) is int
                and receipt['models'] == case['samples'])
    return {'status': 'PASS' if not (missing or extra or differing) and ranks_match and accepted else 'FAIL',
            'missing': missing, 'extra': extra, 'differing': differing,
            'rank_to_original_sample': ranks, 'ranks_match': ranks_match,
            'historical_ranks_checked': ranks_checked, 'batch_verified': accepted,
            'historical_run': expected['historical_run'], 'historical_image_id': expected['image_id']}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=('cyp', 'pxr'), required=True)
    for name in ('output', 'cache', 'checkpoint', 'gpu'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--image', default=batch.IMAGE)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.output = batch.mount_path(args.output)
    args.cache = batch.mount_path(args.cache); args.checkpoint = batch.mount_path(args.checkpoint)
    for source in (CASES.resolve(), args.cache, args.checkpoint):
        require(args.output != source and args.output not in source.parents and source not in args.output.parents,
                'Output must be separate from cases and assets')
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output/'comparison.json'
    result_path.unlink(missing_ok=True)  # No stale PASS survives any attempted check.
    try:
        case, settings, expected, msa_bytes = load_case(CASES/args.case)
        require(batch.sha(args.checkpoint) == expected['checkpoint_sha256'], 'Selected checkpoint hash mismatch')
        materialize(args.output, case, settings, msa_bytes)
        subprocess.run(batch_command(args, case, settings), check=True)
        if args.dry_run:
            return 0
        result = compare(args.output, case, settings, expected)
        atomic_json(result_path, result)
        print(json.dumps(result))
        return 0 if result['status'] == 'PASS' else 1
    except Exception as error:
        atomic_json(result_path, {'status': 'FAIL', 'error': str(error),
                                  'missing': [], 'extra': [], 'differing': []})
        print('Historical comparison failed: '+str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

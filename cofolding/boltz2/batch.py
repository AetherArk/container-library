#!/usr/bin/python3
"""Offline Boltz-2 native YAML-directory batches, one process per explicit seed.
Use /usr/bin/python3 (PyYAML required). No implicit MSA, downloads or retries.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import uuid

ROOT = Path(__file__).absolute().parent
IMAGE = 'container-library/boltz2:2.2.1'


def resolve_image(reference):
    """Resolve a locally installed tag, registry digest or ID; never pull."""
    if not reference or reference.startswith('-') or any(c.isspace() for c in reference):
        raise ValueError('Invalid image reference')
    result = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', reference],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError(f'Image not installed: {reference}. Pull first: docker pull {reference}')
    identity = result.stdout.strip()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', identity):
        raise ValueError('Docker did not return an immutable installed image ID')
    return identity


def lock_root(value=None):
    """Use the same user-local GPU lock namespace for every invocation."""
    if value is None:
        value = os.environ.get('COFOLD_GPU_LOCK_ROOT')
    if value is None:
        state = Path(os.environ.get('XDG_STATE_HOME', Path.home()/'.local/state'))
        value = state/'cofolding/gpu-locks'
    return safe_path(value)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def safe_path(value):
    """Normal portable path resolution, including user homes and symlinks."""
    return Path(value).expanduser().resolve()


def mount_path(value):
    path = safe_path(value)
    # -v uses colons as separators; never reinterpret a host filename as options.
    if ':' in str(path) or '\n' in str(path) or '\r' in str(path):
        raise ValueError('Host mount paths cannot contain colons or newlines')
    return path


def inputs(directory):
    import yaml
    class UniqueLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader, node):
        pairs = loader.construct_pairs(node, deep=True)
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate YAML key: '+str(key))
            result[key] = value
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    directory = safe_path(directory)
    records = []; files = {}
    paths = sorted(directory.iterdir())
    if not paths: raise ValueError('Empty input directory')
    ids = set()
    for path in paths:
        if path.suffix not in ('.yaml', '.yml'): raise ValueError('Input directory must contain YAMLs only; store MSA separately')
        path = safe_path(path)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', path.stem) or path.stem in ids: raise ValueError('Unsafe/duplicate record ID')
        ids.add(path.stem)
        data = yaml.load(path.read_text(), Loader=UniqueLoader)
        if not isinstance(data, dict) or set(data) != {'version','sequences'} or type(data['version']) is not int or data['version'] != 1:
            raise ValueError('Only version:1 and sequences supported; templates, constraints, affinity rejected')
        if not isinstance(data['sequences'], list) or not data['sequences']: raise ValueError('Empty sequences')
        chains = set(); protein = False
        for entry in data['sequences']:
            if not isinstance(entry,dict) or len(entry) != 1: raise ValueError('Invalid entity')
            kind, entity = next(iter(entry.items()))
            allowed = {'id','sequence','msa'} if kind == 'protein' else {'id','smiles'} if kind == 'ligand' else set()
            if not allowed or not isinstance(entity,dict) or set(entity) != allowed: raise ValueError('Only unmodified protein with explicit msa and SMILES ligand supported')
            cid = entity['id']
            if not isinstance(cid,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9]*',cid) or cid in chains: raise ValueError('One unique string chain ID per entity required')
            chains.add(cid)
            if kind == 'protein':
                protein = True
                if not isinstance(entity['sequence'],str) or not entity['sequence'] or set(entity['sequence'])-set('ACDEFGHIKLMNPQRSTVWY'): raise ValueError('Canonical protein sequence required')
                msa = entity['msa']
                if msa != 'empty':
                    if not isinstance(msa,str): raise ValueError('msa must be empty or local .a3m')
                    p = safe_path(Path(msa) if Path(msa).is_absolute() else path.parent/msa)
                    if p.suffix != '.a3m': raise ValueError('Only precomputed A3M supported')
                    digest = sha(p); target = 'msa/'+digest+'.a3m'
                    files[target] = (p, digest)
                    entity['msa'] = '/run/'+target
            elif not isinstance(entity['smiles'],str) or not entity['smiles']: raise ValueError('SMILES required')
        if sum('protein' in e for e in data['sequences']) != 1: raise ValueError('This narrow runner supports exactly one protein chain')
        files['sources/'+path.name] = (path, sha(path))
        records.append({'id':path.stem,'yaml':data})
    return records, files


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input','output','cache','checkpoint','gpu'): p.add_argument('--'+name, required=True)
    p.add_argument('--image', default=IMAGE)
    p.add_argument('--lock-root', help='Shared GPU locks (default: COFOLD_GPU_LOCK_ROOT or user XDG state directory)')
    p.add_argument('--cpus', type=float, default=8, help='Inference CPU limit')
    p.add_argument('--memory', default='36g', help='Inference RAM and RAM+swap limit')
    p.add_argument('--verify-cpus', type=float, default=2)
    p.add_argument('--verify-memory', default='8g')
    p.add_argument('--threads', type=int, default=2, help='Native preprocessing and numerical library threads')
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    for name, default in [('diffusion_samples',1),('recycling_steps',3),('sampling_steps',200),('max_parallel_samples',1)]: p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--no_kernels', action='store_true')
    p.add_argument('--timeout', type=int, default=1800, help='Per-seed inference limit in seconds')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def command(a, output, script, seed, gpu=False):
    memory = a.memory if gpu else a.verify_memory
    cmd = ['docker','run','--rm','--pull','never','--network','none','--cpus',str(a.cpus if gpu else a.verify_cpus),'--memory',memory,'--memory-swap',memory,'--user',f'{os.getuid()}:{os.getgid()}',
           '-v',f'{output}:/run','-v',f'{output}/code:/scripts:ro','-v',f'{a.cache}:/models:ro','-v',f'{a.checkpoint}:/checkpoint:ro']
    if gpu: cmd += ['--gpus','device='+a.gpu]
    else: cmd += ['-e','NVIDIA_VISIBLE_DEVICES=void']
    for k,v in {'TMPDIR':'/run/tmp','HOME':'/run/tmp','NUMBA_CACHE_DIR':'/run/tmp/numba','TRITON_CACHE_DIR':'/run/tmp/triton','HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':str(a.threads),'OPENBLAS_NUM_THREADS':str(a.threads),'MKL_NUM_THREADS':str(a.threads),'PYTHONDONTWRITEBYTECODE':'1'}.items(): cmd += ['-e',k+'='+v]
    return cmd+['--entrypoint','python',a.image,'/scripts/'+script,str(seed)]


def run_owned(cmd, log, timeout):
    name='boltz-batch-'+uuid.uuid4().hex[:16]
    cmd=cmd[:2]+['--name',name]+cmd[2:]
    def interrupt(signum, frame): raise KeyboardInterrupt
    old={s:signal.signal(s,interrupt) for s in (signal.SIGTERM,signal.SIGINT)}
    proc=None
    try:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        rc=proc.wait(timeout=timeout)
        if rc: raise subprocess.CalledProcessError(rc,cmd)
    finally:
        try:
            try:
                subprocess.run(['docker','stop','-t','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
            except (subprocess.TimeoutExpired, OSError):
                subprocess.run(['docker','kill',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
            status=subprocess.run(['docker','inspect','--format','{{.State.Running}}',name],capture_output=True,text=True,timeout=15)
            if status.returncode==0 and status.stdout.strip()=='true':
                subprocess.run(['docker','kill',name],check=True,stdout=subprocess.DEVNULL,timeout=15)
            if proc is not None:
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait(timeout=15)
        finally:
            for s,handler in old.items(): signal.signal(s,handler)


def main():
    a = parser().parse_args()
    if len(set(a.seeds)) != len(a.seeds) or any(not 0 <= s <= 4294967295 for s in a.seeds): raise ValueError('Unique uint32 native seeds required')
    if min(a.diffusion_samples,a.sampling_steps,a.max_parallel_samples) < 1 or a.recycling_steps < 0: raise ValueError('Invalid sampling schedule')
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]+',a.gpu): raise ValueError('Explicit allocated GPU UUID required')
    if a.timeout < 1 or a.threads < 1: raise ValueError('Positive timeout and thread count required')
    if any(not math.isfinite(v) or v <= 0 for v in (a.cpus,a.verify_cpus)): raise ValueError('Positive finite CPU limits required')
    if any(not re.fullmatch(r'[1-9][0-9]*[bkmgBKMG]?',v) for v in (a.memory,a.verify_memory)): raise ValueError('Positive Docker memory limits required (bytes, b, k, m or g)')
    records, files = inputs(a.input)
    a.cache = str(mount_path(a.cache)); a.checkpoint = str(mount_path(a.checkpoint)); out = mount_path(a.output)
    if not (Path(a.cache)/'mols').is_dir(): raise ValueError('Missing offline native cache asset: mols/')
    if not Path(a.checkpoint).is_file(): raise ValueError('Missing structure checkpoint')
    for source in (safe_path(a.input),Path(a.cache),Path(a.checkpoint)):
        if out == source or out in source.parents or source in out.parents:
            raise ValueError('Output must be separate from input, cache and checkpoint')
    a.image = resolve_image(a.image)
    sources = {'batch.py':sha(ROOT/'batch.py'), **{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'batch_native').glob('*.py'))}}
    request = {'records':records,'source_files':{k:v[1] for k,v in files.items()},'code':sources,'checkpoint_sha256':sha(a.checkpoint),
               'settings':{k:v for k,v in vars(a).items() if k not in ('resume','dry_run','output','input','lock_root')}}
    if a.dry_run:
        print(json.dumps({'request':request,'models':len(records)*len(a.seeds)*a.diffusion_samples,'commands':[command(a,out,'observe.py',s,True) for s in a.seeds]},indent=2)); return
    lockdir = lock_root(a.lock_root); lockdir.mkdir(parents=True,exist_ok=True)
    with (lockdir/(a.gpu+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        memory,util=map(int,subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
        apps=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if memory>512 or util>0 or apps: raise RuntimeError('GPU busy; no work started')

        if out.exists():
            if not a.resume or json.loads((out/'request.json').read_text()) != request: raise ValueError('Output exists or request differs; use identical --resume only')
            for k,digest in request['source_files'].items():
                if sha(out/k) != digest: raise ValueError('Staged source changed: '+k)
            for rec in records:
                if json.loads((out/'inputs'/(rec['id']+'.yaml')).read_text()) != rec['yaml']: raise ValueError('Staged YAML changed')
            staged = {'batch_native/'+str(p.relative_to(out/'code')):sha(p) for p in (out/'code').rglob('*.py')}
            if staged != {k:v for k,v in sources.items() if k.startswith('batch_native/')}: raise ValueError('Staged code changed')
        else:
            out.mkdir(parents=True)
            (out/'request.json').write_text(json.dumps(request,indent=2))
            (out/'inputs').mkdir(); (out/'tmp').mkdir()
            shutil.copytree(ROOT/'batch_native',out/'code',ignore=shutil.ignore_patterns('__pycache__'))
            for k,(p,digest) in files.items():
                target=out/k; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,target)
                if sha(target) != digest: raise ValueError('Source changed while staging')
            for rec in records: (out/'inputs'/(rec['id']+'.yaml')).write_text(json.dumps(rec['yaml'],indent=2))
        for seed in a.seeds:
            stage=out/f'seed-{seed}'
            if stage.exists():
                # Never overwrite even interrupted native batches. Completed batches are deeply re-read.
                if not (stage/'verified.json').exists(): raise ValueError('Interrupted seed preserved; select a new output directory: '+str(stage))
                with (out/f'verify-{seed}.log').open('a') as log:
                    run_owned(command(a,out,'verify.py',seed),log,a.timeout)
                continue
            stage.mkdir()
            with (stage/'inference.log').open('x') as log:
                run_owned(command(a,out,'observe.py',seed,True),log,a.timeout)
            with (out/f'verify-{seed}.log').open('a') as log:
                run_owned(command(a,out,'verify.py',seed),log,a.timeout)
        print(json.dumps({'status':'PASS','models':len(records)*len(a.seeds)*a.diffusion_samples,'output':str(out)}))


if __name__ == '__main__': main()

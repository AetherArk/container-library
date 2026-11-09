"""Mounted/local fixed-budget sweeps and physical-host aggregation. No cloud SDK.

Run ``python throughput_benchmark.py {sweep,aggregate} --help``.
Completion trusts the resident worker's native verifier, then independently checks
its sealed output bytes and exact requested cell/row protocol. No torch import.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resident.identity import directory_identity  # noqa: E402  (stdlib only, no torch)
from resident.worker import Settings  # noqa: E402  (no torch import)
from dataclasses import asdict


SCHEMA = 'cofold-throughput-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint_digest(path):
    """Hash a single checkpoint file or a model directory with the worker's own scheme."""
    path = Path(path)
    require(path.exists(), 'checkpoint missing: ' + str(path))
    if path.is_dir():
        return directory_identity(path)['sha256']
    return sha(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def rate(count, seconds):
    return count / seconds if number(seconds) and seconds > 0 else None


def metadata(path):
    value = read(path) if path else {}
    require(isinstance(value, dict), 'metadata must be an object')
    allowed = {'provider', 'site_id', 'physical_host_id', 'machine_id', 'instance_id',
               'host_identity_source', 'evidence_kind'}
    require(not (set(value) - allowed), 'unknown metadata keys: ' + str(set(value) - allowed))
    require(all(isinstance(v, str) and 0 < len(v) <= 256 for v in value.values()),
            'metadata values must be nonempty strings <=256 characters')
    require(value.get('evidence_kind', 'measured') in ('measured', 'fixture'), 'invalid evidence_kind')
    value.setdefault('evidence_kind', 'measured')
    if value.get('physical_host_id') or value.get('machine_id'):
        require(value.get('host_identity_source') == 'authoritative_caller_assertion',
                'physical host identity needs host_identity_source=authoritative_caller_assertion')
        require(value.get('site_id') or value.get('provider'), 'host identity needs site_id or provider namespace')
        require(not (value.get('physical_host_id') and value.get('machine_id') and
                     value['physical_host_id'] != value['machine_id']), 'conflicting physical host IDs')
    return value


def panel(input_dir):
    # PyYAML is already part of the worker/host runtime, not a provider dependency.
    import yaml
    root = Path(input_dir).resolve()
    sources = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in ('.yaml', '.yml'))
    require(bool(sources), 'no native YAML inputs')
    require(len({p.stem for p in sources}) == len(sources), 'duplicate native system stems')
    result = []
    for source in sources:
        refs = {}
        try:
            data = yaml.safe_load(source.read_text())
        except yaml.YAMLError:
            data = None  # preserve invalid input for native isolated failure
        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in ('msa', 'cif', 'pdb', 'path') and isinstance(child, str) and child not in ('empty', 'auto'):
                        p = Path(child)
                        target = p if p.is_absolute() else source.parent / p
                        target = Path(os.path.abspath(target))
                        refs[child] = {'sha256': sha(target) if target.is_file() else None,
                                       'source': str(target), 'relative': not p.is_absolute()}
                    else:
                        walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(data)
        result.append({'system': source.name, 'source': str(source), 'sha256': sha(source), 'references': refs})
    return result


def panel_protocol(items):
    return [{k: item[k] for k in ('system', 'sha256')} | {
        'references': {name: ref['sha256'] for name, ref in item['references'].items()}} for item in items]


def stage_panel(items, destination):
    """Mirror only selected files under their common ancestor; never rewrite YAML.

    ../ MSAs remain at the same relative path. Absolute references stay mounted
    in place. Copies (not hardlinks) plus before/after checks bind transport bytes.
    """
    paths = [Path(x['source']).parent for x in items]
    paths += [Path(ref['source']).parent for x in items for ref in x['references'].values() if ref['relative']]
    anchor = Path(os.path.commonpath(paths))
    tree = destination / 'tree'
    active = tree / Path(items[0]['source']).parent.relative_to(anchor)
    active.mkdir(parents=True)
    copies = {}
    for item in items:
        copies[Path(item['source'])] = (item['sha256'], True)
        for ref in item['references'].values():
            copies[Path(ref['source'])] = (ref['sha256'], ref['relative'])
    for source, (expected, relative) in copies.items():
        if expected is None:
            continue
        require(sha(source) == expected, 'input/dependency changed before staging: ' + str(source))
        if relative:
            target = tree / source.relative_to(anchor)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            require(sha(target) == expected, 'staged bytes differ')
    require(sorted(p.name for p in active.iterdir() if p.suffix.lower() in ('.yaml', '.yml')) ==
            sorted(x['system'] for x in items), 'dependency collides with active YAML panel')
    return active


def verify_receipt(output, job):
    relative = Path(job['output'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe attempt path')
    attempt = output / relative
    receipt = read(attempt / 'receipt.json')
    require(receipt.get('status') == 'completed' and receipt.get('identity') == job['identity'], 'receipt identity/status mismatch')
    actual = {str(p.relative_to(attempt)): sha(p) for p in attempt.rglob('*')
              if p.is_file() and p.name != 'receipt.json'}
    require(bool(actual) and actual == receipt.get('sha256'), 'receipt inventory/hash mismatch')
    return attempt


def gpu_identity(benchmark):
    devices = benchmark.get('gpu_initial', {}).get('devices') or []
    def val(v):
        return v.get('value') if isinstance(v, dict) else v
    result = [{'name': val(d.get('name')), 'uuid': val(d.get('uuid'))} for d in devices]
    # Several visible devices do not identify which device the worker used.
    return {'devices': result, 'gpu_class': result[0]['name'] if len(result) == 1 else None,
            'null_reason': None if len(result) == 1 and result[0]['name'] else 'used_gpu_not_unambiguously_identified'}


def inspect_worker(output, items, seeds, settings, checkpoint_hash):
    """Reject cell/protocol mismatches; sum shared batch metrics exactly once."""
    run = read(output / 'run.json')
    bench = read(output / 'benchmark.json')
    require(run['requested_seeds'] == seeds and run['settings'] == settings, 'worker protocol mismatch: seeds/settings')
    require(run['checkpoint']['sha256'] == checkpoint_hash, 'worker checkpoint mismatch')
    require(bench.get('settings') == settings and bench.get('requested_seeds') == seeds,
            'benchmark protocol mismatch')
    require(run.get('checkpoint_load_count') in (0, 1), 'resident checkpoint loaded more than once')
    wanted = Counter((item['system'], seed, occurrence) for item in items
                     for seed, occurrence in seed_occurrences(seeds))
    got = Counter((Path(j['input']).name, j['seed'], j.get('seed_occurrence', 0)) for j in run['jobs'])
    require(got == wanted, 'worker job panel mismatch')
    raw = (output / 'predictions.jsonl').read_text().splitlines(keepends=True)
    require(all(line.endswith('\n') for line in raw), 'uncommitted prediction tail')
    rows = [json.loads(line) for line in raw]
    by_job = defaultdict(list)
    for row in rows:
        require(row.get('invocation_id') == bench['invocation_id'], 'prediction invocation mismatch')
        by_job[row['job_key']].append(row)
    require(set(by_job) == {j['key'] for j in run['jobs']}, 'prediction job keys mismatch')
    records, seen_shapes = [], set()
    mapping = {i['system']: i for i in items}
    for job in run['jobs']:
        system = Path(job['input']).name
        item = mapping[system]
        identity = job['identity']
        require(identity['input']['sha256'] == item['sha256'] and identity['settings'] == settings and
                identity['seed'] == job['seed'] and identity['checkpoint']['sha256'] == checkpoint_hash,
                'job identity protocol mismatch')
        # Resolve original aliases against this job's transport mirror, just as
        # the worker does. Multiple aliases for one dependency count only once;
        # distinct files with equal bytes must still remain distinct identities.
        references = {str((Path(job['input']).parent / alias).resolve()): ref['sha256']
                      for alias, ref in item['references'].items()}
        require(identity['input'].get('references', {}) == references, 'job referenced bytes mismatch')
        sample_rows = by_job[job['key']]
        require(sorted(r['sample_index'] for r in sample_rows) == list(range(settings['samples'])), 'sample row count/index mismatch')
        for row in sample_rows:
            require(row['status'] == job['status'] and row['verified'] == (job['status'] == 'completed') and
                    Path(row['input']).name == system and row['seed'] == job['seed'] and
                    row.get('seed_occurrence', 0) == job.get('seed_occurrence', 0), 'prediction status/identity mismatch')
        metrics = sample_rows[0].get('metrics') or {}
        require(all((r.get('metrics') or {}) == metrics for r in sample_rows), 'batch-shared metrics disagree')
        completed = job['status'] == 'completed'
        shape = None
        if completed:
            require(not job.get('reused'), 'sweep cannot substitute reused cells')
            attempt = verify_receipt(output, job)
            inventory = attempt / 'capture' / 'consumed_batch_inventory.json'
            if inventory.is_file():
                shape = {k: v['shape'] for k, v in read(inventory).items() if isinstance(v, dict) and 'shape' in v}
        shape_key = (system, digest(shape)) if shape else (system, None)
        first = shape_key not in seen_shapes
        if metrics.get('forward_seconds'):
            seen_shapes.add(shape_key)
        forward = metrics.get('forward_seconds')
        forward_seconds = sum(forward) if isinstance(forward, list) and forward and all(number(v) for v in forward) else None
        prediction = metrics.get('predict_step_seconds')
        stages = [s['seconds'] for s in bench.get('stages', []) if s.get('job_key') == job['key']]
        system_wall = sum(stages) if stages and all(number(v) for v in stages) else None
        records.append({'system': system, 'seed': job['seed'], 'seed_occurrence': job.get('seed_occurrence', 0),
                        'job_key': job['key'], 'invocation_id': bench['invocation_id'],
                        'status': job['status'], 'error_kind': job.get('error_kind'), 'error': job.get('error'),
                        'requested_samples': settings['samples'], 'verified_samples': settings['samples'] if completed else 0,
                        'forward_seconds': forward_seconds, 'full_prediction_seconds': prediction if number(prediction) else None,
                        'system_wall_seconds': system_wall, 'first_use': first, 'input_shape': shape,
                        'first_use_basis': 'input_and_observed_shape' if shape else 'input_identity_only_shape_unavailable',
                        'metrics': metrics, 'output': job.get('output')})
    if any(r['verified_samples'] for r in records):
        require(run.get('checkpoint_load_count') == 1, 'successful predictions without one model load')
    return {'status': run['status'], 'records': records, 'gpu': gpu_identity(bench),
            'worker_wall_seconds': bench.get('wall_seconds'), 'implementation': run.get('implementation'),
            'assets': run.get('assets'), 'invocation_id': bench['invocation_id'],
            'checkpoint_load_count': run.get('checkpoint_load_count')}


def seed_occurrences(seeds):
    counts = Counter()
    for seed in seeds:
        yield seed, counts[seed]
        counts[seed] += 1


def summarize(records, wall=None):
    total = sum(r['verified_samples'] for r in records)
    result = {'requested_samples': sum(r['requested_samples'] for r in records), 'verified_samples': total,
              'outcomes': dict(Counter(r['status'] for r in records)), 'complete_wall_seconds': wall,
              'complete_samples_per_second': rate(total, wall)}
    for field in ('forward_seconds', 'full_prediction_seconds', 'system_wall_seconds'):
        # Partial/missing timing is never treated as zero or assigned verified work.
        timed = [r for r in records if number(r.get(field)) and r[field] > 0]
        seconds = sum(r[field] for r in timed)
        count = sum(r['verified_samples'] for r in timed)
        result[field] = seconds if timed else None
        result[field + '_timed_verified_samples'] = count
        result[field + '_samples_per_second'] = rate(count, seconds)
        result[field + '_coverage_complete'] = len(timed) == len(records)
    return result


def worker_command(args, staged, config_root, level, samples, checkpoint):
    """The exact resident-worker invocation for one configuration level."""
    settings = {**asdict(Settings()), 'samples': samples, 'steps': args.steps,
                'recycles': args.recycles, 'max_parallel_samples': level}
    command = [args.worker_executable, '--input', str(staged), '--output', str(config_root / 'worker'),
               '--assets', str(Path(args.assets).resolve()), '--checkpoint', str(checkpoint),
               '--seeds', *map(str, args.seeds), '--samples', str(samples), '--steps', str(args.steps),
               '--recycles', str(args.recycles), '--max-parallel-samples', str(level),
               '--output-format', args.output_format, '--derived', settings['derived']]
    return command


def validate(args):
    """Engine limits checked before any directory is created or worker started."""
    levels = sorted(set(args.levels))
    require(levels and min(levels) >= 1, 'levels must be positive')
    require(levels == [1], 'Protenix folds one diffusion sample per native call, so only '
                           '--levels 1 is measurable; got ' + str(levels))
    require(not args.use_potentials, '--use-potentials has no Protenix equivalent; '
                                     'the capability is not part of this engine')
    require(not args.no_kernels, '--no-kernels is not exposed by this container; the released '
                                 'protenix-v2 checkpoint runs its configured triangle kernels '
                                 '(cuequivariance) only')
    return levels


def sweep(args):
    levels = validate(args)
    samples = args.samples if args.samples is not None else math.lcm(*levels)
    require(samples >= max(levels), 'samples must be at least maximum parallelism; samples=1 cannot exercise max=6')
    require(all(0 <= s <= 4294967295 for s in args.seeds), 'seeds out of range')
    require(args.steps > 0 and args.recycles >= 0, 'invalid steps/recycles')
    items = panel(args.input)
    checkpoint = Path(args.checkpoint).resolve()
    require(Path(args.assets).is_dir(), 'assets directory missing')
    ckhash = checkpoint_digest(checkpoint)
    meta = metadata(args.metadata)
    root = Path(args.output).absolute()
    root.mkdir(parents=True, exist_ok=False)  # a new output scope is mandatory
    settings = {**asdict(Settings()), 'samples': samples, 'steps': args.steps,
                'recycles': args.recycles, 'max_parallel_samples': levels[0]}
    result = {'schema': SCHEMA, 'sweep_id': uuid.uuid4().hex, 'status': 'running', 'metadata': meta,
              'protocol': {'panel': panel_protocol(items), 'seeds': args.seeds, 'settings': settings,
                           'levels': levels, 'checkpoint_sha256': ckhash},
              'tool_sha256': sha(__file__), 'configurations': []}
    write(root / 'sweep.json', result)
    blocked = {}
    try:
        for level in levels:
            config_root = root / ('parallel-' + str(level))
            config_root.mkdir()
            active = [i for i in items if i['system'] not in blocked]
            skipped = [{'system': i['system'], 'seed': seed, 'seed_occurrence': occurrence,
                        'status': 'skipped', 'error_kind': 'cuda_out_of_memory',
                        'error': 'larger level skipped after observed worker OOM at ' + str(blocked[i['system']]),
                        'requested_samples': samples, 'verified_samples': 0}
                       for i in items if i['system'] in blocked for seed, occurrence in seed_occurrences(args.seeds)]
            config = {'max_parallel_samples': level, 'status': 'running', 'records': skipped,
                      'worker_output': str(config_root / 'worker'), 'gpu': None}
            result['configurations'].append(config)
            write(root / 'sweep.json', result)
            if not active:
                config.update(status='skipped', wall_seconds=0.0, summary=summarize(skipped, 0.0))
                config['systems'] = {i['system']: {'all': summarize([r for r in skipped if r['system'] == i['system']]),
                    'first_use': summarize([]), 'warm': summarize([])} for i in items}
                write(config_root / 'failure_receipts.json', skipped)
                write(root / 'sweep.json', result)
                continue
            staged = stage_panel(active, config_root)
            full_settings = settings | {'max_parallel_samples': level}
            command = worker_command(args, staged, config_root, level, samples, checkpoint)
            config['command'] = command
            start = time.perf_counter()
            try:
                with (config_root / 'worker.log').open('xb') as log:
                    proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                config['returncode'] = proc.returncode
                inspected = inspect_worker(config_root / 'worker', active, args.seeds, full_settings, ckhash)
                config.update({k: v for k, v in inspected.items() if k != 'records'})
                config['records'].extend(inspected['records'])
                require(panel_protocol(panel(staged)) == panel_protocol(active), 'staged input bytes changed during execution')
                require(panel_protocol(panel(args.input)) == result['protocol']['panel'], 'source input bytes changed during execution')
                if proc.returncode != 0 and config['status'] == 'completed':
                    raise ValueError('worker exit/status mismatch')
                for record in inspected['records']:
                    if record.get('error_kind') == 'cuda_out_of_memory' and record['status'] == 'failed':
                        blocked[record['system']] = level
            except (OSError, ValueError, KeyError, TypeError) as exc:
                # Protocol errors are not OOM; do not blacklist unrelated systems.
                config.update(status='protocol_error', error=str(exc))
                config['records'] = skipped + [{'system': i['system'], 'seed': seed,
                    'seed_occurrence': occurrence, 'status': 'pending', 'error_kind': 'worker_protocol_error',
                    'error': str(exc), 'requested_samples': samples, 'verified_samples': 0}
                    for i in active for seed, occurrence in seed_occurrences(args.seeds)]
            config['wall_seconds'] = time.perf_counter() - start
            config['summary'] = summarize(config['records'], config['wall_seconds'])
            config['systems'] = {i['system']: {
                'all': summarize([r for r in config['records'] if r['system'] == i['system']]),
                'first_use': summarize([r for r in config['records'] if r['system'] == i['system'] and r.get('first_use') is True]),
                'warm': summarize([r for r in config['records'] if r['system'] == i['system'] and r.get('first_use') is False])}
                for i in items}
            write(config_root / 'failure_receipts.json', [r for r in config['records'] if r['status'] != 'completed'])
            write(root / 'sweep.json', result)
        result['status'] = 'completed' if all(c['status'] == 'completed' for c in result['configurations']) else 'incomplete'
    finally:
        if result['status'] == 'running':
            result['status'] = 'interrupted'
        write(root / 'sweep.json', result)
    return result


def host_key(meta):
    physical = meta.get('physical_host_id') or meta.get('machine_id')
    if physical and meta.get('host_identity_source') == 'authoritative_caller_assertion':
        namespace = meta.get('site_id') or meta.get('provider')
        if namespace:
            return str(namespace) + ':' + str(physical)
    return None


def aggregate(paths, required_hosts=5):
    require(required_hosts >= 5, 'required host coverage cannot be below five')
    attempts = []
    seen = set()
    expected = None
    measured_identities = set()
    for path in paths:
        data = read(path)
        require(data.get('schema') == SCHEMA, 'unknown sweep schema')
        require(data['sweep_id'] not in seen, 'same sweep supplied twice')
        seen.add(data['sweep_id'])
        protocol = data['protocol']
        require(expected is None or expected == protocol, 'protocol mismatch: input/seeds/samples/settings/levels/checkpoint')
        expected = protocol
        meta = data['metadata']
        configs = {c['max_parallel_samples']: c for c in data['configurations']}
        require(len(configs) == len(data['configurations']), 'duplicate configuration levels')
        for config in configs.values():
            qualified = config.get('implementation') is not None and config.get('assets') is not None
            if meta.get('evidence_kind', 'measured') == 'measured':
                require(qualified or not any(r['status'] in ('completed', 'failed')
                                            for r in config.get('records', [])),
                        'unqualified measured software: implementation/assets missing')
        identities = {(digest(c['implementation']), digest(c['assets'])) for c in configs.values()
                      if c.get('implementation') is not None and c.get('assets') is not None}
        require(len(identities) <= 1, 'implementation/assets changed within sweep')
        if meta.get('evidence_kind', 'measured') == 'measured':
            measured_identities.update(identities)
            require(len(measured_identities) <= 1,
                    'implementation/assets mismatch across measured GPU comparisons')
        # Retain every planned level, including interrupted/not-yet-started levels.
        observed_gpus = {c['gpu']['gpu_class'] for c in configs.values() if c.get('gpu') and c['gpu'].get('gpu_class')}
        require(len(observed_gpus) <= 1, 'actual GPU class changed within sweep')
        gpu = next(iter(observed_gpus), None)
        for level in protocol['levels']:
            config = configs.get(level, {})
            rows = config.get('records', [])
            samples = protocol['settings']['samples']
            expected_cells = Counter((item['system'], seed, occurrence) for item in protocol['panel']
                                     for seed, occurrence in seed_occurrences(protocol['seeds']))
            expected_samples = sum(expected_cells.values()) * samples
            if not rows:
                rows = [{'system': item['system'], 'seed': seed, 'seed_occurrence': occurrence,
                         'status': 'pending', 'error_kind': 'configuration_not_finished',
                         'requested_samples': samples, 'verified_samples': 0}
                        for item in protocol['panel'] for seed, occurrence in seed_occurrences(protocol['seeds'])]
            require(Counter((r['system'], r['seed'], r['seed_occurrence']) for r in rows) == expected_cells,
                    'aggregate panel job identity/count mismatch')
            require(all(r['requested_samples'] == samples and
                        r['verified_samples'] == (samples if r['status'] == 'completed' else 0) for r in rows),
                    'aggregate verified/requested sample count mismatch')
            summary = summarize(rows, config.get('wall_seconds'))
            summary['requested_samples'] = expected_samples
            attempts.append({'source': str(Path(path).absolute()), 'sweep_id': data['sweep_id'],
                'level': level, 'gpu_class': gpu, 'gpu': config.get('gpu'), 'host': host_key(meta),
                'host_identity_source': meta.get('host_identity_source'), 'metadata': meta,
                'evidence_kind': meta.get('evidence_kind', 'measured'), 'status': config.get('status', 'pending'),
                'summary': summary, 'records': rows,
                'implementation_assets': [(digest(config['implementation']), digest(config['assets']))]
                    if config.get('implementation') is not None and config.get('assets') is not None else []})
    groups = []
    for gpu, level, kind in sorted({(a['gpu_class'] or 'unknown', a['level'], a['evidence_kind']) for a in attempts}):
        selected = [a for a in attempts if (a['gpu_class'] or 'unknown', a['level'], a['evidence_kind']) == (gpu, level, kind)]
        identities = {tuple(tuple(x) for x in a['implementation_assets']) for a in selected if a['implementation_assets']}
        require(len(identities) <= 1, 'implementation/assets mismatch between attempts')
        by_host = defaultdict(list)
        for i, a in enumerate(selected):
            by_host[a['host'] or ('unidentified-attempt-' + str(i))].append(a)
        metric_names = ['complete_samples_per_second', 'forward_seconds_samples_per_second',
                        'full_prediction_seconds_samples_per_second', 'system_wall_seconds_samples_per_second']
        hosts = []
        for host, repeats in by_host.items():
            # Missing timing stays missing; all-OOM with measured wall time is 0/s.
            means = {}
            for metric in metric_names:
                rates = [a['summary'][metric] if metric == 'complete_samples_per_second' or
                         a['summary'].get(metric.removesuffix('_samples_per_second') + '_coverage_complete')
                         else None for a in repeats]
                means[metric] = statistics.mean(v for v in rates if v is not None) if all(v is not None for v in rates) else None
            first_warm = {}
            for label, flag in (('first_use', True), ('warm', False)):
                summaries = [summarize([r for r in a['records'] if r.get('first_use') is flag]) for a in repeats]
                first_warm[label] = {metric: statistics.mean([s[metric] for s in summaries])
                    if all(s[metric] is not None and
                           s[metric.removesuffix('_samples_per_second') + '_coverage_complete']
                           for s in summaries) else None for metric in metric_names[1:]}
            hosts.append({'host': repeats[0]['host'], 'attempts': len(repeats), **means,
                          'first_use_and_warm': first_warm,
                          'requested_samples': sum(a['summary']['requested_samples'] for a in repeats),
                          'verified_samples': sum(a['summary']['verified_samples'] for a in repeats),
                          'statuses': dict(Counter(a['status'] for a in repeats))})
        distinct = len({a['host'] for a in selected if a['host'] and a['implementation_assets']}) if gpu != 'unknown' else 0
        unknown_hosts = any(a['host'] is None or not a['implementation_assets'] for a in selected)
        between = {}
        for metric in metric_names:
            values = [h[metric] for h in hosts]
            between['equal_host_' + metric] = statistics.mean(values) if values and not unknown_hosts and all(v is not None for v in values) else None
        groups.append({'gpu_class': gpu, 'level': level, 'evidence_kind': kind,
            'distinct_physical_hosts': distinct, 'required_physical_hosts': required_hosts,
            'coverage_complete': distinct >= required_hosts and kind != 'fixture',
            'coverage_note': 'fixtures are not cloud evidence' if kind == 'fixture' else 'caller-asserted physical IDs; instance IDs do not count',
            'hosts': hosts, 'host_weighting': 'within-host mean, then equal-weight between-host mean',
            'equal_host_first_use_and_warm': {label: {metric:
                statistics.mean([h['first_use_and_warm'][label][metric] for h in hosts])
                if hosts and not unknown_hosts and all(h['first_use_and_warm'][label][metric] is not None for h in hosts)
                else None for metric in metric_names[1:]} for label in ('first_use', 'warm')},
            **between,
            'attempts': len(selected), 'unidentified_attempts': sum(a['host'] is None for a in selected),
            'requested_samples': sum(a['summary']['requested_samples'] for a in selected),
            'verified_samples': sum(a['summary']['verified_samples'] for a in selected)})
    return {'schema': SCHEMA, 'protocol': expected, 'required_physical_hosts': required_hosts,
            'groups': groups, 'attempts': attempts, 'coverage_complete': bool(groups) and all(g['coverage_complete'] for g in groups)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('sweep', help='execute one resident worker invocation per ascending configuration')
    for name in ('input', 'output', 'assets', 'checkpoint'):
        run.add_argument('--' + name, required=True, type=Path)
    run.add_argument('--seeds', nargs='+', type=int, required=True)
    run.add_argument('--levels', nargs='+', type=int, default=[1])
    run.add_argument('--samples', type=int)
    run.add_argument('--steps', type=int, default=200)
    run.add_argument('--recycles', type=int, default=10)
    run.add_argument('--no-kernels', action='store_true')
    run.add_argument('--use-potentials', action='store_true')
    run.add_argument('--output-format', choices=('mmcif',), default='mmcif')
    run.add_argument('--worker-executable', default='cofold-run', help='public executable path; test fixtures must use evidence_kind=fixture')
    run.add_argument('--metadata', type=Path, help='optional provider-neutral nonsecret JSON; no provider discovery')
    agg = sub.add_parser('aggregate', help='aggregate sweep.json files; preserve failed/missing attempts')
    agg.add_argument('sweeps', nargs='+', type=Path)
    agg.add_argument('--required-hosts', type=int, default=5)
    agg.add_argument('--output', type=Path, required=True, help='new report file, never overwritten')
    args = parser.parse_args(argv)
    try:
        if args.command == 'sweep':
            validate(args)
            result = sweep(args)
            print(json.dumps({'status': result['status'], 'output': str(args.output / 'sweep.json')}))
            return 0 if result['status'] == 'completed' else 1
        require(not args.output.exists(), 'aggregate output already exists')
        result = aggregate(args.sweeps, args.required_hosts)
        # Exclusive publication, unlike sweep's owned live snapshot.
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'coverage_complete': result['coverage_complete'], 'output': str(args.output)}))
        return 0 if result['coverage_complete'] else 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())

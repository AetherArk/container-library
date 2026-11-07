"""Provider-independent observations; never changes scientific settings or RNG state."""
import contextlib
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import threading
import time
import uuid

SCHEMA = 1
PROVIDER_KEYS = ('SALAD_ORGANIZATION_NAME', 'SALAD_ORGANIZATION_ID',
                 'SALAD_PROJECT_NAME', 'SALAD_PROJECT_ID',
                 'SALAD_CONTAINER_GROUP_NAME', 'SALAD_CONTAINER_GROUP_ID',
                 'SALAD_MACHINE_ID', 'SALAD_INSTANCE_ID')
METADATA_KEYS = set(PROVIDER_KEYS) | {'provider', 'region', 'zone', 'machine_type', 'gpu_class', 'node_id', 'price_per_hour', 'currency'}


def sanitize_metadata(value):
    """Small, explicit nonsecret identifier/price allowlist, not an environment dump."""
    if not isinstance(value, dict):
        raise ValueError('benchmark metadata must be a JSON object')
    result = {}
    for key, item in value.items():
        if key not in METADATA_KEYS:
            continue
        if key == 'price_per_hour':
            if type(item) in (int, float) and math.isfinite(item) and item >= 0:
                result[key] = item
        elif isinstance(item, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:/ -]{0,127}', item) and '://' not in item:
            result[key] = item
    return result


def load_metadata(path=None):
    if path is None:
        return {}
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(16385)
    if len(raw) > 16384:
        raise ValueError('benchmark metadata too large')
    return sanitize_metadata(json.loads(raw))


def error_label(exc):
    # Exception messages can contain capabilities, passwords or input URL values.
    return type(exc).__name__ + ': details suppressed; see phase and local inputs'


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def observation(value, reason='unavailable'):
    return {'value': value, 'null_reason': reason if value is None else None}


def hardware():
    versions = {}
    for package in ('torch', 'boltz', 'numpy', 'pytorch-lightning'):
        try:
            versions[package] = observation(importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            versions[package] = observation(None, 'package_not_installed')
    meminfo = read_text('/proc/meminfo') or ''
    match = re.search(r'^MemTotal:\s+(\d+) kB', meminfo, re.M)
    cpu = read_text('/proc/cpuinfo') or ''
    model = re.search(r'^model name\s*:\s*(.+)', cpu, re.M)
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (OSError, AttributeError):
        affinity = None
    # cgroup v2 path in a namespaced container, or the process-relative host path.
    group = read_text('/proc/self/cgroup') or ''
    relative = next((line[3:] for line in group.splitlines() if line.startswith('0::')), '/')
    roots = [Path('/sys/fs/cgroup') / relative.lstrip('/'), Path('/sys/fs/cgroup')]
    allocation = {}
    for field in ('cpu.max', 'cpuset.cpus.effective', 'memory.max', 'memory.high', 'memory.current', 'memory.peak', 'memory.swap.max'):
        value = next((v for root in roots if (v := read_text(root / field)) is not None), None)
        allocation[field] = observation(value, 'cgroup_v2_field_unavailable')
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions,
            'cpu_model': observation(model.group(1) if model else None),
            'cpu_count_host': observation(os.cpu_count()), 'cpu_affinity': observation(affinity),
            'ram_host_bytes': observation(int(match.group(1)) * 1024 if match else None),
            'cgroup': allocation}


GPU_FIELDS = ('uuid', 'name', 'driver_version', 'memory.total', 'utilization.gpu', 'memory.used',
              'temperature.gpu', 'clocks.sm', 'clocks.mem', 'power.draw', 'power.limit')


def gpu_sample(executable):
    if not executable:
        return {'devices': None, 'null_reason': 'nvidia_smi_missing'}
    try:
        proc = subprocess.run([executable, '--query-gpu=' + ','.join(GPU_FIELDS), '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5, check=True)
        devices = []
        for row in csv.reader(proc.stdout.splitlines()):
            if len(row) != len(GPU_FIELDS):
                raise ValueError('invalid telemetry columns')
            device = {}
            for key, raw in zip(GPU_FIELDS, row):
                raw = raw.strip()
                if raw in ('N/A', '[N/A]', '[Not Supported]', 'Not Supported', ''):
                    device[key] = observation(None, 'unsupported_or_unavailable')
                elif key in GPU_FIELDS[:3]:
                    device[key] = observation(raw)
                else:
                    try:
                        number = float(raw)
                        device[key] = observation(number if math.isfinite(number) else None, 'nonfinite')
                    except ValueError:
                        device[key] = observation(None, 'unsupported_or_unavailable')
            devices.append(device)
        return {'devices': devices, 'null_reason': None if devices else 'no_visible_devices'}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {'devices': None, 'null_reason': 'nvidia_smi_query_failed'}


class Benchmark:
    def __init__(self, output, metadata=None, interval=2.0, sample_gpu=True):
        from .worker import atomic_json
        self.atomic_json = atomic_json
        self.output = Path(output)
        self.id = uuid.uuid4().hex
        self.start = time.perf_counter()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.interval = interval
        self.summary = {'schema': SCHEMA, 'invocation_id': self.id, 'status': 'running',
                        'started_at_unix': time.time(), 'hardware': hardware(),
                        'caller_metadata': sanitize_metadata(metadata or {}), 'stages': [],
                        'gpu_sample_interval_seconds': interval, 'gpu_sample_count': 0,
                        'timer_clock': 'perf_counter; elapsed wall seconds', 'gpu_units': {
                            'memory.total': 'MiB', 'memory.used': 'MiB', 'utilization.gpu': 'percent',
                            'temperature.gpu': 'C', 'clocks.sm': 'MHz', 'clocks.mem': 'MHz',
                            'power.draw': 'W', 'power.limit': 'W'}}
        self.smi = shutil.which('nvidia-smi') if sample_gpu else None
        self.sample_gpu = sample_gpu
        self.thread = None
        self.checked_tails = set()

    def preserve_partial_tail(self, path):
        """Called under the append lock (and worker ownership lock on resume).

        A newline is the commit boundary. Preserve any unterminated bytes as
        evidence before truncating, never interpret them as verified records.
        Scan backwards in bounded blocks; copy even a large fragment streaming.
        """
        if not path.exists():
            return
        with path.open('r+b') as stream:
            end = stream.seek(0, os.SEEK_END)
            boundary = end
            while boundary:
                begin = max(0, boundary - 65536)
                stream.seek(begin)
                block = stream.read(boundary - begin)
                newline = block.rfind(b'\n')
                if newline >= 0:
                    boundary = begin + newline + 1
                    break
                boundary = begin
            if boundary == end:
                return
            evidence = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
            # Persist evidence first: interruption can duplicate, never erase it.
            with evidence.open('xb') as fragment:
                evidence.chmod(0o600)
                stream.seek(boundary)
                shutil.copyfileobj(stream, fragment, 65536)
                fragment.flush()
                os.fsync(fragment.fileno())
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            stream.truncate(boundary)
            stream.flush()
            os.fsync(stream.fileno())

    def append(self, filename, record, stdout=False):
        record = dict(record, schema=SCHEMA, invocation_id=self.id)
        line = json.dumps(record, allow_nan=False, separators=(',', ':'))
        with self.lock:
            path = self.output / filename
            if filename not in self.checked_tails:
                self.preserve_partial_tail(path)
                self.checked_tails.add(filename)
            with path.open('a') as stream:
                stream.write(line + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            if stdout:
                print(line, flush=True)

    def event(self, event, **data):
        self.append('events.jsonl', dict(data, event=event, timestamp_unix=time.time(),
                                      elapsed_seconds=time.perf_counter() - self.start), stdout=True)

    @contextlib.contextmanager
    def stage(self, name, **identity):
        begin = time.perf_counter()
        self.event('stage_started', stage=name, **identity)
        status = 'completed'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            row = dict(identity, stage=name, started_elapsed_seconds=begin - self.start,
                       seconds=time.perf_counter() - begin, status=status)
            self.summary['stages'].append(row)
            self.event('stage_finished', **row)

    def sample(self):
        sample = gpu_sample(self.smi) if self.sample_gpu else {'devices': None, 'null_reason': 'not_sampled_in_this_process'}
        if self.summary['gpu_sample_count'] == 0:
            self.summary['gpu_initial'] = sample
        self.summary['gpu_sample_count'] += 1
        self.event('gpu_sample', **sample)

    def __enter__(self):
        self.event('invocation_started')
        self.sample()
        def monitor():
            while not self.stop.wait(self.interval):
                try:
                    self.sample()
                except Exception:
                    # Monitoring loss must not terminate native inference.
                    self.summary['telemetry_error'] = 'sampling_failed'
        if self.smi:
            self.thread = threading.Thread(target=monitor, daemon=True)
            self.thread.start()
        self.atomic_json(self.output / 'benchmark.json', self.summary)
        return self

    def __exit__(self, typ, value, tb):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=6)
        if typ:
            self.summary['status'] = 'fatal'
            self.summary['error'] = error_label(value)
        totals, counts = {}, {}
        for stage in self.summary['stages']:
            name = stage['stage']
            totals[name] = totals.get(name, 0.0) + stage['seconds']
            counts[name] = counts.get(name, 0) + 1
        self.summary['stage_totals_seconds'] = totals
        self.summary['stage_observation_counts'] = counts
        self.summary['wall_seconds'] = time.perf_counter() - self.start
        self.event('invocation_finished', status=self.summary['status'])
        self.atomic_json(self.output / 'benchmark.json', self.summary)


def timed_forward(call, synchronize, metrics):
    """Wrap the original forward callable, INSIDE module post-hooks; no capture I/O."""
    def forward(*args, **kwargs):
        synchronize()
        begin = time.perf_counter()
        try:
            return call(*args, **kwargs)
        finally:
            synchronize()
            metrics['forward_seconds'].append(time.perf_counter() - begin)
    return forward

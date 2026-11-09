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
import traceback
import uuid
from .resources import Resources, update_peaks

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


# Redaction governs credentials, not the diagnosis: the type, message, phase and
# traceback of a failure must survive in failure.json. Ordered: URLs first, then
# key=value pairs, then opaque blobs. Absolute paths are left as written.
REDACT_PATTERNS = (
    # Keep scheme://host/path and drop the query string, which carries signatures.
    (re.compile(r'(?i)\b([a-z][a-z0-9+.\-]*://[^\s?#]+)\?[^\s#]*'), r'\1'),
    (re.compile(r'(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=\-]{8,}'), r'\1 [REDACTED]'),
    (re.compile(r'(?i)([A-Za-z0-9_.\-]*(?:TOKEN|SECRET|PASSWORD|KEY|CREDENTIAL)[A-Za-z0-9_.\-]*\s*=\s*)'
                r'(?:"[^"]*"|\'[^\']*\'|[^\s,;]+)'), r'\1[REDACTED]'),
    (re.compile(r'\b[0-9a-fA-F]{32,}\b'), '[HEX_REDACTED]'),
    # No '/', so absolute paths in a traceback survive. Require both cases, so an
    # uppercase-only protein or nucleotide sequence in a diagnosis is left alone;
    # a mixed-case blob is the base64/secret shape. An uppercase-only secret survives.
    (re.compile(r'(?<![A-Za-z0-9+=])(?=[A-Za-z0-9+=]*[a-z])(?=[A-Za-z0-9+=]*[A-Z])'
                r'[A-Za-z0-9+=]{32,}(?![A-Za-z0-9+=])'), '[BLOB_REDACTED]'),
)


def redact(text):
    for pattern, replacement in REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def write_failure(attempt, job, exc):
    """Write <attempt>/failure.json atomically and reference it from the receipt.

    Call this inside the exception handler, before any release_exception(), so the
    traceback still exists. The receipt (written by the worker) then carries
    job['failure'] = 'failure.json', so receipt -> failure.json is followable.
    """
    from .worker import atomic_json
    attempt = Path(attempt)
    payload = {'schema': SCHEMA,
               'error_kind': job.get('error_kind') or 'native_error',
               'phase': job.get('phase'),
               'exception_type': type(exc).__name__,
               'message': redact(str(exc))[:4096],
               'traceback_tail': redact(traceback.format_exc())[-2000:],
               'attempt_path': str(attempt)}
    atomic_json(attempt / 'failure.json', payload)
    job['failure'] = 'failure.json'
    return payload


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def observation(value, reason='unavailable'):
    return {'value': value, 'null_reason': reason if value is None else None}


def hardware():
    versions = {}
    for package in ('torch', 'protenix', 'numpy', 'rdkit', 'biotite'):
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


def resource_association(context):
    """Keep CPU coverage and associated GPU-query coverage independent."""
    return dict(context, sample_count=0, sampled_maxima={}, null_reason='no_valid_samples',
                gpu_sample_count=0, gpu_valid_sample_count=0,
                sampled_gpu_memory_used_max_mib={},
                sampled_gpu_memory_used_max_null_reason='no_associated_samples')


class Benchmark:
    def __init__(self, output, metadata=None, interval=2.0, sample_gpu=True):
        from .worker import atomic_json
        self.atomic_json = atomic_json
        self.output = Path(output)
        self.id = uuid.uuid4().hex
        self.start = time.perf_counter()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        interval = interval if isinstance(interval, (int, float)) and math.isfinite(interval) and interval >= 0.01 else 2.0
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
        self.resource_lock = threading.Lock()
        self.context_lock = threading.Lock()
        self.active_stages = []
        self.finalized = False
        try:
            self.resources = Resources()
        except Exception:
            self.resources = None
        self.summary.update(resource_sample_interval_seconds=interval,
                            resource_sample_count=0, sampled_resource_maxima={},
                            sampled_gpu_memory_used_max_mib={}, resource_associations={},
                            resource_scope='process_tree_and_observable_cgroup; see RESOURCES.md')
        self.summary['resource_units'] = {
            'rss_bytes': 'bytes; summed live process RSS, shared pages may repeat',
            'cpu_seconds_observed': 'seconds; accumulated observed own-process CPU deltas, not waited-child CPU',
            'cores_used': 'CPU seconds delta / monotonic wall interval seconds; not affinity-normalized percent',
            'io_observed_bytes': 'bytes; accumulated proc IO deltas, waited-child IO can overlap',
            'cgroup_cpu_seconds': 'seconds since cgroup creation; separate from process-tree counters',
            'cgroup_fields': 'kernel text: memory bytes, cpu.max quota/period microseconds, cpuacct.usage nanoseconds, cpu.stat microseconds, io.stat bytes/operations',
            'system_memory': 'host bytes; not container allocation',
            'maxima': 'sampled, not true allocator peaks'}
        self.summary['sampled_gpu_memory_used_max_null_reason'] = 'no_valid_samples'

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
            if self.finalized and record.get('event') in ('resource_sample', 'gpu_sample'):
                return
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
        context = dict(identity, stage=name, stage_observation_id=uuid.uuid4().hex)
        self.sample_resources()
        with self.context_lock:
            self.active_stages.append(context)
            self.summary['resource_associations'][context['stage_observation_id']] = resource_association(context)
        self.event('stage_started', **context)
        self.sample_resources()
        begin = time.perf_counter()
        status = 'completed'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            seconds = time.perf_counter() - begin
            self.sample_resources()
            with self.context_lock:
                self.active_stages.remove(context)
            row = dict(context, started_elapsed_seconds=begin - self.start,
                       seconds=seconds, status=status)
            self.summary['stages'].append(row)
            self.event('stage_finished', **row)

    def associations(self):
        with self.context_lock:
            return [dict(row) for row in self.active_stages]

    def sample_resources(self):
        if self.finalized or not self.resource_lock.acquire(blocking=False):
            return
        try:
            contexts = self.associations()
            sample: dict = self.resources.sample() if self.resources else {'null_reason': 'resource_sampler_unavailable'}
            if self.finalized:
                return
            self.summary['resource_sample_count'] += 1
            if 'process_tree' in sample:
                update_peaks(self.summary['sampled_resource_maxima'], sample)
                self.summary['resource_last'] = sample
                for context in contexts:
                    target = self.summary['resource_associations'].setdefault(context['stage_observation_id'],
                        resource_association(context))
                    target['sample_count'] += 1
                    target['null_reason'] = None
                    update_peaks(target['sampled_maxima'], sample)
                    target['last_process_tree_counters'] = sample['process_tree']
                    cpu = sample['process_tree']['cpu_seconds_delta']['value']
                    if cpu is not None:
                        target['cpu_seconds_observed_interval_sum'] = target.get('cpu_seconds_observed_interval_sum', 0.0) + cpu
                    io = target.setdefault('io_observed_interval_sum_bytes', {})
                    for key, observed in sample['process_tree']['io_delta_bytes'].items():
                        if observed['value'] is not None:
                            io[key] = io.get(key, 0) + observed['value']
            self.event('resource_sample', associations=contexts, **sample)
        except Exception:
            self.summary['resource_telemetry_error'] = 'sampling_failed'
        finally:
            self.resource_lock.release()

    def sample(self):
        self.sample_resources()
        contexts = self.associations()
        sample = gpu_sample(self.smi) if self.sample_gpu else {'devices': None, 'null_reason': 'not_sampled_in_this_process'}
        if self.finalized:
            return
        if self.summary['gpu_sample_count'] == 0:
            self.summary['gpu_initial'] = sample
        self.summary['gpu_sample_count'] += 1
        valid_memory = any(device['memory.used']['value'] is not None
                           for device in sample.get('devices') or [])
        for context in contexts:
            target = self.summary['resource_associations'].setdefault(context['stage_observation_id'],
                resource_association(context))
            target['gpu_sample_count'] += 1
            if valid_memory:
                target['gpu_valid_sample_count'] += 1
                target['sampled_gpu_memory_used_max_null_reason'] = None
            elif target['gpu_valid_sample_count'] == 0:
                target['sampled_gpu_memory_used_max_null_reason'] = sample.get('null_reason') or 'no_valid_samples'
        for index, device in enumerate(sample.get('devices') or []):
            key = device['uuid']['value'] or 'visible_index_' + str(index)
            value = device['memory.used']
            maxima = self.summary['sampled_gpu_memory_used_max_mib']
            old = maxima.get(key, observation(None, 'no_valid_samples'))
            if value['value'] is not None and (old['value'] is None or value['value'] > old['value']):
                old = value
                self.summary['sampled_gpu_memory_used_max_null_reason'] = None
            maxima[key] = old
            for context in contexts:
                target = self.summary['resource_associations'].setdefault(context['stage_observation_id'],
                    resource_association(context))
                gpu_max = target.setdefault('sampled_gpu_memory_used_max_mib', {})
                prior = gpu_max.get(key, observation(None, 'no_valid_samples'))
                if value['value'] is not None and (prior['value'] is None or value['value'] > prior['value']):
                    prior = value
                gpu_max[key] = prior
        self.event('gpu_sample', associations=contexts, **sample)

    def __enter__(self):
        self.event('invocation_started')
        try:
            self.sample()
        except Exception:
            self.summary['telemetry_error'] = 'sampling_failed'
        def monitor():
            while not self.stop.wait(self.interval):
                try:
                    self.sample()
                except Exception:
                    # Monitoring loss must not terminate native inference.
                    self.summary['telemetry_error'] = 'sampling_failed'
        self.atomic_json(self.output / 'benchmark.json', self.summary)
        if self.resources or self.smi:
            self.thread = threading.Thread(target=monitor, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, typ, value, tb):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=6)
        self.sample_resources()
        self.finalized = True
        if self.thread and self.thread.is_alive():
            self.summary['telemetry_error'] = 'sampler_shutdown_timeout; late observations discarded'
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

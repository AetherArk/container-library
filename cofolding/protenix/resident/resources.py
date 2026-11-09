"""Bounded Linux resource observations using only the standard library.

Never uses child-inclusive /proc CPU counters: adding cutime/cstime to live
children would count their CPU twice. Unobserved short-lived children are lost.
"""
import os
from pathlib import Path
import re
import time


def observation(value, reason='unavailable'):
    return {'value': value, 'null_reason': reason if value is None else None}


def read(path, limit=65536):
    with Path(path).open() as stream:
        text = stream.read(limit + 1)
    if len(text) > limit:
        raise ValueError('observation_too_large')
    return text.strip()


def stat(text):
    # comm may contain spaces and parentheses. Fields below start at field 3.
    fields = text[text.rindex(')') + 2:].split()
    return {'ppid': int(fields[1]), 'ticks': int(fields[11]) + int(fields[12]),
            'start': int(fields[19]), 'rss_pages': int(fields[21])}


def pairs(text):
    return {key.rstrip(':'): int(value.split()[0])
            for line in text.splitlines() if len(parts := line.split(None, 1)) == 2
            for key, value in [parts]}


def unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)


class Resources:
    """One sampler per invocation; no provider metadata or required dependencies."""
    def __init__(self, proc: str | os.PathLike = '/proc', pid=None, clock=time.monotonic,
                 max_processes=4096, budget_seconds=0.15, max_history=16384):
        self.proc = Path(proc)
        self.pid = os.getpid() if pid is None else pid
        self.clock = clock
        self.max_processes = max_processes
        self.budget = budget_seconds
        self.max_history = max_history
        self.hz = os.sysconf('SC_CLK_TCK')
        self.page = os.sysconf('SC_PAGE_SIZE')
        self.started = clock()
        self.previous_time = None
        self.history = {}
        self.totals = {'cpu_seconds': 0.0, 'read_bytes': 0, 'write_bytes': 0,
                       'rchar': 0, 'wchar': 0}
        self.cgroup_previous = {}
        self.root_identity = None
        try:
            self.boot_ticks = float(read(self.proc / 'uptime').split()[0]) * self.hz
        except (OSError, ValueError, IndexError):
            self.boot_ticks = None
        self.groups, self.discovery_reason = self.discover()

    def discover(self):
        """Resolve membership against actual mount roots, including v1 splits."""
        groups = []
        try:
            memberships = []
            for line in read(self.proc / str(self.pid) / 'cgroup').splitlines():
                _, controllers, path = line.split(':', 2)
                memberships.append((set(controllers.split(',')) if controllers else set(), path))
            for line in read(self.proc / str(self.pid) / 'mountinfo').splitlines():
                left, right = line.split(' - ', 1)
                fields, tail = left.split(), right.split()
                if tail[0] not in ('cgroup', 'cgroup2'):
                    continue
                root, mount = unescape(fields[3]), Path(unescape(fields[4]))
                controllers = set(tail[2].split(','))
                for member_controllers, member in memberships:
                    if (tail[0] == 'cgroup2') != (not member_controllers):
                        continue
                    if member_controllers and not member_controllers.intersection(controllers):
                        continue
                    # A membership outside the mounted subtree is not observable.
                    if member == root:
                        relative = ''
                    elif member.startswith(root.rstrip('/') + '/'):
                        relative = member[len(root.rstrip('/')):].lstrip('/')
                    else:
                        continue
                    if '..' in Path(relative).parts:
                        continue
                    path = mount / relative
                    groups.append({'version': 2 if tail[0] == 'cgroup2' else 1,
                                   'controllers': sorted(member_controllers),
                                   'path': str(path), 'mount_root': root,
                                   'membership': member})
                    if len(groups) >= 32:
                        return groups, 'cgroup_mount_limit'
            return groups, None if groups else 'no_resolvable_cgroup_mount'
        except (OSError, ValueError, IndexError):
            return [], 'cgroup_discovery_unavailable_or_malformed'

    def cgroups(self, now, deadline):
        rows = []
        for group in self.groups:
            row = dict(group, fields={})
            names = (('cpu.stat', 'cpu.max', 'cpuset.cpus.effective', 'memory.current',
                      'memory.peak', 'memory.max', 'memory.high', 'io.stat')
                     if group['version'] == 2 else
                     ('cpuacct.usage', 'cpu.cfs_quota_us', 'cpu.cfs_period_us',
                      'cpuset.cpus', 'memory.usage_in_bytes', 'memory.max_usage_in_bytes',
                      'memory.limit_in_bytes', 'blkio.throttle.io_service_bytes'))
            for name in names:
                if self.clock() > deadline:
                    row['fields'][name] = observation(None, 'collection_time_budget_exceeded')
                    continue
                try:
                    row['fields'][name] = observation(read(Path(group['path']) / name))
                except (OSError, ValueError):
                    row['fields'][name] = observation(None, 'cgroup_field_unavailable')
            try:
                fields = row['fields']
                cpu = (pairs(fields['cpu.stat']['value'])['usage_usec'] / 1e6
                       if group['version'] == 2 else int(fields['cpuacct.usage']['value']) / 1e9)
                previous = self.cgroup_previous.get(group['path'])
                cores = ((cpu - previous[1]) / (now - previous[0])
                         if previous and now > previous[0] and cpu >= previous[1] else None)
                reason = ('no_previous_valid_counter' if previous is None else
                          'counter_reset' if cpu < previous[1] else 'nonpositive_interval')
                self.cgroup_previous[group['path']] = (now, cpu)
                row['cpu_seconds'] = observation(cpu)
                row['cores_used'] = observation(cores, reason)
            except (TypeError, KeyError, ValueError):
                row['cpu_seconds'] = observation(None, 'cgroup_cpu_unavailable')
                row['cores_used'] = observation(None, 'cgroup_cpu_unavailable')
            rows.append(row)
        return {'groups': rows, 'null_reason': self.discovery_reason}

    def sample(self):
        begin = self.clock()
        rows, issues = {}, set()
        try:
            # Do not materialize an unbounded /proc listing.
            with os.scandir(self.proc) as entries:
                count = 0
                for entry in entries:
                    if self.clock() - begin > self.budget:
                        issues.add('scan_time_budget_exceeded'); break
                    if not entry.name.isdigit():
                        continue
                    count += 1
                    if count > self.max_processes:
                        issues.add('process_scan_limit'); break
                    try:
                        rows[int(entry.name)] = stat(read(Path(entry.path) / 'stat'))
                    except (OSError, ValueError, IndexError):
                        issues.add('process_exited_denied_or_malformed')
        except OSError:
            issues.add('proc_unavailable')
        children = {}
        for pid, row in rows.items():
            children.setdefault(row['ppid'], []).append(pid)
        tree, todo = set(), [self.pid] if self.pid in rows else []
        while todo:
            pid = todo.pop()
            if pid in tree:
                continue
            tree.add(pid)
            todo.extend(children.get(pid, []))
        if not tree:
            issues.add('root_process_unavailable')
        elif self.root_identity is None:
            self.root_identity = rows[self.pid]['start']
        elif rows[self.pid]['start'] != self.root_identity:
            tree.clear()
            issues.add('root_pid_reused')
        delta = dict.fromkeys(self.totals, 0)
        io_count = 0
        for pid in tree:
            row = rows[pid]
            identity = (pid, row['start'])
            counters = {'cpu_seconds': row['ticks'] / self.hz}
            if self.clock() - begin <= self.budget:
                try:
                    data = pairs(read(self.proc / str(pid) / 'io'))
                    if stat(read(self.proc / str(pid) / 'stat'))['start'] != row['start']:
                        issues.add('pid_reused_during_sample')
                        continue
                    counters.update({key: data[key] for key in ('read_bytes', 'write_bytes', 'rchar', 'wchar')})
                    io_count += 1
                except (OSError, ValueError, KeyError, IndexError):
                    issues.add('process_io_unavailable')
            else:
                issues.add('scan_time_budget_exceeded')
            old = self.history.get(identity)
            if old is None and len(self.history) >= self.max_history:
                issues.add('identity_history_limit'); continue
            # New observed descendants born since startup can contribute lifetime
            # own counters. Older processes start at the first observed baseline.
            new_child = (pid != self.pid and self.boot_ticks is not None
                         and row['start'] >= self.boot_ticks)
            for key, value in counters.items():
                baseline = old.get(key, value) if old is not None else (0 if new_child else value)
                delta[key] += max(0, value - baseline)
            self.history[identity] = dict(old or {}, **counters)
        now = self.clock()
        elapsed = now - self.previous_time if self.previous_time is not None else None
        self.previous_time = now
        for key in delta:
            self.totals[key] += delta[key]
        process = {'process_count': len(tree),
                   'rss_bytes': observation(sum(max(0, rows[p]['rss_pages']) * self.page for p in tree) if tree else None,
                                            'root_process_unavailable'),
                   'cpu_seconds_observed': observation(self.totals['cpu_seconds'] if tree else None, 'root_process_unavailable'),
                   'cpu_seconds_delta': observation(delta['cpu_seconds'] if tree else None, 'root_process_unavailable'),
                   'cores_used': observation(delta['cpu_seconds'] / elapsed if tree and elapsed and elapsed > 0 else None,
                                             'no_previous_sample_or_root_unavailable'),
                   'io_observed_bytes': {key: observation(self.totals[key] if io_count else None, 'process_io_unavailable')
                                         for key in ('read_bytes', 'write_bytes', 'rchar', 'wchar')},
                   'io_delta_bytes': {key: observation(delta[key] if io_count else None, 'process_io_unavailable')
                                      for key in ('read_bytes', 'write_bytes', 'rchar', 'wchar')}}
        try:
            memory = pairs(read(self.proc / 'meminfo'))
            system = {key + '_bytes': observation(memory[key] * 1024 if key in memory else None,
                                                  'meminfo_field_unavailable')
                      for key in ('MemTotal', 'MemAvailable')}
            system['used_estimate_bytes'] = observation(
                (memory['MemTotal'] - memory['MemAvailable']) * 1024
                if 'MemTotal' in memory and 'MemAvailable' in memory else None,
                'meminfo_field_unavailable')
        except (OSError, ValueError):
            system = {key + '_bytes': observation(None, 'meminfo_unavailable') for key in ('MemTotal', 'MemAvailable')}
            system['used_estimate_bytes'] = observation(None, 'meminfo_unavailable')
        return {'process_tree': process, 'system_memory': system, 'cgroup': self.cgroups(now, begin + self.budget),
                'partial_reasons': sorted(issues), 'collection_seconds': self.clock() - begin,
                'interval_seconds': observation(elapsed, 'first_sample')}


def update_peaks(target, sample):
    """Sampled maxima only, deliberately separate from CUDA allocator peaks."""
    values = {'process_tree_rss_bytes': sample['process_tree']['rss_bytes'],
              'process_tree_cores_used': sample['process_tree']['cores_used'],
              'host_system_used_estimate_bytes': sample['system_memory']['used_estimate_bytes']}
    for group in sample['cgroup']['groups']:
        prefix = 'cgroup_' + group['path'] + ':'
        for name in ('memory.current', 'memory.usage_in_bytes'):
            raw = group['fields'].get(name, observation(None))['value']
            try:
                values[prefix + 'memory_bytes'] = observation(int(raw))
            except (TypeError, ValueError):
                pass
        values[prefix + 'cores_used'] = group['cores_used']
    for key, value in values.items():
        old = target.get(key, observation(None, 'no_valid_samples'))
        if value['value'] is not None and (old['value'] is None or value['value'] > old['value']):
            old = value
        target[key] = old

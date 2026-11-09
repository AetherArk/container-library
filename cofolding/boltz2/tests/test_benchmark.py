"""CPU/fake-engine observations, fake clock, and actual local HTTP progress transport."""
import ast
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resident import benchmark as bench
from resident.worker import run, Settings
from test_resident import FakeEngine
import test_cloud
import cloud_job as cloud
import prepare_cloud_job as host


class BenchmarkTests(unittest.TestCase):
    def test_missing_smi_and_unsupported_fields(self):
        self.assertEqual(bench.gpu_sample(None)['null_reason'], 'nvidia_smi_missing')
        with patch.object(bench.subprocess, 'run', side_effect=FileNotFoundError('SECRET')):
            self.assertEqual(bench.gpu_sample('fake')['null_reason'], 'nvidia_smi_query_failed')
        row = 'GPU-test, Test GPU, 123, 24000, N/A, 100, [Not Supported], 800, 900, 42, [N/A]\n'
        with patch.object(bench.subprocess, 'run', return_value=types.SimpleNamespace(stdout=row)):
            value = bench.gpu_sample('fake')['devices'][0]
        self.assertIsNone(value['utilization.gpu']['value'])
        self.assertEqual(value['utilization.gpu']['null_reason'], 'unsupported_or_unavailable')
        self.assertEqual(value['memory.total']['value'], 24000)
        self.assertIsNone(value['power.limit']['value'])

    def test_missing_smi_collector_still_completes_and_emits_null_reason(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(bench.shutil, 'which', return_value=None), contextlib.redirect_stdout(io.StringIO()):
            with bench.Benchmark(directory) as observer:
                with observer.stage('unit_test'):
                    observer.summary['status'] = 'completed'
            summary = json.loads((Path(directory) / 'benchmark.json').read_text())
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['gpu_initial']['null_reason'], 'nvidia_smi_missing')
            self.assertEqual(summary['stage_observation_counts']['unit_test'], 1)

    def test_metadata_allowlist_and_exception_redaction(self):
        value = {'SALAD_MACHINE_ID': 'machine-1', 'SALAD_MACHINE': 'obsolete', 'SALAD_API_KEY': 'SECRET', 'provider': 'generic',
                 'region': 'https://private?signature=SECRET', 'AWS_SECRET_ACCESS_KEY': 'SECRET',
                 'price_per_hour': 0.4, 'currency': 'USD', 'node_id': 'node\nSECRET'}
        clean = bench.sanitize_metadata(value)
        self.assertEqual(clean, {'SALAD_MACHINE_ID': 'machine-1', 'provider': 'generic',
                                 'price_per_hour': 0.4, 'currency': 'USD'})
        self.assertNotIn('SECRET', bench.error_label(ValueError('https://private?SECRET')))
        self.assertNotIn('SECRET', cloud.redact_log_line("'Authorization': 'Bearer SECRET'"))

    def test_lifecycle_repeat_seeds_resume_metrics_and_jsonl(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            inputs = root / 'inputs'
            inputs.mkdir()
            # Use real existing small neutral-ligand native fixture bytes, with FakeEngine.
            fixture = Path(__file__).parent / 'fixtures/resident_inputs/ethanol.yaml'
            (inputs / 'ethanol.yaml').write_bytes(fixture.read_bytes())
            assets = root / 'assets'
            assets.mkdir()
            checkpoint = root / 'checkpoint'
            checkpoint.write_bytes(b'CPU TEST ONLY')
            class MetricsEngine(FakeEngine):
                loads = 0
                calls = []
                fail = None
                stop = None
                def predict(self, prepared, attempt, seed):
                    super().predict(prepared, attempt, seed)
                    self.benchmark_metrics = {'forward_seconds': [1.0], 'test_double': True}
            output = root / 'output'
            self.assertEqual(run(inputs, output, assets, checkpoint, [12, 11, 12], Settings(samples=2), engine_factory=MetricsEngine), 0)
            first = json.loads((output / 'benchmark.json').read_text())
            rows = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual([rows[i]['model_was_loaded_before_cell'] for i in (0, 2, 4)], [False, True, True])
            self.assertEqual(len({r['job_key'] for r in rows}), 3)
            self.assertTrue(all(r['metrics']['test_double'] for r in rows))
            self.assertEqual(first['caller_metadata'], {})
            self.assertEqual(first['checkpoint_load_count'], 1)
            self.assertEqual(run(inputs, output, assets, checkpoint, [12, 11, 12], Settings(samples=2), resume=True, engine_factory=MetricsEngine), 0)
            rows = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(r['metrics'] is None and r['metrics_null_reason'] == 'reused_not_measured' for r in rows[6:]))
            self.assertEqual(MetricsEngine.loads, 1)
            self.assertNotEqual(first['invocation_id'], rows[-1]['invocation_id'])
            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
            self.assertTrue(all('schema' in e and 'elapsed_seconds' in e for e in events))
            self.assertFalse(any(stage['stage'] == 'load' for stage in json.loads((output / 'benchmark.json').read_text())['stages']))

    def test_resume_quarantines_partial_jsonl_tails_preserving_completed_history(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            inputs = root / 'inputs'; inputs.mkdir()
            (inputs / 'ethanol.yaml').write_bytes((Path(__file__).parent / 'fixtures/resident_inputs/ethanol.yaml').read_bytes())
            assets = root / 'assets'; assets.mkdir()
            checkpoint = root / 'checkpoint'; checkpoint.write_bytes(b'CPU TEST ONLY')
            class Engine(FakeEngine):
                loads = 0
                calls = []
                fail = None
                stop = None
            output = root / 'output'
            self.assertEqual(run(inputs, output, assets, checkpoint, [12], Settings(), engine_factory=Engine), 0)
            originals = {name: (output / name).read_bytes() for name in ('events.jsonl', 'predictions.jsonl')}
            fragments = {'events.jsonl': b'{"event":"unfinished',
                         'predictions.jsonl': b'{"status":"verified","incomplete":'}
            for name, fragment in fragments.items():
                with (output / name).open('ab') as stream:
                    stream.write(fragment)
            self.assertEqual(run(inputs, output, assets, checkpoint, [12], Settings(), resume=True, engine_factory=Engine), 0)
            for name, original in originals.items():
                data = (output / name).read_bytes()
                self.assertTrue(data.startswith(original))
                rows = [json.loads(line) for line in data.splitlines()]
                self.assertGreater(len(rows), len(original.splitlines()))
                evidence = list(output.glob(name + '.partial-*'))
                self.assertEqual(len(evidence), 1)
                self.assertEqual(evidence[0].read_bytes(), fragments[name])
                self.assertEqual(evidence[0].stat().st_mode & 0o777, 0o600)
                self.assertFalse(any(row.get('incomplete') for row in rows))
            predictions = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
            self.assertEqual([row['status'] for row in predictions], ['completed', 'completed'])
            self.assertEqual([row['reused'] for row in predictions], [False, True])
            self.assertTrue(all(row['verified'] for row in predictions))
            self.assertEqual(Engine.loads, 1)

    def test_tail_quarantine_handles_large_and_fragment_only_files_and_save_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for prefix in (b'', b'{"old":true}\n'):
                with self.subTest(prefix=prefix):
                    path = root / 'rows.jsonl'
                    fragment = b'{"partial":"' + b'x' * 150000
                    path.write_bytes(prefix + fragment)
                    observer = bench.Benchmark(root, sample_gpu=False)
                    observer.append(path.name, {'new': True})
                    self.assertEqual([json.loads(row) for row in path.read_bytes().splitlines()],
                                     ([{'old': True}] if prefix else []) + [{'new': True, 'schema': 1, 'invocation_id': observer.id}])
                    self.assertEqual(sorted(p.read_bytes() for p in root.glob('rows.jsonl.partial-*'))[-1], fragment)
            path.write_bytes(b'{"unfinished')
            observer = bench.Benchmark(root, sample_gpu=False)
            with patch.object(bench.shutil, 'copyfileobj', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    observer.append(path.name, {'new': True})
            self.assertEqual(path.read_bytes(), b'{"unfinished')

    def test_failure_rows_and_suppressed_error(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            inputs = root / 'inputs'; inputs.mkdir()
            (inputs / 'good.yaml').write_text('good')
            assets = root / 'assets'; assets.mkdir()
            checkpoint = root / 'checkpoint'; checkpoint.write_bytes(b'TEST')
            class Failed(FakeEngine):
                def prepare(self, *args):
                    raise ValueError('https://secret.example?token=DO-NOT-LEAK')
            output = root / 'output'
            self.assertEqual(run(inputs, output, assets, checkpoint, [11, 12], Settings(), engine_factory=Failed), 1)
            for path in output.glob('*json*'):
                self.assertNotIn('DO-NOT-LEAK', path.read_text())
            rows = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
            self.assertEqual([r['status'] for r in rows], ['failed', 'failed'])
            summary = json.loads((output / 'benchmark.json').read_text())
            self.assertEqual(summary['jobs_failed'], 2)
            self.assertTrue(all(s['status'] == 'failed' for s in summary['stages'] if s['stage'] == 'preparation'))

    def test_native_predict_real_method_with_cpu_doubles(self):
        # Compile the actual method only; no torch/Boltz imports or CUDA access.
        source = Path(__file__).resolve().parents[1] / 'resident/native.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NativeEngine')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'predict')
        clock = [0.0]
        order = []
        def advance(seconds, label):
            order.append(label); clock[0] += seconds
        cuda = types.SimpleNamespace(synchronize=lambda: order.append('sync'),
                    memory_allocated=lambda: 10, memory_reserved=lambda: 20,
                    reset_peak_memory_stats=lambda: order.append('reset'),
                    max_memory_allocated=lambda: 30, max_memory_reserved=lambda: 40)
        torch = types.SimpleNamespace(cuda=cuda, version=types.SimpleNamespace(cuda='TEST'),
                    backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(version=lambda: None)),
                    inference_mode=contextlib.nullcontext, autocast=lambda **kw: contextlib.nullcontext(),
                    bfloat16='TEST')
        class Model:
            device = 'CPU TEST ONLY'
            def named_parameters(self):
                return []
            def forward(self, batch):
                advance(3, 'forward'); return {'result': True}
            def register_forward_hook(self, hook):
                self.hook = hook
                return types.SimpleNamespace(remove=lambda: order.append('remove_hook'))
            def predict_step(self, batch, index):
                result = self.forward(batch)
                self.hook(self, (batch,), result)
                advance(2, 'predict_tail')
                return {'exception': False}
        class Writer:
            def __init__(self, **kw):
                pass
            def write_on_batch_end(self, *args):
                advance(11, 'writer')
        def packet(directory, label, value):
            advance(7 if label == 'full_forward' else 5, label)
            return value
        namespace = {'torch': torch, 'time': types.SimpleNamespace(perf_counter=lambda: clock[0]),
                     'atomic_json': lambda *args: None, 'save_packet': packet,
                     'timed_forward': bench.timed_forward, 'seed_everything': lambda *a, **kw: None,
                     'native_boundary': lambda call: call,
                     'BoltzWriter': Writer, 'gc': types.SimpleNamespace(collect=lambda: None),
                     'Path': Path, 'save_token_map': lambda *a, **kw: None,
                     'token_map': lambda *a, **kw: {},
                     'derived': types.SimpleNamespace(TokenMapError=ValueError)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        engine = types.SimpleNamespace(model=Model(), load_args={}, settings=Settings())
        with tempfile.TemporaryDirectory() as directory, patch.object(bench.time, 'perf_counter', side_effect=lambda: clock[0]):
            root = Path(directory)
            prepared = {'batch': {}, 'data': {}, 'processed': root, 'native': root,
                        'module': types.SimpleNamespace(transfer_batch_to_device=lambda batch, *a: batch)}
            namespace['predict'](engine, prepared, root, 11)
        metrics = engine.benchmark_metrics
        self.assertEqual(metrics['forward_seconds'], [3])
        self.assertEqual(metrics['capture_hook_write_seconds'], 7)
        self.assertEqual(metrics['predict_step_seconds'], 12)  # 3 compute + 7 capture + 2 post-forward
        self.assertEqual(metrics['output_write_seconds'], 21)  # consumed + prediction + native writer
        self.assertLess(order.index('reset'), order.index('forward'))
        self.assertEqual(order[order.index('forward') - 1], 'sync')
        self.assertEqual(order[order.index('forward') + 1], 'sync')
        self.assertEqual(order[order.index('forward') + 2], 'full_forward')
        self.assertNotIn('forward', engine.model.__dict__)  # original descriptor restored
        self.assertEqual(metrics['allocator_baseline_bytes'], {'allocated': 10, 'reserved': 20})
        self.assertEqual(metrics['allocator_peak_bytes'], {'allocated': 30, 'reserved': 40})
        # A capture storage failure is timed as storage, restores an existing
        # instance forward override, and never proceeds to the native writer.
        def failing_packet(directory, label, value):
            result = packet(directory, label, value)
            if label == 'full_forward':
                raise OSError('test capture write failure')
            return result
        namespace['save_packet'] = failing_packet
        engine.model = Model()
        original_override = engine.model.forward
        engine.model.forward = original_override
        order.clear()
        with tempfile.TemporaryDirectory() as directory, patch.object(bench.time, 'perf_counter', side_effect=lambda: clock[0]):
            root = Path(directory)
            prepared.update(batch={}, processed=root, native=root)
            with self.assertRaises(OSError):
                namespace['predict'](engine, prepared, root, 11)
        failed = engine.benchmark_metrics
        self.assertEqual(failed['forward_seconds'], [3])
        self.assertEqual(failed['capture_hook_write_seconds'], 7)
        self.assertEqual(failed['predict_step_seconds'], 10)
        self.assertEqual(failed['output_write_seconds'], 5)
        self.assertIs(engine.model.forward, original_override)
        self.assertIn('remove_hook', order)
        self.assertNotIn('writer', order)


@contextlib.contextmanager
def slow_http():
    """Actual localhost slow headers/body; no external network or provider."""
    entered = threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def transfer(self):
            if self.command == 'PUT':
                self.rfile.read(int(self.headers['Content-Length']))
            entered.set()
            try:
                if self.path.startswith('/headers'):
                    for byte in b'HTTP/1.0 200 OK\r\nContent-Length: 6\r\n\r\nabcdef':
                        self.wfile.write(bytes([byte])); self.wfile.flush()
                        time.sleep(0.04)
                else:
                    self.send_response(200)
                    self.send_header('Content-Length', '6')
                    self.end_headers()
                    for byte in b'abcdef':
                        time.sleep(0.08)
                        self.wfile.write(bytes([byte])); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        do_GET = transfer
        do_PUT = transfer
        def log_message(self, format, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', entered
    finally:
        server.shutdown(); server.server_close(); thread.join()


class ProgressDeadlineTests(unittest.TestCase):
    def test_progress_deadline_covers_slow_headers_body_and_put_and_reaps(self):
        http = cloud.HTTP(int(time.time()) + 60)
        with slow_http() as (base, entered):
            # Preserve generic inactivity-based streaming: .08s bytes keep a
            # .1s socket timeout alive beyond .4s, as for large result transfers.
            begin = time.monotonic()
            self.assertEqual(http.request(base + '/body', timeout_seconds=0.1),
                             (hashlib.sha256(b'abcdef').hexdigest(), 6))
            self.assertGreater(time.monotonic() - begin, 0.4)
            real_popen = cloud.subprocess.Popen
            for endpoint, method in (('/body', 'GET'), ('/headers', 'GET'), ('/headers', 'PUT')):
                with self.subTest(endpoint=endpoint, method=method):
                    entered.clear()
                    children = []
                    def spawn(argv, **kwargs):
                        self.assertNotIn('TEST-CAPABILITY', repr(argv))
                        self.assertEqual(kwargs['env'], {})
                        child = real_popen(argv, **kwargs); children.append(child)
                        return child
                    begin = time.monotonic()
                    with patch.object(cloud.subprocess, 'Popen', side_effect=spawn), self.assertRaises(cloud.TransportError) as caught:
                        http.progress_request(base + endpoint + '?secret=TEST-CAPABILITY', method,
                                              data=b'{}' if method == 'PUT' else None, timeout_seconds=0.25)
                    self.assertLess(time.monotonic() - begin, 0.8)
                    self.assertNotIn('TEST-CAPABILITY', str(caught.exception))
                    self.assertEqual(len(children), 1)
                    self.assertEqual(children[0].returncode, -cloud.signal.SIGKILL)
                    self.assertTrue(entered.is_set())
                    with self.assertRaises(ChildProcessError):
                        os.waitpid(children[0].pid, os.WNOHANG)
                    self.assertTrue(children[0].stdin.closed and children[0].stdout.closed)

    def test_progress_deadline_covers_blocked_dns_and_connect(self):
        real_popen = cloud.subprocess.Popen
        for operation in ('getaddrinfo', 'connect'):
            with self.subTest(operation=operation):
                children = []
                def spawn(argv, **kwargs):
                    # Explicit isolated-process double for otherwise OS-dependent
                    # DNS/connect stalls. All transport below is the real method.
                    argv = list(argv)
                    target = 'socket.getaddrinfo' if operation == 'getaddrinfo' else 'socket.socket.connect'
                    argv[-1] = f'import socket,time; {target} = lambda *a,**k: time.sleep(2); ' + argv[-1]
                    child = real_popen(argv, **kwargs); children.append(child)
                    return child
                begin = time.monotonic()
                with patch.object(cloud.subprocess, 'Popen', side_effect=spawn), self.assertRaises(cloud.TransportError):
                    cloud.HTTP(int(time.time()) + 60).progress_request('http://127.0.0.1:9/', timeout_seconds=0.25)
                self.assertLess(time.monotonic() - begin, 0.8)
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].returncode)
                with self.assertRaises(ChildProcessError):
                    os.waitpid(children[0].pid, os.WNOHANG)

    def test_close_joins_inflight_progress_before_freeze_without_late_writes(self):
        with tempfile.TemporaryDirectory() as directory, slow_http() as (base, entered), contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            observer = bench.Benchmark(root, sample_gpu=False)
            uploader = cloud.ProgressUploads(cloud.HTTP(int(time.time()) + 60),
                        {'job_id': 'test', 'attempt_id': 'test'},
                        [{'put': base + '/headers', 'get': base + '/body'}], root, observer)
            uploader.timeout_seconds = 0.3
            uploader.thread = threading.Thread(target=uploader.publish)
            uploader.thread.start()
            self.assertTrue(entered.wait(2))
            begin = time.monotonic()
            result = uploader.close()
            self.assertLess(time.monotonic() - begin, 0.8)
            self.assertFalse(uploader.thread.is_alive())
            self.assertEqual(result['verified_slots'], [])
            self.assertEqual(result['unverified_slots'], [0])
            frozen = (root / 'events.jsonl').read_bytes()
            uploader.publish()  # closed publishers cannot write another event
            time.sleep(0.4)
            self.assertEqual((root / 'events.jsonl').read_bytes(), frozen)
            self.assertEqual(uploader.close(), result)


class ProgressTests(unittest.TestCase):
    setUpClass = classmethod(test_cloud.CloudTests.setUpClass.__func__)
    tearDownClass = classmethod(test_cloud.CloudTests.tearDownClass.__func__)
    setUp = test_cloud.CloudTests.setUp
    run_cloud = test_cloud.CloudTests.run_cloud
    receipt = test_cloud.CloudTests.receipt
    unpack_result = test_cloud.CloudTests.unpack_result

    # Reuse the actual local HTTP fixture and fake child, not a provider or GPU.
    def enable_progress(self, slots=3):
        self.args.output = self.root / 'with-progress'
        self.args.progress_slots = slots
        self.args.progress_interval = 10
        self.job = host.prepare(self.args, client=self.store)
        self.spec = json.loads(json.loads((self.args.output / 'runtime-env.secret.json').read_text())['COFOLD_CLOUD_SPEC'])

    def test_progress_host_slots_and_verified_roundtrip(self):
        self.enable_progress()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_cloud(), 0)
        receipt = self.receipt()
        self.assertEqual(receipt['progress_uploads']['verified_slots'], [0, 1])
        self.assertEqual(receipt['progress_uploads']['unverified_slots'], [])
        for key in self.job['progress_keys'][:2]:
            body = self.store.objects[key]
            self.assertLess(len(body), 65536)
            self.assertNotIn(b'TEST-CAPABILITY', body)
            self.assertIn(('GET', key), self.store.events)
        self.assertEqual(len(self.spec['progress_urls']), 3)
        self.assertIn('outputs/cloud_transport/benchmark.json', self.unpack_result())

    def test_slow_progress_does_not_block_final_archive_or_claim_success(self):
        self.enable_progress(2)
        real_init = cloud.ProgressUploads.__init__
        def accelerated_init(uploader, *args, **kwargs):
            real_init(uploader, *args, **kwargs)
            uploader.timeout_seconds = 0.25  # test-only shortened wall deadline
        with slow_http() as (base, entered):
            self.spec['progress_urls'] = [{'put': base + '/headers', 'get': base + '/body'}] * 2
            with patch.object(cloud.ProgressUploads, '__init__', accelerated_init), contextlib.redirect_stdout(io.StringIO()):
                begin = time.monotonic()
                self.assertEqual(self.run_cloud(), 0)
                self.assertLess(time.monotonic() - begin, 3)
        progress = self.receipt()['progress_uploads']
        self.assertEqual(progress['verified_slots'], [])
        self.assertEqual(progress['unverified_slots'], [0, 1])
        archive = self.unpack_result()
        events = [json.loads(row) for row in archive['outputs/cloud_transport/events.jsonl'].splitlines()]
        self.assertEqual(sum(event['event'] == 'progress_upload_unverified' for event in events), 2)
        self.assertFalse(any(event['event'] == 'progress_upload_verified' for event in events))

    def test_archive_is_refused_if_publisher_cannot_join(self):
        joins = []
        thread = types.SimpleNamespace(join=lambda **kwargs: joins.append(kwargs), is_alive=lambda: True)
        uploader = cloud.ProgressUploads(None, {'job_id': 'test', 'attempt_id': 'test'}, [], self.root,
                                        types.SimpleNamespace(event=lambda *a, **kw: None))
        uploader.thread = thread
        with self.assertRaises(cloud.TransportError):
            uploader.close()
        self.assertEqual(joins, [{'timeout': 22.0}])
        self.assertFalse(uploader.closed)
        with (patch.object(cloud.ProgressUploads, 'shutdown', side_effect=cloud.TransportError('test-only')),
              patch.object(cloud, 'archive_outputs') as archive, contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaises(cloud.TransportError):
                self.run_cloud()
            archive.assert_not_called()

    def test_progress_failure_does_not_stop_worker_or_claim_durability(self):
        self.enable_progress()
        self.store.fail_put = self.job['progress_keys'][0]
        self.store.corrupt_get = self.job['progress_keys'][1]
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            self.assertEqual(self.run_cloud(), 0)
        progress = self.receipt()['progress_uploads']
        self.assertEqual(progress['verified_slots'], [])
        self.assertEqual(progress['unverified_slots'], [0, 1])
        self.assertNotIn('TEST-CAPABILITY', log.getvalue())

    def test_progress_periodic_cadence_is_bounded_and_snapshot_survives_without_archive(self):
        self.enable_progress(2)
        output = self.root / 'live'; output.mkdir()
        (output / 'progress.json').write_text(json.dumps({'status': 'running', 'jobs_completed': 1}))
        observer = types.SimpleNamespace(event=lambda *a, **kw: None)
        uploader = cloud.ProgressUploads(cloud.HTTP(self.job['expires_at']), self.job, self.spec['progress_urls'], output, observer)
        uploader.interval = 0.01  # test-only clock acceleration; production min 10s
        uploader.start()
        deadline = time.monotonic() + 3
        while uploader.index < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        result = uploader.close()
        self.assertEqual(result['verified_slots'], [0, 1])
        self.assertEqual(result['slots_attempted'], 2)
        self.assertNotIn(self.job['result_key'], self.store.objects)
        self.assertEqual(json.loads(self.store.objects[self.job['progress_keys'][1]])['progress']['jobs_completed'], 1)

    def test_missing_or_expired_progress_is_not_durable_success(self):
        self.enable_progress(1)
        observer = types.SimpleNamespace(event=lambda *a, **kw: None)
        uploader = cloud.ProgressUploads(cloud.HTTP(self.job['expires_at']), self.job, [], self.root, observer)
        uploader.start()
        self.assertEqual(uploader.close()['verified_slots'], [])
        http = cloud.HTTP(self.job['expires_at']); http.expires_at = 0
        uploader = cloud.ProgressUploads(http, self.job, self.spec['progress_urls'], self.root, observer)
        uploader.start()
        result = uploader.close()
        self.assertEqual(result['verified_slots'], [])
        self.assertEqual(result['unverified_slots'], [0])
        missing = cloud.ProgressUploads(http, self.job, [{'put': 'https://missing/SECRET'}], self.root, observer)
        missing.start()
        self.assertEqual(missing.close()['unverified_slots'], [0])

    def test_worker_log_urls_redacted_before_disk_and_safe_events_forwarded(self):
        self.worker.write_text(self.worker.read_text() + '\nprint("https://private/path?signature=DO-NOT-LEAK")\nprint("api_key=DO-NOT-LEAK")\nprint(\'{"schema":1,"event":"stage_started","stage":"test_only"}\')\n')
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            self.assertEqual(self.run_cloud(), 0)
        archive = self.unpack_result()
        self.assertIn(b'[URL_REDACTED]', archive['outputs/worker.log'])
        self.assertNotIn(b'DO-NOT-LEAK', archive['outputs/worker.log'])
        self.assertNotIn('DO-NOT-LEAK', log.getvalue())
        self.assertIn('worker_event', log.getvalue())

    def test_wrapper_provider_metadata_file_precedes_environment_stripping(self):
        # Hard-code the provider contract, not values derived from the allowlist.
        expected = {'SALAD_ORGANIZATION_NAME': 'organization', 'SALAD_ORGANIZATION_ID': 'organization-1',
                    'SALAD_PROJECT_NAME': 'project', 'SALAD_PROJECT_ID': 'project-1',
                    'SALAD_CONTAINER_GROUP_NAME': 'group', 'SALAD_CONTAINER_GROUP_ID': 'group-1',
                    'SALAD_MACHINE_ID': 'machine-1', 'SALAD_INSTANCE_ID': 'instance-1'}
        with patch.dict(os.environ, dict(expected, SALAD_API_KEY='MUST-NOT-LEAK',
                        SALAD_MACHINE='obsolete', SALAD_UNKNOWN_ID='unknown')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_cloud(), 0)
        metadata = json.loads((self.root / 'work/runtime/benchmark-metadata.json').read_text())
        self.assertEqual(metadata, expected)
        self.assertEqual(set(bench.PROVIDER_KEYS), set(expected))
        self.assertEqual(self.receipt()['benchmark']['caller_metadata'], metadata)
        self.assertNotIn(b'MUST-NOT-LEAK', self.store.objects[self.job['result_key']])


if __name__ == '__main__':
    unittest.main()

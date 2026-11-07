"""CPU transport tests: real local HTTP/subprocess, explicit FAKE worker, NO inference/R2."""
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cofolding/boltz2'))
import cloud_job as cloud
import prepare_cloud_job as host

IMAGE = 'test/boltz2@sha256:' + 'a' * 64


class ObjectStore:
    """In-memory S3 SDK double, actual HTTP presigned-capability double."""
    def __init__(self):
        self.objects = {}
        self.events = []
        self.fail_put = None
        self.corrupt_get = None
        self.signatures = []
        store = self

        class Handler(BaseHTTPRequestHandler):
            def do_PUT(self):
                key = self.path.split('?')[0].lstrip('/')
                store.events.append(('PUT', key))
                data = self.rfile.read(int(self.headers['Content-Length']))
                if self.headers.get('If-None-Match') != '*':
                    self.send_error(400)
                elif key == store.fail_put:
                    self.send_error(503)
                elif key in store.objects:
                    self.send_error(412)
                else:
                    store.objects[key] = data
                    self.send_response(200)
                    self.end_headers()

            def do_GET(self):
                key = self.path.split('?')[0].lstrip('/')
                store.events.append(('GET', key))
                if key not in store.objects:
                    self.send_error(404)
                    return
                data = store.objects[key]
                if key == store.corrupt_get:
                    data = b'x' * len(data)
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def put_object(self, Bucket, Key, Body, IfNoneMatch):
        assert IfNoneMatch == '*'
        if Key in self.objects:
            raise ValueError('collision')
        self.objects[Key] = Body.read() if hasattr(Body, 'read') else Body

    def get_object(self, Bucket, Key):
        return {'Body': io.BytesIO(self.objects[Key])}

    def generate_presigned_url(self, method, Params, ExpiresIn):
        self.signatures.append((method, Params, ExpiresIn))
        return f'http://127.0.0.1:{self.server.server_port}/{Params["Key"]}?secret=TEST-CAPABILITY'


class CloudTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = ObjectStore()

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        (self.inputs / 'msa').mkdir()
        (self.inputs / 'msa/local.a3m').write_bytes(b'>test\nAAAA\n')
        self.yaml = b'# preserve formatting\nsequences:\n- protein:\n    id: A\n    sequence: AAAA\n    msa: msa/local.a3m\n'
        (self.inputs / 'input.yaml').write_bytes(self.yaml)
        (self.inputs / 'unreferenced.secret.json').write_text('TEST-CAPABILITY must not be bundled')
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        self.checkpoint = self.assets / 'checkpoint'
        self.checkpoint.write_bytes(b'explicit-test-only')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.worker = self.bin / 'cofold-run'
        self.worker.write_text('''#!''' + sys.executable + '''
# Explicit transport test double; not Boltz, not science.
import json, os, pathlib, sys, signal
args = sys.argv[1:]
out = pathlib.Path(args[args.index('--output')+1])
assert not any(out.iterdir()), 'wrapper polluted worker output'
assert not any(k.startswith(('AWS_', 'R2_', 'SALAD_', 'COFOLD_CLOUD_')) for k in os.environ)
assert 'HF_TOKEN' not in os.environ
assert os.environ['HF_HUB_OFFLINE'] == '1'
for key in ('HOME','TMPDIR','HF_HOME','XDG_CACHE_HOME'):
    assert pathlib.Path(os.environ[key]).is_dir()
(out/'payload.bin').write_bytes(b'full native output placeholder: UNIT TEST ONLY')
(out/'dispatch.json').write_text(json.dumps(args))
(out/'run.json').write_text(json.dumps({'status':'completed'}))
''')
        self.worker.chmod(0o755)
        self.env = patch.dict(os.environ, {'PATH': str(self.bin) + os.pathsep + os.environ['PATH'],
                        'AWS_SECRET_ACCESS_KEY': 'MUST-NOT-LEAK', 'HF_TOKEN': 'MUST-NOT-LEAK'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.store.objects.clear()
        self.store.events.clear()
        self.store.signatures.clear()
        self.store.fail_put = None
        self.store.corrupt_get = None
        self.args = host.parser().parse_args(['--inputs', str(self.inputs), '--output', str(self.root/'prepared'),
                       '--bucket', 'tests', '--image', IMAGE, '--seeds', '12', '11', '12', '--samples', '5', '--no-kernels'])
        self.job = host.prepare(self.args, client=self.store)
        self.spec = json.loads(json.loads((self.root/'prepared/runtime-env.secret.json').read_text())['COFOLD_CLOUD_SPEC'])

    def run_cloud(self):
        return cloud.run_cloud(self.spec, self.root/'work', self.assets, self.checkpoint)

    def receipt(self):
        return json.loads(self.store.objects[self.job['receipt_key']])

    def unpack_result(self):
        data = self.store.objects[self.job['result_key']]
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            return {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}

    def test_round_trip_dispatch_full_outputs_hash_then_receipt(self):
        self.assertEqual(self.run_cloud(), 0)
        receipt = self.receipt()
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['result_sha256'], hashlib.sha256(self.store.objects[self.job['result_key']]).hexdigest())
        result = self.unpack_result()
        self.assertIn('outputs/payload.bin', result)
        self.assertIn('outputs/worker.log', result)
        args = json.loads(result['outputs/dispatch.json'])
        self.assertEqual(args[args.index('--seeds')+1:args.index('--samples')], ['12', '11', '12'])
        self.assertEqual(args[args.index('--samples')+1], '5')
        self.assertIn('--no-kernels', args)
        self.assertEqual(self.store.events[-4:], [('PUT', self.job['result_key']),
                         ('GET', self.job['result_key']), ('PUT', self.job['receipt_key']),
                         ('GET', self.job['receipt_key'])])

    def test_phase_logs_are_visible_without_capabilities(self):
        log = io.StringIO()
        with redirect_stdout(log):
            self.assertEqual(self.run_cloud(), 0)
        text = log.getvalue()
        self.assertIn('cloud phase: downloading inputs', text)
        self.assertIn('cloud terminal status: completed', text)
        self.assertNotIn('TEST-CAPABILITY', text)
        self.assertNotIn('MUST-NOT-LEAK', text)
        self.assertNotIn('http://', text)

    def test_upload_failure_has_no_completed_log(self):
        self.store.fail_put = self.job['result_key']
        log = io.StringIO()
        with redirect_stdout(log), self.assertRaises(cloud.TransportError):
            self.run_cloud()
        self.assertNotIn('cloud terminal status: completed', log.getvalue())

    def test_host_layout_unchanged_inputs_private_modes_and_no_public_secrets(self):
        job = self.job
        self.assertRegex(job['job_id'], r'^boltz2-\d{8}T\d{6}Z-[a-f0-9]{12}$')
        self.assertTrue(job['input_key'].startswith('salad-benchmarking/test/' + job['job_id'] + '/inputs/'))
        self.assertIn('/results/' + job['attempt_id'] + '/', job['result_key'])
        self.assertEqual(set(job['files']), {'input.yaml', 'msa/local.a3m'})
        with tarfile.open(self.root/'prepared/inputs.tar.gz') as tar:
            self.assertNotIn('inputs/unreferenced.secret.json', tar.getnames())
            self.assertEqual(tar.extractfile('inputs/input.yaml').read(), self.yaml)
            self.assertEqual(tar.extractfile('inputs/msa/local.a3m').read(), b'>test\nAAAA\n')
        for name in ('runtime-env.secret.json', 'deployment.secret.json'):
            self.assertEqual((self.root/'prepared'/name).stat().st_mode & 0o777, 0o600)
        for name in ('job.json', 'launch.json'):
            public = (self.root/'prepared'/name).read_text()
            self.assertNotIn('TEST-CAPABILITY', public)
            self.assertNotIn('MUST-NOT-LEAK', public)
        payload = json.loads((self.root/'prepared/deployment.secret.json').read_text())
        self.assertEqual(payload['command'], ['cofold-cloud-run'])
        for method, params, ttl in self.store.signatures:
            if method == 'put_object':
                self.assertEqual(params['IfNoneMatch'], '*')
            self.assertGreater(ttl, 0)
            self.assertLessEqual(ttl, 86400)

    def test_bad_input_hash_failed_receipt_without_worker(self):
        self.store.objects[self.job['input_key']] = b'corrupt'
        self.assertEqual(self.run_cloud(), 1)
        self.assertEqual(self.receipt()['status'], 'failed')
        self.assertEqual(self.receipt()['phase'], 'inputs')
        self.assertNotIn('outputs/payload.bin', self.unpack_result())

    def test_worker_failure_uploads_partial_outputs(self):
        self.worker.write_text(self.worker.read_text() + '\nsys.exit(7)\n')
        self.assertEqual(self.run_cloud(), 1)
        self.assertEqual(self.receipt()['worker_exit_code'], 7)
        self.assertEqual(self.receipt()['status'], 'failed')
        self.assertIn('outputs/payload.bin', self.unpack_result())

    def test_zero_exit_not_completed_metadata_is_failure(self):
        self.worker.write_text(self.worker.read_text().replace("'status':'completed'", "'status':'failed'"))
        self.assertEqual(self.run_cloud(), 1)
        self.assertEqual(self.receipt()['status'], 'failed')

    def test_upload_failure_never_publishes_receipt(self):
        self.store.fail_put = self.job['result_key']
        with self.assertRaises(cloud.TransportError):
            self.run_cloud()
        self.assertNotIn(self.job['receipt_key'], self.store.objects)
        self.assertTrue((self.root/'work/results.tar').is_file())

    def test_readback_corruption_never_publishes_receipt(self):
        self.store.corrupt_get = self.job['result_key']
        with self.assertRaises(cloud.TransportError):
            self.run_cloud()
        self.assertNotIn(self.job['receipt_key'], self.store.objects)

    def test_claim_prevents_restart_before_worker(self):
        self.assertEqual(self.run_cloud(), 0)
        before = dict(self.store.objects)
        with self.assertRaises(cloud.TransportError):
            cloud.run_cloud(self.spec, self.root/'restart', self.assets, self.checkpoint)
        self.assertFalse((self.root/'restart').exists())
        self.assertEqual(self.store.objects, before)

    def test_result_collision_does_not_overwrite_or_complete(self):
        self.store.objects[self.job['result_key']] = b'prior'
        with self.assertRaises(cloud.TransportError):
            self.run_cloud()
        self.assertEqual(self.store.objects[self.job['result_key']], b'prior')
        self.assertNotIn(self.job['receipt_key'], self.store.objects)

    def test_receipt_collision_does_not_overwrite(self):
        self.store.objects[self.job['receipt_key']] = b'prior'
        with self.assertRaises(cloud.TransportError):
            self.run_cloud()
        self.assertEqual(self.store.objects[self.job['receipt_key']], b'prior')

    def test_expired_or_missing_expiration_no_claim(self):
        for expires in (None, int(time.time()) - 1):
            spec = deepcopy(self.spec)
            spec['job']['expires_at'] = expires
            with self.assertRaises(ValueError):
                cloud.run_cloud(spec, self.root/'unused', self.assets, self.checkpoint)
        self.assertNotIn(self.job['claim_key'], self.store.objects)

    def test_expiry_after_input_download_prevents_asset_or_worker_dispatch(self):
        clock = [time.time()]
        real_extract = cloud.extract_inputs
        def expire_after_extract(*args):
            result = real_extract(*args)
            clock[0] = self.job['expires_at'] + 1
            return result
        with patch.object(cloud.time, 'time', side_effect=lambda: clock[0]), \
             patch.object(cloud, 'extract_inputs', side_effect=expire_after_extract), \
             patch.object(cloud.subprocess, 'Popen') as spawn:
            with self.assertRaises(cloud.TransportError):
                self.run_cloud()
            spawn.assert_not_called()
        self.assertNotIn(self.job['receipt_key'], self.store.objects)

    def test_expiry_during_assets_prevents_worker_dispatch(self):
        from unittest.mock import Mock
        clock = [time.time()]
        def finish_assets():
            clock[0] = self.job['expires_at'] + 1
            return 0
        child = Mock()
        child.wait.side_effect = finish_assets
        with patch.object(cloud.time, 'time', side_effect=lambda: clock[0]), \
             patch.object(cloud.subprocess, 'Popen', return_value=child) as spawn:
            with self.assertRaises(cloud.TransportError):
                cloud.run_cloud(self.spec, self.root/'work')
        self.assertEqual(spawn.call_count, 1)
        self.assertIn('--structure-only', spawn.call_args.args[0])
        self.assertNotIn(self.job['receipt_key'], self.store.objects)

    def check_stop_during_spawn(self, with_assets):
        from unittest.mock import Mock
        child = Mock()
        child.poll.return_value = None
        child.wait.return_value = 143
        def spawn(*args, **kwargs):
            # A real signal arrives before Popen returns/assigns the new child.
            signal.raise_signal(signal.SIGTERM)
            return child
        with patch.object(cloud.subprocess, 'Popen', side_effect=spawn):
            rc = (self.run_cloud() if with_assets else
                  cloud.run_cloud(self.spec, self.root/'work'))
        self.assertEqual(rc, 1)
        child.send_signal.assert_called_once_with(signal.SIGTERM)
        self.assertEqual(self.receipt()['status'], 'stopped')

    def test_sigterm_at_worker_spawn_is_forwarded(self):
        self.check_stop_during_spawn(with_assets=True)

    def test_sigterm_at_asset_spawn_is_forwarded(self):
        self.check_stop_during_spawn(with_assets=False)

    def test_receipt_readback_corruption_cannot_report_success(self):
        self.store.corrupt_get = self.job['receipt_key']
        log = io.StringIO()
        with redirect_stdout(log), self.assertRaises(cloud.TransportError):
            self.run_cloud()
        self.assertEqual(self.store.events[-1], ('GET', self.job['receipt_key']))
        self.assertNotIn('cloud terminal status: completed', log.getvalue())

    def test_main_redacts_signed_url_exception(self):
        import contextlib
        stderr = io.StringIO()
        with patch.dict(os.environ, {'COFOLD_CLOUD_SPEC': json.dumps(self.spec)}), \
             patch.object(cloud, 'run_cloud', side_effect=ValueError('https://secret?TEST-CAPABILITY')), \
             contextlib.redirect_stderr(stderr):
            self.assertEqual(cloud.main([]), 1)
        self.assertNotIn('TEST-CAPABILITY', stderr.getvalue())
        self.assertNotIn('COFOLD_CLOUD_SPEC', os.environ)

    def test_transport_exception_redacts_signed_url(self):
        http = cloud.HTTP(int(time.time()) + 60)
        with patch.object(http.opener, 'open', side_effect=ValueError('https://secret?TEST-CAPABILITY')):
            with self.assertRaises(cloud.TransportError) as error:
                http.request('https://example.org/?secret=TEST-CAPABILITY', path=self.root/'download')
            self.assertNotIn('TEST-CAPABILITY', str(error.exception))
            self.assertIsNone(error.exception.__cause__)

    def test_host_reuse_and_invalid_seeds_settings_paths(self):
        with self.assertRaises(FileExistsError):
            host.prepare(self.args, self.store)
        for seeds in ([], [-1], [4294967296], [True]):
            with self.assertRaises(ValueError):
                cloud.validate_settings(seeds, self.job['settings'])
        for key, value in (('samples', 0), ('recycles', -1), ('steps', True), ('no_kernels', 1)):
            settings = dict(self.job['settings'], **{key: value})
            with self.assertRaises(ValueError):
                cloud.validate_settings([11], settings)
        for value in ('../escape', '/absolute', 'a/b', ''):
            with self.assertRaises(ValueError):
                cloud.token(value)

    def test_host_rejects_external_missing_absolute_and_symlink_dependencies(self):
        for ref in ('../outside.a3m', '/absolute.a3m', 'missing.a3m'):
            (self.inputs/'input.yaml').write_text('sequences:\n- protein:\n    msa: ' + ref + '\n')
            with self.assertRaises(ValueError):
                host.inspect_inputs(self.inputs)
        (self.inputs/'input.yaml').write_bytes(self.yaml)
        (self.inputs/'link').symlink_to(self.checkpoint)
        with self.assertRaises(ValueError):
            host.inspect_inputs(self.inputs)

    def test_host_rejects_invalid_digest_expiry_and_output_overlap(self):
        for field, value in (('image', 'mutable:tag'), ('expires_in', 0), ('expires_in', 604801),
                             ('output', self.inputs/'nested-output')):
            args = deepcopy(self.args)
            setattr(args, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                host.prepare(args, self.store)
        self.assertFalse((self.inputs/'nested-output').exists())

    def test_http_bounded_download_and_unsafe_scheme(self):
        http = cloud.HTTP(int(time.time()) + 60)
        with self.assertRaises(cloud.TransportError):
            http.request(self.spec['urls']['input_get'], path=self.root/'bounded', max_bytes=1)
        with self.assertRaises(cloud.TransportError):
            http.request('file:///etc/passwd', path=self.root/'not-created')
        self.assertFalse((self.root/'not-created').exists())

    def test_second_prepare_has_isolated_job_and_attempt(self):
        self.args.output = self.root/'second'
        job = host.prepare(self.args, self.store)
        self.assertNotEqual(job['job_id'], self.job['job_id'])
        self.assertNotEqual(job['attempt_id'], self.job['attempt_id'])
        self.assertNotEqual(job['result_key'], self.job['result_key'])

    def test_host_upload_readback_failure_issues_no_capabilities(self):
        self.args.output = self.root/'bad-upload'
        with patch.object(self.store, 'get_object', return_value={'Body': io.BytesIO(b'corrupt')}):
            with self.assertRaisesRegex(ValueError, 'readback hash mismatch'):
                host.prepare(self.args, self.store)
        self.assertFalse((self.args.output/'runtime-env.secret.json').exists())
        self.assertFalse((self.args.output/'deployment.secret.json').exists())

    def test_r2_config_alias_and_non_executable_parsing(self):
        path = self.root/'unit-config'
        path.write_text('export AWS_ENDPOINT_URL="https://example.invalid"\nAWS_ACCESS_KEY="TEST-ID"\nAWS_SECRET_ACCESS_KEY="$(do-not-execute)" # comment\n')
        config = host.read_config(path)
        self.assertEqual(config['aws_access_key_id'], 'TEST-ID')
        self.assertEqual(config['aws_secret_access_key'], '$(do-not-execute)')
        path.write_text(json.dumps({'AWS_ENDPOINT_URL': 'https://example.invalid', 'AWS_ACCESS_KEY_ID': 'JSON-ID',
                                    'AWS_SECRET_ACCESS_KEY': 'TEST-SECRET'}))
        self.assertEqual(host.read_config(path)['aws_access_key_id'], 'JSON-ID')

    def test_sdk_signs_conditional_header_with_dummy_credentials_no_network(self):
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            self.skipTest('host-only cloud SDK not installed; CI installs requirements-cloud-host.txt')
        from urllib.parse import urlsplit, parse_qs
        client = boto3.client('s3', endpoint_url='https://example.invalid', region_name='auto',
                               aws_access_key_id='UNIT-TEST', aws_secret_access_key='UNIT-TEST',
                               config=Config(signature_version='s3v4'))
        self.assertIn('IfNoneMatch', client.meta.service_model.operation_model('PutObject').input_shape.members)
        url = client.generate_presigned_url('put_object', Params={'Bucket': 'test', 'Key': 'test', 'IfNoneMatch': '*'}, ExpiresIn=60)
        self.assertIn('if-none-match', parse_qs(urlsplit(url).query)['X-Amz-SignedHeaders'][0].split(';'))

    def test_private_payloads_gitignored(self):
        proc = subprocess.run(['git', 'check-ignore', '--no-index', 'runtime-env.secret.json', 'deployment.secret.json'],
                              cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(len(proc.stdout.splitlines()), 2)

    def test_sigterm_forwarded_to_worker_then_partial_uploaded(self):
        # Real wrapper CLI + subprocess signal. Fake worker handles boundary then exits143.
        self.worker.write_text(self.worker.read_text() + '''
import time
stop = False
def term(*_):
    global stop
    stop = True
signal.signal(signal.SIGTERM, term)
(out/'ready').write_text('ready')
while not stop:
    time.sleep(.02)
(out/'boundary-finished').write_text('done')
sys.exit(143)
''')
        env = dict(os.environ, COFOLD_CLOUD_SPEC=json.dumps(self.spec))
        proc = subprocess.Popen([sys.executable, str(ROOT/'cofolding/boltz2/cloud_job.py'),
                    '--work', str(self.root/'signal-work'), '--assets', str(self.assets),
                    '--checkpoint', str(self.checkpoint)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            while not (self.root/'signal-work/outputs/ready').exists():
                if proc.poll() is not None or time.monotonic() > deadline:
                    self.fail('fake worker did not become ready: ' + str(proc.communicate(timeout=2)))
                time.sleep(.02)
            proc.send_signal(signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=10)
            self.assertEqual(proc.returncode, 1, stderr)
            self.assertEqual(self.receipt()['status'], 'stopped')
            self.assertIn('outputs/boundary-finished', self.unpack_result())
            self.assertNotIn(b'TEST-CAPABILITY', stdout + stderr)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def archive(self, members):
        path = self.root/'malicious.tar'
        with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as tar:
            for name, kind in members:
                member = tarfile.TarInfo(name)
                member.type = kind
                member.linkname = '/etc/passwd'
                member.size = 1 if kind == tarfile.REGTYPE else 0
                tar.addfile(member, io.BytesIO(b'x') if member.size else None)
        return path

    def test_reject_traversal_absolute_links_devices_and_duplicates(self):
        for i, members in enumerate(([(name, tarfile.REGTYPE)] for name in
                                    ('/absolute', '../escape', 'inputs/../../escape', './inputs/a', 'inputs//a', 'inputs/./a'))):
            with self.subTest(members=members), self.assertRaises(ValueError):
                cloud.extract_inputs(self.archive(members), self.root/str(i))
        for i, kind in enumerate((tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE)):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                cloud.extract_inputs(self.archive([('inputs/x', kind)]), self.root/('special'+str(i)))
        with self.assertRaises(ValueError):
            cloud.extract_inputs(self.archive([('inputs/x', tarfile.REGTYPE)] * 2), self.root/'duplicate')

    def test_bounded_members_bytes_and_disk(self):
        archive = self.archive([('inputs/a', tarfile.REGTYPE), ('inputs/b', tarfile.REGTYPE)])
        with patch.object(cloud, 'MAX_MEMBERS', 1), self.assertRaises(ValueError):
            cloud.extract_inputs(archive, self.root/'count')
        with patch.object(cloud, 'MAX_INPUT_BYTES', 1), self.assertRaises(ValueError):
            cloud.extract_inputs(archive, self.root/'bytes')
        with patch.object(cloud.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 0})()), self.assertRaises(ValueError):
            cloud.extract_inputs(archive, self.root/'disk')

    def test_extended_headers_are_rejected_before_metadata_allocation(self):
        for i, kind in enumerate((tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_SPARSE)):
            path = self.root/'extended.tar'
            info = tarfile.TarInfo('inputs/header')
            info.type = kind
            info.size = 1024 ** 4  # No payload needed: reject header before reading.
            path.write_bytes(info.tobuf(format=tarfile.GNU_FORMAT))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                cloud.extract_inputs(path, self.root/('extended'+str(i)))

    def test_structure_only_assets_skip_affinity_keep_default_behavior(self):
        import fetch_assets
        for structure_only in (False, True):
            argv = ['fetch_assets.py', '--cache', str(self.root)] + (['--structure-only'] if structure_only else [])
            with patch.object(sys, 'argv', argv), patch.object(fetch_assets, 'verify') as verify, \
                 patch.object(fetch_assets, 'mols_ready', return_value=True), patch('builtins.print'):
                fetch_assets.main()
            names = {c.args[0].name for c in verify.call_args_list}
            self.assertEqual(names, {'boltz2_conf.ckpt', 'mols.tar'} | (set() if structure_only else {'boltz2_aff.ckpt'}))

    def test_asset_downloader_and_native_defaults_are_packaged_without_default_override(self):
        docker = (ROOT/'cofolding/boltz2/Dockerfile').read_text()
        self.assertIn('COPY cloud_job.py fetch_assets.py assets.json /opt/cofold/', docker)
        self.assertIn('ENTRYPOINT ["cofold-run"]', docker)
        self.assertIn('CMD ["--help"]', docker)
        ignore = (ROOT/'cofolding/boltz2/.dockerignore').read_text().splitlines()
        for name in ('cloud_job.py', 'cofold-cloud-run', 'fetch_assets.py', 'assets.json'):
            self.assertIn('!' + name, ignore)


if __name__ == '__main__':
    unittest.main()

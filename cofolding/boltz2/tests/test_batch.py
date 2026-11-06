"""CPU-only contract/lifecycle tests; Docker/GPU/model calls are mocked.

All molecular inputs come from the one common canonical case. No prediction
outputs or historical fixtures are bundled or traversed by this suite.
Run: /usr/bin/python3 -m unittest discover -s cofolding/boltz2/tests -v
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import Mock, patch, call

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


batch = load('boltz_batch', ROOT/'batch.py')
runtime = load('boltz_runtime', ROOT/'batch_native/runtime.py')
IMAGE_ID = 'sha256:'+'a'*64
GPU = 'GPU-01234567-89ab-cdef-0123-456789abcdef'


def shared():
    case = json.loads((ROOT.parent/'test-cases/trpcage-ethanol/input.json').read_text())
    return {'version': 1, 'sequences': [
        {'protein': dict(case['protein'], msa=case['msa'])},
        {'ligand': case['ligand']} ]}


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.indir = self.root/'input'
        self.indir.mkdir()
        self.source = self.indir/'shared.yaml'
        self.write(shared())
        self.cache = self.root/'cache'
        (self.cache/'mols').mkdir(parents=True)
        self.checkpoint = self.root/'structure.ckpt'
        # Operational test bytes, not model weights or a molecular output.
        self.checkpoint.write_bytes(b'checkpoint identity test')
        self.out = self.root/'output'
        self.argv = ['batch.py', '--input', str(self.indir), '--output', str(self.out),
                     '--cache', str(self.cache), '--checkpoint', str(self.checkpoint),
                     '--gpu', GPU, '--lock-root', str(self.root/'locks')]

    def write(self, value):
        self.source.write_text(yaml.safe_dump(value))

    def invoke(self, extra=(), owned=None, image=IMAGE_ID):
        def gpu_query(cmd, **kwargs):
            return '' if '--query-compute-apps=pid' in cmd else '0, 0'
        with patch.object(sys, 'argv', self.argv+list(extra)), \
             patch.object(batch, 'resolve_image', return_value=image), \
             patch.object(batch.subprocess, 'check_output', side_effect=gpu_query), \
             patch.object(batch, 'run_owned', side_effect=owned) as runner, \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            batch.main()
        return stdout.getvalue(), runner

    def fake_owned(self, cmd, log, timeout):
        seed = int(cmd[-1])
        if cmd[-2] == '/scripts/verify.py':
            (self.out/f'seed-{seed}/verified.json').write_text('{"status":"PASS"}')

    def test_shared_case_exactly_matches_canonical(self):
        self.assertEqual(yaml.safe_load((ROOT/'examples/shared.yaml').read_text()), shared())
        records, files = batch.inputs(ROOT/'examples')
        self.assertEqual(records, [{'id': 'shared', 'yaml': shared()}])
        self.assertEqual(set(files), {'sources/shared.yaml'})

    def test_default_image(self):
        self.assertEqual(batch.IMAGE, 'container-library/boltz2:2.2.1')

    def test_image_tag_digest_and_id_resolve_locally(self):
        for reference in (batch.IMAGE, 'ghcr.io/aetherark/boltz2@sha256:'+'b'*64, IMAGE_ID):
            with self.subTest(reference=reference), patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, IMAGE_ID+'\n', '')) as run:
                self.assertEqual(batch.resolve_image(reference), IMAGE_ID)
                self.assertEqual(run.call_args.args[0], ['docker', 'image', 'inspect', '--format', '{{.Id}}', reference])

    def test_missing_image_requires_explicit_pull(self):
        with patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, '', 'missing')) as run:
            with self.assertRaisesRegex(ValueError, 'Pull first: docker pull'):
                batch.resolve_image(batch.IMAGE)
            self.assertEqual(run.call_count, 1)

    def test_image_inspect_invalid_identity_rejected(self):
        with patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'tag', '')):
            with self.assertRaisesRegex(ValueError, 'immutable'):
                batch.resolve_image(batch.IMAGE)
        for reference in ('--help', '', 'bad image'):
            with self.assertRaises(ValueError):
                batch.resolve_image(reference)

    def test_normal_path_resolution_spaces_relative_and_symlink(self):
        target = self.root/'some directory'
        target.mkdir()
        (self.root/'link').symlink_to(target, target_is_directory=True)
        self.assertEqual(batch.safe_path(self.root/'link/../some directory/file'), target/'file')
        with patch.dict(os.environ, {'HOME': str(self.root)}):
            self.assertEqual(batch.safe_path('~/some directory'), target)
        self.assertEqual(batch.mount_path(target), target)
        for name in ('name:ro', 'name\nother'):
            with self.assertRaisesRegex(ValueError, 'mount paths'):
                batch.mount_path(self.root/name)

    def test_symlink_loop_fails(self):
        (self.root/'loop').symlink_to('loop')
        with self.assertRaises((RuntimeError, OSError)):
            batch.safe_path(self.root/'loop')

    def test_portable_lock_defaults_and_overrides(self):
        with patch.dict(os.environ, {'HOME': str(self.root)}, clear=True):
            self.assertEqual(batch.lock_root(), self.root/'.local/state/cofolding/gpu-locks')
            with patch.dict(os.environ, {'XDG_STATE_HOME': str(self.root/'state')}):
                self.assertEqual(batch.lock_root(), self.root/'state/cofolding/gpu-locks')
            with patch.dict(os.environ, {'COFOLD_GPU_LOCK_ROOT': str(self.root/'common')}):
                self.assertEqual(batch.lock_root(), self.root/'common')
                self.assertEqual(batch.lock_root(self.root/'explicit'), self.root/'explicit')

    def test_lock_excludes_second_owner_before_output_creation(self):
        lockdir = self.root/'locks'
        lockdir.mkdir()
        with (lockdir/(GPU+'.lock')).open('a') as owner:
            batch.fcntl.flock(owner, batch.fcntl.LOCK_EX | batch.fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.invoke()
        self.assertFalse(self.out.exists())

    def test_reject_unsupported_and_malformed_shared_variants(self):
        variants = []
        for key in ('templates', 'constraints', 'properties'):
            data = shared(); data[key] = []; variants.append(data)
        for field, value in (('sequence', 'X'), ('msa', None), ('id', ['A'])):
            data = shared(); data['sequences'][0]['protein'][field] = value; variants.append(data)
        data = shared(); data['sequences'][1]['ligand']['smiles'] = ''; variants.append(data)
        data = shared(); data['sequences'][1]['ligand']['id'] = data['sequences'][0]['protein']['id']; variants.append(data)
        data = shared(); data['sequences'].append(copy.deepcopy(data['sequences'][0])); variants.append(data)
        data = shared(); data['version'] = True; variants.append(data)
        data = shared(); data['sequences'][1] = {'ligand': {'id': 'L', 'ccd': 'ETH'}}; variants.append(data)
        for data in variants:
            with self.subTest(data=data):
                self.write(data)
                owned = Mock()
                with self.assertRaises(ValueError):
                    self.invoke(owned=owned)
                owned.assert_not_called()
                self.assertFalse(self.out.exists())

    def test_duplicate_yaml_key_rejected(self):
        self.source.write_text('version: 1\n'+yaml.safe_dump(shared()))
        with self.assertRaisesRegex(ValueError, 'Duplicate YAML key'):
            batch.inputs(self.indir)

    def test_non_yaml_duplicate_record_and_unsafe_id_rejected(self):
        for name in ('extra.txt', 'shared.yml', 'unsafe.name.yaml'):
            p = self.indir/name; p.write_text(self.source.read_text())
            with self.subTest(name=name), self.assertRaises(ValueError):
                batch.inputs(self.indir)
            p.unlink()

    def test_local_msa_staging_is_content_bound(self):
        msa = self.root/'local.a3m'
        sequence = shared()['sequences'][0]['protein']['sequence']
        msa.write_text('>query\n'+sequence+'\n')
        data = shared(); data['sequences'][0]['protein']['msa'] = '../local.a3m'; self.write(data)
        records, files = batch.inputs(self.indir)
        key = 'msa/'+batch.sha(msa)+'.a3m'
        self.assertEqual(records[0]['yaml']['sequences'][0]['protein']['msa'], '/run/'+key)
        self.assertEqual(files[key][0], msa)
        msa.write_text('>renamed_query\n'+sequence+'\n')
        self.assertNotEqual(set(files), set(batch.inputs(self.indir)[1]))

    def test_invalid_schedule_and_limits_do_not_launch(self):
        for flags in (['--seeds', '0', '0'], ['--seeds', '-1'], ['--diffusion_samples', '0'],
                      ['--recycling_steps', '-1'], ['--timeout', '0'], ['--threads', '-1'],
                      ['--cpus', 'nan'], ['--verify-cpus', '0'], ['--memory', '0g'],
                      ['--verify-memory', 'unbounded'], ['--gpu', '0']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.invoke(flags)
        self.assertFalse(self.out.exists())

    def test_offline_command_limits_and_immutable_identity(self):
        text, runner = self.invoke(['--dry-run', '--cpus', '3.5', '--memory', '12g', '--threads', '3'])
        runner.assert_not_called()
        self.assertFalse(self.out.exists())
        report = json.loads(text)
        self.assertEqual(report['request']['settings']['image'], IMAGE_ID)
        cmd = report['commands'][0]
        for flag, value in (('--pull', 'never'), ('--network', 'none'), ('--cpus', '3.5'), ('--memory', '12g'), ('--memory-swap', '12g')):
            self.assertEqual(cmd[cmd.index(flag)+1], value)
        self.assertIn('OMP_NUM_THREADS=3', cmd)
        self.assertEqual(cmd[-3:], [IMAGE_ID, '/scripts/observe.py', '0'])

    def test_output_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Output must be separate'):
            self.invoke(['--output', str(self.cache/'run')])

    def test_no_unused_affinity_or_default_checkpoint_required(self):
        runtime.offline_assets(self.cache, self.checkpoint)
        text, _ = self.invoke(['--dry-run'])
        self.assertEqual(json.loads(text)['models'], 1)
        self.assertEqual(list(self.cache.iterdir()), [self.cache/'mols'])
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            runtime.offline_assets(self.cache, self.root/'missing')

    def test_native_args_keep_full_outputs_explicit_seed_and_no_server(self):
        settings = dict(diffusion_samples=1, recycling_steps=3, sampling_steps=200,
                        max_parallel_samples=1, no_kernels=True, threads=4)
        cmd = runtime.native_args(Path('/run/seed-0'), 0, settings)
        self.assertEqual(cmd[:2], ['predict', '/run/inputs'])
        for flag in ('--write_full_pae', '--write_full_pde', '--write_embeddings', '--no_kernels'):
            self.assertIn(flag, cmd)
        self.assertNotIn('--use_msa_server', cmd)
        self.assertEqual(cmd[cmd.index('--preprocessing-threads')+1], '4')
        self.assertEqual(cmd[cmd.index('--checkpoint')+1], '/checkpoint')

    def test_identical_resume_verifies_without_inference_or_byte_changes(self):
        _, runner = self.invoke(owned=self.fake_owned)
        self.assertEqual([c.args[0][-2] for c in runner.call_args_list], ['/scripts/observe.py', '/scripts/verify.py'])
        before = {str(p.relative_to(self.out)):p.read_bytes() for p in self.out.rglob('*') if p.is_file()}
        _, runner = self.invoke(['--resume', '--image', 'registry.example/boltz:tag'], owned=self.fake_owned)
        self.assertEqual([c.args[0][-2] for c in runner.call_args_list], ['/scripts/verify.py'])
        after = {str(p.relative_to(self.out)):p.read_bytes() for p in self.out.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_changed_image_checkpoint_source_or_settings_refuse_resume(self):
        self.invoke(owned=self.fake_owned)
        with self.assertRaisesRegex(ValueError, 'request differs'):
            self.invoke(['--resume'], image='sha256:'+'b'*64)
        with self.assertRaisesRegex(ValueError, 'request differs'):
            self.invoke(['--resume', '--sampling_steps', '201'])
        self.checkpoint.write_bytes(b'changed checkpoint identity')
        with self.assertRaisesRegex(ValueError, 'request differs'):
            self.invoke(['--resume'])
        self.checkpoint.write_bytes(b'checkpoint identity test')
        self.source.write_text(self.source.read_text()+'\n# changed source bytes\n')
        with self.assertRaisesRegex(ValueError, 'request differs'):
            self.invoke(['--resume'])

    def test_removed_staged_code_and_changed_staged_source_refuse_resume(self):
        self.invoke(owned=self.fake_owned)
        code = self.out/'code/runtime.py'; original = code.read_bytes(); code.unlink()
        with self.assertRaisesRegex(ValueError, 'Staged code changed'):
            self.invoke(['--resume'])
        code.write_bytes(original)
        (self.out/'sources/shared.yaml').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Staged source changed'):
            self.invoke(['--resume'])

    def test_interrupted_seed_is_preserved_and_refused(self):
        def fail(cmd, log, timeout):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(owned=fail)
        self.assertTrue((self.out/'seed-0').is_dir())
        with self.assertRaisesRegex(ValueError, 'Interrupted seed preserved'):
            self.invoke(['--resume'])

    def test_one_directory_process_per_seed_not_per_record(self):
        (self.indir/'same-case.yaml').write_bytes(self.source.read_bytes())
        _, runner = self.invoke(['--seeds', '0', '1'], owned=self.fake_owned)
        commands = [c.args[0] for c in runner.call_args_list]
        self.assertEqual([(c[-2],c[-1]) for c in commands], [('/scripts/observe.py','0'),('/scripts/verify.py','0'),('/scripts/observe.py','1'),('/scripts/verify.py','1')])
        self.assertEqual(len(json.loads((self.out/'request.json').read_text())['records']), 2)


class OwnedLifecycleTests(unittest.TestCase):
    def exercise(self, waits, stop_error=False, running=False):
        proc = Mock(); proc.wait.side_effect = waits
        calls = []
        def run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[1] == 'stop' and stop_error:
                raise subprocess.TimeoutExpired(cmd, 30)
            return subprocess.CompletedProcess(cmd, 0, 'true' if running else 'false', '')
        handlers = {s:signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        with patch.object(batch.subprocess, 'Popen', return_value=proc) as popen, patch.object(batch.subprocess, 'run', side_effect=run):
            try:
                batch.run_owned(['docker','run',IMAGE_ID,'/scripts/observe.py','0'], io.StringIO(), 10)
            finally:
                self.assertEqual(handlers, {s:signal.getsignal(s) for s in handlers})
                command = popen.call_args.args[0]
                name = command[command.index('--name')+1]
                self.assertTrue(name.startswith('boltz-batch-'))
                self.assertTrue(calls)
                for cmd in calls:
                    self.assertEqual(cmd[-1], name)
        return calls, proc

    def test_normal_exit_stops_only_owned_container(self):
        calls, _ = self.exercise([0, 0])
        self.assertEqual([c[1] for c in calls], ['stop', 'inspect'])

    def test_timeout_and_interrupt_clean_up(self):
        for error in (subprocess.TimeoutExpired('test', 10), KeyboardInterrupt()):
            with self.subTest(error=error), self.assertRaises(type(error)):
                self.exercise([error, 0])

    def test_nonzero_exit_is_not_success(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.exercise([1, 0])

    def test_stuck_container_and_stop_timeout_get_killed(self):
        calls, _ = self.exercise([0, 0], running=True)
        self.assertIn('kill', [c[1] for c in calls])
        calls, _ = self.exercise([0, 0], stop_error=True)
        self.assertIn('kill', [c[1] for c in calls])


class VerifierInvalidationTests(unittest.TestCase):
    def test_corrupt_prior_payload_or_request_cannot_retain_pass(self):
        # Import only the verifier's stdlib boundary. No scientific calculation is
        # simulated or asserted here; failure precedes every tensor/graph read.
        def require(test, label):
            if not test: raise ValueError(label)
        stubs = {name:types.ModuleType(name) for name in ('gemmi', 'numpy', 'torch', 'rdkit', 'graph')}
        setattr(stubs['rdkit'], 'Chem', Mock())
        setattr(stubs['graph'], 'graph_check', Mock())
        setattr(stubs['graph'], 'require', require)
        stubs['runtime'] = runtime
        with patch.dict(sys.modules, stubs):
            verifier = load('isolated_verifier', ROOT/'batch_native/verify.py')
        for corrupt in ('payload', 'request', 'receipt'):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); stage = root/'seed-0'; stage.mkdir()
                payload = stage/'payload'; payload.write_bytes(b'original')
                saved = stage/'verified.json'
                saved.write_text(json.dumps({'status':'PASS', 'sha256':{'payload':batch.sha(payload)}}))
                (root/'request.json').write_text('invalid JSON')
                if corrupt == 'payload': payload.write_bytes(b'corrupted')
                if corrupt == 'receipt': saved.write_text('invalid JSON')
                with self.assertRaises((ValueError, KeyError)):
                    verifier.verify(root, 0)
                self.assertFalse(saved.exists())


class AssetTests(unittest.TestCase):
    def test_partial_directory_cannot_report_ready(self):
        assets = load('boltz_assets', ROOT/'fetch_assets.py')
        for download in (False, True):
            with self.subTest(download=download), tempfile.TemporaryDirectory() as tmp:
                cache = Path(tmp); (cache/'mols').mkdir()
                argv = ['fetch_assets.py', '--cache', str(cache)] + (['--download'] if download else [])
                manifest = json.loads((ROOT/'assets.json').read_text())
                for name in manifest['files']:
                    (cache/name).touch()
                with patch.object(sys, 'argv', argv), patch.object(assets, 'verify'), patch('builtins.print') as output:
                    with self.assertRaisesRegex(ValueError, 'Unverified/incomplete'):
                        assets.main()
                    self.assertNotIn(call('Assets ready'), output.call_args_list)

    def test_interrupted_extraction_is_not_published_and_retry_succeeds(self):
        assets = load('boltz_assets', ROOT/'fetch_assets.py')
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)/'archive.tar'; cache = Path(tmp)/'cache'; cache.mkdir()
            source = (ROOT/'examples/shared.yaml').read_bytes()
            with tarfile.open(archive, 'w') as tar:
                member = tarfile.TarInfo('mols/shared.yaml'); member.size = len(source)
                tar.addfile(member, io.BytesIO(source))
            def interrupted(destination, **kwargs):
                (destination/'mols').mkdir()
                (destination/'mols/shared.yaml').write_bytes(source[:1])
                raise OSError('simulated disk failure')
            with patch.object(tarfile.TarFile, 'extractall', side_effect=interrupted):
                with self.assertRaisesRegex(OSError, 'simulated disk failure'):
                    assets.extract(archive, cache)
            self.assertFalse((cache/'mols').exists())
            self.assertEqual(list(cache.iterdir()), [])
            assets.extract(archive, cache)
            self.assertTrue(assets.mols_ready(cache, batch.sha(archive)))
            self.assertFalse(assets.mols_ready(cache, '0'*64))
            self.assertEqual((cache/'mols/shared.yaml').read_bytes(), source)

    def test_verify_rejects_missing_wrong_size_and_wrong_hash(self):
        assets = load('boltz_assets', ROOT/'fetch_assets.py')
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'shared.yaml'
            source = (ROOT/'examples/shared.yaml').read_bytes()
            spec = {'bytes': len(source), 'sha256': batch.sha(ROOT/'examples/shared.yaml')}
            with self.assertRaisesRegex(ValueError, 'Missing or incorrect size'):
                assets.verify(target, spec)
            target.write_bytes(source)
            assets.verify(target, spec)
            target.write_bytes(source+b'\n')
            with self.assertRaisesRegex(ValueError, 'incorrect size'):
                assets.verify(target, spec)
            target.write_bytes(b'!'+source[1:])
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                assets.verify(target, spec)

    def test_extract_refuses_traversal_absolute_links_and_devices(self):
        assets = load('boltz_assets', ROOT/'fetch_assets.py')
        for name, kind in (('mols/../escape', tarfile.REGTYPE),
                           ('/mols/absolute', tarfile.REGTYPE),
                           ('other/file', tarfile.REGTYPE),
                           ('mols/symlink', tarfile.SYMTYPE),
                           ('mols/hardlink', tarfile.LNKTYPE),
                           ('mols/device', tarfile.CHRTYPE)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp)/'archive.tar'
                cache = Path(tmp)/'cache'; cache.mkdir()
                with tarfile.open(archive, 'w') as tar:
                    member = tarfile.TarInfo(name); member.type = kind
                    if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                        member.linkname = '../escape'
                    tar.addfile(member)
                with self.assertRaisesRegex(ValueError, 'Unsafe archive member'):
                    assets.extract(archive, cache)
                self.assertEqual(list(cache.iterdir()), [])

    def test_extract_regular_shared_bytes(self):
        assets = load('boltz_assets', ROOT/'fetch_assets.py')
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)/'archive.tar'; cache = Path(tmp)/'cache'; cache.mkdir()
            source = (ROOT/'examples/shared.yaml').read_bytes()
            with tarfile.open(archive, 'w') as tar:
                member = tarfile.TarInfo('mols/shared.yaml'); member.size = len(source)
                tar.addfile(member, io.BytesIO(source))
            assets.extract(archive, cache)
            self.assertEqual((cache/'mols/shared.yaml').read_bytes(), source)


if __name__ == '__main__':
    unittest.main()

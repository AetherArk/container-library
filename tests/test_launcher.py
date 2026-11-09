"""Unit tests: subprocess mocks only, no Docker, checkpoint or GPU inference."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('launcher', Path(__file__).resolve().parents[1] / 'run_cofold.py')
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
IMAGE = 'sha256:' + 'a' * 64
GPU = 'GPU-01234567-89ab-cdef-0123-456789abcdef'


class LauncherUnitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('inputs', 'assets'):
            (self.root / name).mkdir()
        for name in ('one.yaml', 'two.yaml'):
            (self.root / 'inputs' / name).write_text('version: 1\n')
        (self.root / 'weights.any').write_bytes(b'unit fixture, not a checkpoint')
        # The memory preflight reads the real host; pin it so launcher tests are portable.
        memory = patch.object(launcher, 'available_memory_bytes', return_value=64 * 1024 ** 3)
        memory.start()
        self.addCleanup(memory.stop)
        self.argv = ['--image', 'candidate:local', '--inputs', str(self.root / 'inputs'),
                     '--assets', str(self.root / 'assets'), '--checkpoint', str(self.root / 'weights.any'),
                     '--output', str(self.root / 'results'), '--gpu', GPU, '--seeds', '11', '12',
                     '--lock-root', str(self.root / 'locks')]

    def test_checkpoint_may_be_a_directory(self):
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        argv = list(self.argv)
        argv[argv.index('--checkpoint') + 1] = str(model)
        args = launcher.parser().parse_args(argv)
        launcher.validate(args)
        launcher.command(args, 'sha256:' + 'a' * 64, 'token')
        self.assertTrue(model.is_dir())

    def test_input_format_is_owned_by_selected_worker(self):
        for file in (self.root / 'inputs').iterdir():
            file.unlink()
        (self.root / 'inputs' / 'native.json').write_text('{}')
        args = launcher.parser().parse_args(self.argv)
        launcher.validate(args)
        self.assertEqual(args.inputs, self.root / 'inputs')

    def test_one_launch_all_files_and_seeds_literal_options(self):
        options = ['--samples', '1', '--recycles', '3', '--steps', '200', '--max-parallel-samples', '1',
                   '--no-kernels', '--use-potentials', '--output-format', 'mmcif', '--capture', 'none',
                   '--resume']
        commands = []
        def popen(cmd):
            commands.append(cmd)
            (self.root / 'results/run.json').write_text(json.dumps({'status': 'complete'}))
            return Mock(poll=Mock(return_value=0), wait=Mock(return_value=0))
        with patch.object(launcher, 'resolve_image', return_value=IMAGE), patch.object(launcher, 'gpu_available'), \
                patch.object(launcher.subprocess, 'Popen', side_effect=popen):
            self.assertEqual(launcher.main(self.argv + options), 0)
        self.assertEqual(len(commands), 1)
        cmd = commands[0]
        self.assertEqual(cmd[:2], ['docker', 'run'])
        self.assertEqual(cmd[cmd.index(IMAGE) + 1:], ['--input', '/inputs', '--output', '/outputs',
                         '--assets', '/models', '--checkpoint', '/checkpoint', '--seeds', '11', '12'] + options)
        for flag in ('--pull=never', '--network=none'):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index('--entrypoint') + 1], 'cofold-run')
        self.assertEqual(cmd[cmd.index('--gpus') + 1], 'device=' + GPU)
        self.assertEqual(cmd[cmd.index('--user') + 1], f'{os.getuid()}:{os.getgid()}')
        mounts = [cmd[i+1] for i, v in enumerate(cmd) if v == '--mount']
        self.assertEqual(len(mounts), 4)
        self.assertEqual(sum(m.endswith(',readonly') for m in mounts), 3)
        self.assertTrue(all('src=' + str(self.root) + '/' in m for m in mounts))
        self.assertFalse(any('/workspace' in x for x in mounts))
        self.assertNotIn(str(spec.origin), ' '.join(mounts))

    def test_zero_exit_without_current_terminal_receipt_fails(self):
        with patch.object(launcher, 'resolve_image', return_value=IMAGE), patch.object(launcher, 'gpu_available'), \
                patch.object(launcher, 'run_worker', return_value=0):
            self.assertEqual(launcher.main(self.argv), 1)
            (self.root / 'results/run.json').write_text('{"status":"complete"}')
            self.assertEqual(launcher.main(self.argv), 1)  # stale success is not current

    def test_worker_nonzero_preserved(self):
        with patch.object(launcher, 'resolve_image', return_value=IMAGE), patch.object(launcher, 'gpu_available'), \
                patch.object(launcher, 'run_worker', return_value=42):
            self.assertEqual(launcher.main(self.argv), 42)

    def test_dry_run_does_not_launch_or_create_output(self):
        with patch.object(launcher, 'resolve_image', return_value=IMAGE), patch.object(launcher.subprocess, 'Popen') as p:
            self.assertEqual(launcher.main(self.argv + ['--dry-run']), 0)
            p.assert_not_called()
        self.assertFalse((self.root / 'results').exists())

    def test_image_must_resolve_locally_to_immutable_id(self):
        with patch.object(launcher.subprocess, 'run', return_value=Mock(returncode=0, stdout=IMAGE, stderr='')) as run:
            self.assertEqual(launcher.resolve_image('candidate:local'), IMAGE)
            self.assertEqual(run.call_args.args[0], ['docker', 'image', 'inspect', '--format', '{{.Id}}', 'candidate:local'])
        for value in ('--privileged', 'bad image'):
            with self.assertRaises(ValueError):
                launcher.resolve_image(value)
        with patch.object(launcher, 'checked', return_value='mutable:tag'):
            with self.assertRaises(ValueError):
                launcher.resolve_image('candidate:local')

    def test_path_gpu_and_resource_safety(self):
        for extra in (['--gpu', '0'], ['--output', str(self.root / 'assets/out')],
                      ['--checkpoint', str(self.root / 'missing')], ['--cpus', 'nan'],
                      ['--timeout', '0'], ['--memory', '--privileged']):
            with self.subTest(extra=extra), patch.object(launcher, 'resolve_image') as image:
                try:
                    rc = launcher.main(self.argv + extra)
                except SystemExit as exc:
                    rc = exc.code
                self.assertEqual(rc, 2)
                image.assert_not_called()
        with self.assertRaises(ValueError):
            launcher.mount_path(self.root / 'path:ro')
        with self.assertRaises(ValueError):
            launcher.mount_path(self.root / 'path,readonly')

    def test_gpu_occupied_rejected(self):
        with patch.object(launcher, 'checked', side_effect=[GPU, '1234']):
            with self.assertRaisesRegex(ValueError, 'active compute'):
                launcher.gpu_available(GPU)

    def test_only_owned_valid_cid_can_be_signalled(self):
        cidfile = self.root / 'owned.cid'
        with patch.object(launcher.subprocess, 'run') as run:
            self.assertFalse(launcher.owned_signal(cidfile, 'TERM'))
            cidfile.write_text('other-container')
            self.assertFalse(launcher.owned_signal(cidfile, 'TERM'))
            run.assert_not_called()
            cidfile.write_text('b' * 64)
            self.assertTrue(launcher.owned_signal(cidfile, 'TERM'))
            self.assertEqual(run.call_args.args[0], ['docker', 'kill', '--signal', 'TERM', 'b' * 64])

    def test_interrupt_term_then_bounded_owned_kill_no_restart(self):
        handlers = {}
        def register(sig, fn):
            old = handlers.get(sig, signal.SIG_DFL)
            handlers[sig] = fn
            return old
        process = Mock()
        def popen(cmd):
            handlers[signal.SIGINT](signal.SIGINT, None)
            return process
        process.poll.side_effect = [None, None, 137]
        process.wait.return_value = 137
        with patch.object(launcher.signal, 'signal', side_effect=register), \
                patch.object(launcher.subprocess, 'Popen', side_effect=popen) as p, \
                patch.object(launcher, 'owned_signal', return_value=True) as kill, \
                patch.object(launcher.time, 'monotonic', side_effect=[0, 1, 4]), patch.object(launcher.time, 'sleep'):
            self.assertEqual(launcher.run_worker(['docker', 'run'], self.root/'owned.cid', None, 2), 137)
        self.assertEqual(p.call_count, 1)
        self.assertEqual([c.args[1] for c in kill.call_args_list], ['TERM', 'KILL'])
        self.assertEqual(handlers[signal.SIGINT], signal.SIG_DFL)

    def test_receipt_status_fail_closed_without_full_schema(self):
        path = self.root / 'run.json'
        for data in ({'status': 'running'}, {}, [], {'status': 'unknown'}):
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                launcher.terminal_receipt(path)
        for status in ('failed', 'fatal', 'stopped'):
            path.write_text(json.dumps({'status': status}))
            self.assertFalse(launcher.terminal_receipt(path))
        path.write_text('{"status":"completed"}')
        self.assertTrue(launcher.terminal_receipt(path))

    def test_shared_gpu_lock_prevents_docker_launch(self):
        lockroot = self.root / 'locks'
        lockroot.mkdir()
        with (lockroot / (GPU + '.lock')).open('a') as lock:
            launcher.fcntl.flock(lock, launcher.fcntl.LOCK_EX | launcher.fcntl.LOCK_NB)
            with patch.object(launcher, 'resolve_image', return_value=IMAGE), \
                    patch.object(launcher.subprocess, 'Popen') as popen:
                self.assertEqual(launcher.main(self.argv), 2)
                popen.assert_not_called()

    def test_second_launcher_on_the_same_output_directory_exits_2(self):
        runtime = self.root / 'results' / '.runtime'
        runtime.mkdir(parents=True)
        with (runtime / 'launcher.lock').open('a') as lock:
            launcher.fcntl.flock(lock, launcher.fcntl.LOCK_EX | launcher.fcntl.LOCK_NB)
            with patch.object(launcher, 'resolve_image', return_value=IMAGE), \
                    patch.object(launcher, 'gpu_available'), \
                    patch.object(launcher.subprocess, 'Popen') as popen:
                self.assertEqual(launcher.main(self.argv), 2)
                popen.assert_not_called()

    def test_preflight_rejects_an_estimate_above_free_disk(self):
        import contextlib
        import io
        stderr = io.StringIO()
        with patch.object(launcher, 'resolve_image', return_value=IMAGE), \
                patch.object(launcher, 'gpu_available'), \
                patch.object(launcher.subprocess, 'Popen') as popen, \
                patch.object(launcher.shutil, 'disk_usage', return_value=Mock(free=100)), \
                contextlib.redirect_stderr(stderr):
            self.assertEqual(launcher.main(self.argv + ['--estimate-attempt-bytes', '100']), 2)
        popen.assert_not_called()
        message = stderr.getvalue()
        self.assertIn('preflight', message)
        self.assertIn('100 B/attempt x 4 jobs', message)

    def test_preflight_rejects_memory_above_host_available(self):
        import contextlib
        import io
        stderr = io.StringIO()
        with patch.object(launcher, 'resolve_image', return_value=IMAGE), \
                patch.object(launcher, 'gpu_available'), \
                patch.object(launcher.subprocess, 'Popen') as popen, \
                patch.object(launcher, 'available_memory_bytes', return_value=1024 ** 3), \
                contextlib.redirect_stderr(stderr):
            self.assertEqual(launcher.main(self.argv + ['--memory', '2g']), 2)
        popen.assert_not_called()
        message = stderr.getvalue()
        self.assertIn('preflight', message)
        self.assertIn('2g', message)
        self.assertIn(str(1024 ** 3), message)

    def test_preflight_without_an_estimate_records_free_space_and_skips_comparison(self):
        import contextlib
        import io
        stdout = io.StringIO()
        def popen(cmd):
            (self.root / 'results/run.json').write_text(json.dumps({'schema': 1, 'status': 'complete'}))
            return Mock(poll=Mock(return_value=0), wait=Mock(return_value=0))
        with patch.object(launcher, 'resolve_image', return_value=IMAGE), \
                patch.object(launcher, 'gpu_available'), \
                patch.object(launcher.subprocess, 'Popen', side_effect=popen), \
                patch.object(launcher.shutil, 'disk_usage', return_value=Mock(free=123456789)), \
                contextlib.redirect_stdout(stdout):
            self.assertEqual(launcher.main(self.argv), 0)
        preflight = json.loads((self.root / 'results/run.json').read_text())['preflight']
        self.assertIsNone(preflight['estimate_attempt_bytes'])
        self.assertEqual(preflight['free_bytes_output'], 123456789)
        self.assertEqual(preflight['disk_check'], 'skipped_no_estimate')
        self.assertIn('comparison skipped', stdout.getvalue())

    def test_preflight_records_estimate_and_measured_free_space(self):
        def popen(cmd):
            (self.root / 'results/run.json').write_text(json.dumps({'schema': 1, 'status': 'complete'}))
            return Mock(poll=Mock(return_value=0), wait=Mock(return_value=0))
        with patch.object(launcher, 'resolve_image', return_value=IMAGE), \
                patch.object(launcher, 'gpu_available'), \
                patch.object(launcher.subprocess, 'Popen', side_effect=popen), \
                patch.object(launcher.shutil, 'disk_usage', return_value=Mock(free=10 ** 12)):
            self.assertEqual(launcher.main(self.argv + ['--estimate-attempt-bytes', '1000']), 0)
        preflight = json.loads((self.root / 'results/run.json').read_text())['preflight']
        self.assertEqual(preflight['estimate_attempt_bytes'], 1000)
        self.assertEqual(preflight['estimated_total_bytes'], 4000)
        self.assertEqual(preflight['free_bytes_output'], 10 ** 12)
        self.assertEqual(preflight['disk_check'], 'compared')

    def test_launcher_flags_match_the_shared_cli_contract(self):
        contract = json.loads((Path(__file__).resolve().parents[1] / 'cofolding/cli-contract.json').read_text())
        parsed = launcher.parser()
        for flag, spec in contract['flags'].items():
            action = parsed._option_string_actions.get(flag)
            if action is None:
                continue
            with self.subTest(flag=flag):
                self.assertEqual(action.required, spec['required'])
                if spec['type'] == 'enum':
                    self.assertEqual(list(action.choices), list(spec['domain']))

    def test_timeout_terminates_owned_container_and_returns_124(self):
        process = Mock(poll=Mock(side_effect=[None, None, 137]), wait=Mock(return_value=137))
        with patch.object(launcher.subprocess, 'Popen', return_value=process) as p, \
                patch.object(launcher, 'owned_signal', return_value=True) as kill, \
                patch.object(launcher.time, 'monotonic', side_effect=[0, 5, 10]), patch.object(launcher.time, 'sleep'):
            self.assertEqual(launcher.run_worker(['docker', 'run'], self.root/'owned.cid', 3, 2), 124)
        self.assertEqual(p.call_count, 1)
        self.assertEqual([c.args[1] for c in kill.call_args_list], ['TERM', 'KILL'])

    def test_local_harness_image_cpu_only_no_source_overlay(self):
        harness_spec = importlib.util.spec_from_file_location('test_local_harness', Path(__file__).resolve().parents[1] / 'scripts/test_local.py')
        assert harness_spec is not None and harness_spec.loader is not None
        harness = importlib.util.module_from_spec(harness_spec)
        harness_spec.loader.exec_module(harness)
        with patch.object(harness.launcher, 'resolve_image', return_value=IMAGE), patch.object(harness, 'run') as run:
            self.assertEqual(harness.main(['--suite', 'image', '--image', 'candidate:local']), 0)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            cmd = call.args[0]
            self.assertIn(IMAGE, cmd)
            self.assertIn('--network=none', cmd)
            self.assertIn('--pull=never', cmd)
            self.assertNotIn('--mount', cmd)
            self.assertNotIn('--gpus', cmd)

    def test_unlimited_timeout_default(self):
        a = launcher.parser().parse_args(self.argv)
        self.assertIsNone(a.timeout)
        self.assertEqual(a.memory, '36g')
        self.assertEqual(a.cpus, 8)

    def test_benchmark_metadata_is_inside_inputs_and_forwarded_as_container_path(self):
        inside = self.root / 'inputs' / 'meta.json'
        inside.write_text('{"provider": "unit-test"}')
        args = launcher.parser().parse_args(self.argv + ['--benchmark-metadata', str(inside)])
        launcher.validate(args)
        cmd = launcher.command(args, IMAGE, 'token')
        forwarded = cmd[cmd.index('--benchmark-metadata') + 1]
        self.assertEqual(forwarded, '/inputs/meta.json')
        self.assertNotIn(str(self.root), forwarded)

    def test_benchmark_metadata_outside_inputs_missing_or_not_a_file_is_rejected(self):
        outside = self.root / 'meta.json'
        outside.write_text('{}')
        (self.root / 'inputs' / 'nested').mkdir()
        for extra in (['--benchmark-metadata', str(outside)],
                      ['--benchmark-metadata', str(self.root / 'inputs' / 'absent.json')],
                      ['--benchmark-metadata', str(self.root / 'inputs' / 'nested')]):
            with self.subTest(extra=extra), patch.object(launcher, 'resolve_image') as image:
                self.assertEqual(launcher.main(self.argv + extra), 2)
                image.assert_not_called()

    def test_job_timeout_is_validated_and_forwarded(self):
        for value in ('0', '-1', 'nan', 'inf'):
            with self.subTest(value=value), patch.object(launcher, 'resolve_image') as image:
                self.assertEqual(launcher.main(self.argv + ['--job-timeout', value]), 2)
                image.assert_not_called()
        args = launcher.parser().parse_args(self.argv + ['--job-timeout', '900'])
        launcher.validate(args)
        cmd = launcher.command(args, IMAGE, 'token')
        self.assertEqual(cmd[cmd.index('--job-timeout') + 1], '900.0')

    def test_termination_json_written_on_timeout_with_container_and_progress(self):
        polls = [None, None, 137]
        process = Mock(poll=Mock(side_effect=lambda: polls.pop(0) if polls else 137),
                       wait=Mock(return_value=137))
        def popen(cmd):
            cidfile = Path(cmd[cmd.index('--cidfile') + 1])
            cidfile.write_text('c' * 64)
            (cidfile.parents[1] / 'progress.json').write_text(json.dumps({'schema': 1, 'jobs_completed': 2}))
            return process
        with patch.object(launcher, 'resolve_image', return_value=IMAGE), \
                patch.object(launcher, 'gpu_available'), patch.object(launcher.subprocess, 'Popen', side_effect=popen), \
                patch.object(launcher, 'owned_signal', return_value=True), \
                patch.object(launcher.time, 'monotonic', side_effect=[0, 5, 10]), patch.object(launcher.time, 'sleep'):
            self.assertEqual(launcher.main(self.argv + ['--timeout', '3']), 124)
        termination = json.loads((self.root / 'results' / 'termination.json').read_text())
        self.assertEqual(termination['schema'], 1)
        self.assertEqual(termination['reason'], 'timeout')
        self.assertIsNone(termination['signal'])
        self.assertEqual(termination['exit_code'], 124)
        self.assertEqual(termination['container_id'], 'c' * 64)
        self.assertEqual(termination['last_progress']['jobs_completed'], 2)
        self.assertGreater(termination['timestamp_unix'], 0)

    def test_no_termination_json_on_success(self):
        def popen(cmd):
            (self.root / 'results/run.json').write_text(json.dumps({'status': 'complete'}))
            return Mock(poll=Mock(return_value=0), wait=Mock(return_value=0))
        with patch.object(launcher, 'resolve_image', return_value=IMAGE), \
                patch.object(launcher, 'gpu_available'), patch.object(launcher.subprocess, 'Popen', side_effect=popen):
            self.assertEqual(launcher.main(self.argv), 0)
        self.assertFalse((self.root / 'results' / 'termination.json').exists())

    def test_termination_reason_signal_names_the_launcher_signal(self):
        handlers = {}
        def register(sig, fn):
            old = handlers.get(sig, signal.SIG_DFL)
            handlers[sig] = fn
            return old
        polls = [None, None, 143]
        process = Mock(poll=Mock(side_effect=lambda: polls.pop(0) if polls else 143),
                       wait=Mock(return_value=143))
        def popen(cmd):
            handlers[signal.SIGINT](signal.SIGINT, None)
            return process
        termination = {}
        with patch.object(launcher.signal, 'signal', side_effect=register), \
                patch.object(launcher.subprocess, 'Popen', side_effect=popen), \
                patch.object(launcher, 'owned_signal', return_value=True), \
                patch.object(launcher.time, 'monotonic', side_effect=[0, 1, 4]), patch.object(launcher.time, 'sleep'):
            self.assertEqual(launcher.run_worker(['docker', 'run'], self.root / 'owned.cid', None, 2, termination), 143)
        self.assertEqual(termination, {'reason': 'signal', 'signal': 'SIGINT'})


if __name__ == '__main__':
    unittest.main()

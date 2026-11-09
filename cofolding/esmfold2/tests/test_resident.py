"""CPU-only lifecycle unit tests. FakeEngine never loads ESMFold2 or a GPU."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resident.worker import run, Settings, Stop


class FakeEngine:
    loads = 0
    calls = []
    fail: int | None = None
    fatal = False
    stop: Stop | None = None

    def __init__(self, assets, checkpoint, settings):
        self.loaded = False

    def prepare(self, source, attempt, seed):
        if source.read_text() == 'bad':
            raise ValueError('bad input')
        return source

    def load(self):
        type(self).loads += 1
        self.loaded = True

    def snapshot(self):
        return {'model_id': 123, 'device': 'cpu-unit-test-only', 'parameter_data_ptr': 456}

    def predict(self, prepared, attempt, seed):
        type(self).calls.append(seed)
        (attempt / 'payload').write_text(str(seed))
        if seed == self.fail:
            raise RuntimeError('prediction failed')
        if self.stop:
            self.stop.requested = True

    def verify(self, prepared, attempt):
        assert (attempt / 'payload').is_file()

    def healthy(self):
        return not self.fatal


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        self.ckpt = self.root / 'checkpoint'
        self.ckpt.write_bytes(b'checkpoint')
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        self.output = self.root / 'output'
        FakeEngine.loads = 0
        FakeEngine.calls = []
        FakeEngine.fail = None
        FakeEngine.fatal = False
        FakeEngine.stop = None

    def go(self, seeds=(11, 12), resume=False, settings=None, stop=None):
        return run(self.inputs, self.output, self.assets, self.ckpt, list(seeds),
                   settings or Settings(), resume, FakeEngine, stop)

    def metadata(self):
        return json.loads((self.output / 'run.json').read_text())

    def test_one_load_seeds_verbatim_and_metadata(self):
        (self.inputs / 'good.yaml').write_text('good')
        self.assertEqual(self.go((12, 11, 12)), 0)
        self.assertEqual(FakeEngine.loads, 1)
        self.assertEqual(FakeEngine.calls, [12, 11, 12])
        m = self.metadata()
        self.assertEqual(m['requested_seeds'], [12, 11, 12])
        self.assertEqual(m['checkpoint_load_count'], 1)
        self.assertEqual(len(m['jobs']), 3)
        for job in m['jobs']:
            self.assertEqual(job['model_before'], job['model_after'])
            self.assertEqual(job['status'], 'completed')

    def test_bad_input_does_not_stop_good(self):
        (self.inputs / 'a.yaml').write_text('bad')
        (self.inputs / 'z.yaml').write_text('good')
        self.assertEqual(self.go(), 1)
        self.assertEqual(FakeEngine.calls, [11, 12])
        self.assertEqual(FakeEngine.loads, 1)
        self.assertEqual([j['status'] for j in self.metadata()['jobs']],
                         ['failed', 'failed', 'completed', 'completed'])

    def test_resume_verifies_and_does_not_load(self):
        (self.inputs / 'good.yaml').write_text('good')
        self.go()
        self.assertEqual(self.go(resume=True), 0)
        self.assertEqual(FakeEngine.loads, 1)
        self.assertEqual(self.metadata()['checkpoint_load_count'], 0)
        self.assertTrue(all(j['reused'] for j in self.metadata()['jobs']))

    def test_resume_changed_input_checkpoint_settings_and_corruption(self):
        src = self.inputs / 'good.yaml'
        src.write_text('good')
        self.go((11,))
        m = self.metadata()
        payload = self.output / m['jobs'][0]['output'] / 'payload'
        payload.write_text('corrupted')
        self.go((11,), True)
        self.assertEqual(FakeEngine.loads, 2)
        self.assertEqual(payload.read_text(), 'corrupted')  # old attempt retained
        src.write_text('changed')
        self.go((11,), True)
        self.ckpt.write_bytes(b'changed checkpoint')
        self.go((11,), True)
        self.go((11,), True, Settings(steps=201))
        self.assertEqual(FakeEngine.loads, 5)

    def test_partial_attempt_preserved_and_recoverable_prediction_continues(self):
        (self.inputs / 'good.yaml').write_text('good')
        FakeEngine.fail = 11
        self.assertEqual(self.go(), 1)
        old = self.output / self.metadata()['jobs'][0]['output']
        self.assertTrue((old / 'payload').exists())
        FakeEngine.fail = None
        self.assertEqual(self.go(resume=True), 0)
        self.assertEqual(FakeEngine.calls, [11, 12, 11])
        self.assertTrue(old.exists())

    def test_fatal_gpu_stops_without_reload(self):
        (self.inputs / 'good.yaml').write_text('good')
        FakeEngine.fail = 11
        FakeEngine.fatal = True
        self.assertEqual(self.go(), 1)
        self.assertEqual(FakeEngine.calls, [11])
        self.assertEqual(FakeEngine.loads, 1)
        self.assertEqual(self.metadata()['status'], 'fatal')
        self.assertEqual(self.metadata()['jobs'][1]['status'], 'pending')

    def test_stop_at_job_boundary(self):
        (self.inputs / 'good.yaml').write_text('good')
        stop = Stop()
        FakeEngine.stop = stop
        self.assertEqual(self.go(stop=stop), 143)
        self.assertEqual(FakeEngine.calls, [11])
        self.assertEqual(self.metadata()['status'], 'stopped')

    def test_asset_changes_invalidate_reuse(self):
        (self.inputs / 'good.yaml').write_text('good')
        mol = self.assets / 'mols' / 'ALA.pkl'
        mol.parent.mkdir()
        mol.write_bytes(b'asset-A')
        self.go((11,))
        mol.write_bytes(b'asset-B')
        self.go((11,), resume=True)
        self.assertEqual(FakeEngine.loads, 2)
        archive = self.assets / 'mols.tar'
        archive.write_bytes(b'package-A')
        self.go((11,), resume=True)
        archive.write_bytes(b'package-B')
        self.go((11,), resume=True)
        self.assertEqual(FakeEngine.loads, 4)

    def test_large_work_list_does_not_rewrite_full_metadata_per_job(self):
        import resident.worker as worker
        from unittest.mock import patch
        (self.inputs / 'good.yaml').write_text('good')
        actual = worker.atomic_json
        full_writes = []
        def observe(path, value):
            if Path(path).name == 'run.json':
                full_writes.append(len(value['jobs']))
            return actual(path, value)
        with patch.object(worker, 'atomic_json', side_effect=observe):
            self.assertEqual(self.go(range(50)), 0)
        self.assertEqual(full_writes, [50, 50])
        self.assertEqual(json.loads((self.output / 'progress.json').read_text())['jobs_completed'], 50)

    def test_launcher_runtime_directory_is_allowed_for_new_run(self):
        (self.inputs / 'good.yaml').write_text('good')
        (self.output / '.runtime' / 'tmp').mkdir(parents=True)
        self.assertEqual(self.go(), 0)
        self.assertEqual(FakeEngine.loads, 1)

    def test_model_storage_change_stops_without_reload(self):
        (self.inputs / 'good.yaml').write_text('good')
        class DriftingEngine(FakeEngine):
            drift = False
            def predict(self, prepared, attempt, seed):
                super().predict(prepared, attempt, seed)
                self.drift = True
            def snapshot(self):
                return {'model_id': 123, 'device': 'cpu-unit-test-only',
                        'parameter_data_ptr': 457 if self.drift else 456}
        self.assertEqual(run(self.inputs, self.output, self.assets, self.ckpt, [11, 12],
                             Settings(), engine_factory=DriftingEngine), 1)
        self.assertEqual(self.metadata()['status'], 'fatal')
        self.assertEqual(self.metadata()['jobs'][1]['status'], 'pending')

    def test_image_identity_change_does_not_reuse_prior_job(self):
        from unittest.mock import patch
        (self.inputs / 'good.yaml').write_text('good')
        with patch.dict('os.environ', {'COFOLD_IMAGE_ID': 'image-A'}):
            self.go((11,))
        with patch.dict('os.environ', {'COFOLD_IMAGE_ID': 'image-B'}):
            self.go((11,), resume=True)
        self.assertEqual(FakeEngine.loads, 2)
        self.assertFalse(self.metadata()['jobs'][0]['reused'])

    def test_existing_output_requires_resume(self):
        (self.inputs / 'good.yaml').write_text('good')
        self.go()
        with self.assertRaises(FileExistsError):
            self.go()

    def test_resume_seed_order_or_subset_does_not_invalidate_identity(self):
        (self.inputs / 'good.yaml').write_text('good')
        self.go()
        self.assertEqual(self.go((12, 11), resume=True), 0)
        self.assertEqual(self.go((12,), resume=True), 0)
        self.assertEqual(FakeEngine.loads, 1)

    def test_reference_bytes_invalidate_resume(self):
        try:
            import yaml
        except ImportError:
            self.skipTest('PyYAML absent in this CPU interpreter')
        msa = self.inputs / 'local.a3m'
        msa.write_text('first MSA')
        (self.inputs / 'good.yaml').write_text('sequences:\n- protein:\n    msa: local.a3m\n')
        self.go((11,))
        msa.write_text('changed MSA')
        self.go((11,), resume=True)
        self.assertEqual(FakeEngine.loads, 2)

    def test_cli_help_without_native_imports(self):
        import subprocess
        proc = subprocess.run([sys.executable, '-m', 'resident.worker', '--help'],
                              cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for flag in ('--seeds', '--samples', '--recycles', '--loops', '--steps', '--backend',
                     '--max-parallel-samples', '--no-kernels', '--output-format', '--resume'):
            self.assertIn(flag, proc.stdout)

    def test_checkpoint_load_failure_is_fatal_no_hidden_retry(self):
        (self.inputs / 'good.yaml').write_text('good')
        class FailingLoad(FakeEngine):
            def load(self):
                FakeEngine.loads += 1
                raise RuntimeError('load failed')
        rc = run(self.inputs, self.output, self.assets, self.ckpt, [11, 12],
                 Settings(), engine_factory=FailingLoad)
        self.assertEqual(rc, 1)
        self.assertEqual(FakeEngine.loads, 1)
        self.assertEqual(FakeEngine.calls, [])
        self.assertEqual(self.metadata()['status'], 'fatal')
        self.assertEqual(self.metadata()['jobs'][1]['status'], 'pending')

    def test_defaults_and_invalid_seed_or_settings_before_load(self):
        self.assertEqual(Settings(), Settings(1, 20, 200, 'fused', 1, 'mmcif'))
        (self.inputs / 'good.yaml').write_text('good')
        for seeds in ([], [-1], [4294967296]):
            with self.assertRaises(ValueError):
                self.go(seeds)
        with self.assertRaises(ValueError):
            self.go(settings=Settings(samples=0))
        self.assertEqual(FakeEngine.loads, 0)

    def test_extra_output_file_is_tolerated_and_counted(self):
        (self.inputs / 'good.yaml').write_text('good')
        self.go((11,))
        attempt = self.output / self.metadata()['jobs'][0]['output']
        (attempt / 'unrecorded').write_text('extra')
        self.assertEqual(self.go((11,), resume=True), 0)
        job = self.metadata()['jobs'][0]
        self.assertTrue(job['reused'])
        self.assertEqual(job['inventory_extras'], 1)
        self.assertEqual(FakeEngine.loads, 1)

    def test_missing_recorded_output_invalidates_receipt(self):
        (self.inputs / 'good.yaml').write_text('good')
        self.go((11,))
        attempt = self.output / self.metadata()['jobs'][0]['output']
        (attempt / 'payload').unlink()
        self.go((11,), resume=True)
        self.assertFalse(self.metadata()['jobs'][0]['reused'])
        self.assertEqual(FakeEngine.loads, 2)

    def test_changed_recorded_output_invalidates_receipt(self):
        (self.inputs / 'good.yaml').write_text('good')
        self.go((11,))
        attempt = self.output / self.metadata()['jobs'][0]['output']
        (attempt / 'payload').write_text('corrupted')
        self.go((11,), resume=True)
        self.assertFalse(self.metadata()['jobs'][0]['reused'])
        self.assertEqual(FakeEngine.loads, 2)

    def test_image_identity_prefers_label_then_env_then_reason(self):
        (self.inputs / 'good.yaml').write_text('good')
        from unittest.mock import patch
        from resident import worker
        label = self.root / 'image-label.json'
        with patch.object(worker, 'IMAGE_LABEL', label):
            label.write_text(json.dumps({'image_id': 'sha256:' + 'b' * 64}))
            with patch.dict(os.environ, {'COFOLD_IMAGE_ID': 'sha256:' + 'c' * 64}):
                self.assertEqual(worker.image_identity(),
                                 {'image_id': 'sha256:' + 'b' * 64, 'image_id_source': 'image_label'})
                self.go((11,))
                implementation = self.metadata()['implementation']
                self.assertEqual(implementation['image_id'], 'sha256:' + 'b' * 64)
                self.assertEqual(implementation['image_id_source'], 'image_label')
            label.unlink()
            with patch.dict(os.environ, {'COFOLD_IMAGE_ID': 'sha256:' + 'c' * 64}):
                self.assertEqual(worker.image_identity(),
                                 {'image_id': 'sha256:' + 'c' * 64, 'image_id_source': 'env'})
            with patch.dict(os.environ, {}, clear=True):
                result = worker.image_identity()
                self.assertIsNone(result['image_id'])
                self.assertIn('no image label', result['image_id_reason'])

    def test_output_lock_prevents_concurrent_workers(self):
        import fcntl
        (self.inputs / 'good.yaml').write_text('good')
        self.output.mkdir()
        with (self.output / '.worker.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.go(resume=True)
        self.assertEqual(FakeEngine.loads, 0)

    def test_failed_job_writes_redacted_failure_json_referenced_by_the_receipt(self):
        (self.inputs / 'good.yaml').write_text('good')

        class Leaky(FakeEngine):
            def predict(self, prepared, attempt, seed):
                if seed == 11:
                    raise RuntimeError('SMILES parse failed for /inputs/good.yaml: '
                                       'https://example.invalid/signed/x?sig=QUERYSECRET TOKEN=DO-NOT-LEAK MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTT')
                super().predict(prepared, attempt, seed)
        self.assertEqual(run(self.inputs, self.output, self.assets, self.ckpt, [11, 12],
                             Settings(), engine_factory=Leaky), 1)
        job = self.metadata()['jobs'][0]
        self.assertEqual(job['failure'], 'failure.json')
        attempt = self.output / job['output']
        failure = json.loads((attempt / 'failure.json').read_text())
        self.assertEqual(failure['schema'], 1)
        self.assertEqual(failure['error_kind'], 'native_error')
        self.assertEqual(failure['phase'], 'prediction')
        self.assertEqual(failure['exception_type'], 'RuntimeError')
        self.assertEqual(failure['attempt_path'], str(attempt))
        self.assertIn('SMILES parse failed for /inputs/good.yaml', failure['message'])
        self.assertIn('https://example.invalid/signed/x', failure['message'])
        self.assertNotIn('QUERYSECRET', failure['message'])
        self.assertNotIn('DO-NOT-LEAK', failure['message'])
        # A 40-residue sequence is a diagnosis, not a secret blob.
        self.assertIn('MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTT', failure['message'])
        self.assertIn('Traceback (most recent call last)', failure['traceback_tail'])
        self.assertNotIn('DO-NOT-LEAK', failure['traceback_tail'])
        receipt = json.loads((attempt / 'receipt.json').read_text())
        self.assertEqual(receipt['failure'], 'failure.json')
        self.assertEqual(receipt['schema'], 1)

    def test_progress_liveness_and_schema_stamps_every_json_artifact(self):
        (self.inputs / 'good.yaml').write_text('good')
        self.assertEqual(self.go(), 0)
        progress = json.loads((self.output / 'progress.json').read_text())
        self.assertEqual(progress['schema'], 1)
        self.assertIsInstance(progress['updated_at_unix'], float)
        self.assertEqual(progress['pid'], os.getpid())
        self.assertEqual(progress['jobs_completed'], 2)
        self.assertEqual(self.metadata()['schema'], 1)
        for job in self.metadata()['jobs']:
            receipt = json.loads((self.output / job['output'] / 'receipt.json').read_text())
            self.assertEqual(receipt['schema'], 1)

    def test_job_timeout_records_one_job_and_later_jobs_still_run(self):
        (self.inputs / 'a.yaml').write_text('good')
        (self.inputs / 'z.yaml').write_text('good')

        class Slow(FakeEngine):
            def predict(self, prepared, attempt, seed):
                type(self).calls.append(seed)
                if Path(prepared).name == 'a.yaml':
                    time.sleep(0.6)
                (attempt / 'payload').write_text(str(seed))
        self.assertEqual(run(self.inputs, self.output, self.assets, self.ckpt, [11], Settings(),
                             engine_factory=Slow, job_timeout=0.5), 1)
        jobs = self.metadata()['jobs']
        self.assertEqual([j['status'] for j in jobs], ['failed', 'completed'])
        self.assertEqual(jobs[0]['error_kind'], 'timeout')
        self.assertEqual(jobs[0]['failure'], 'failure.json')
        failure = json.loads((self.output / jobs[0]['output'] / 'failure.json').read_text())
        self.assertEqual(failure['error_kind'], 'timeout')
        self.assertIn('--job-timeout', failure['message'])
        self.assertEqual(self.metadata()['status'], 'failed')
        self.assertEqual(self.metadata()['checkpoint_load_count'], 1)
        self.assertTrue((self.output / jobs[1]['output'] / 'payload').is_file())

    def test_job_timeout_must_be_positive_and_finite(self):
        (self.inputs / 'good.yaml').write_text('good')
        for value in (0, -1.0, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run(self.inputs, self.output, self.assets, self.ckpt, [11], Settings(),
                    engine_factory=FakeEngine, job_timeout=value)
        self.assertEqual(FakeEngine.loads, 0)

    def test_real_sigterm_handler_writes_final_metadata(self):
        import subprocess
        (self.inputs / 'good.yaml').write_text('good')
        script = '''
import os, signal, sys, types
from test_resident import FakeEngine
from resident.worker import main
class SignalEngine(FakeEngine):
    def predict(self, prepared, attempt, seed):
        super().predict(prepared, attempt, seed)
        os.kill(os.getpid(), signal.SIGTERM)
module = types.ModuleType('resident.engine')
module.ESMFold2Engine = SignalEngine
sys.modules['resident.engine'] = module
sys.argv = ['cofold-run'] + sys.argv[1:]
raise SystemExit(main())
'''
        proc = subprocess.run([sys.executable, '-c', script,
                               '--input', str(self.inputs), '--output', str(self.output),
                               '--assets', str(self.assets), '--checkpoint', str(self.ckpt),
                               '--seeds', '11', '12'],
                              cwd=Path(__file__).resolve().parent, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 143, proc.stderr)
        self.assertEqual(self.metadata()['status'], 'stopped')
        self.assertEqual([j['status'] for j in self.metadata()['jobs']], ['completed', 'pending'])


if __name__ == '__main__':
    unittest.main()

"""CPU bookkeeping only: synthetic bytes and explicitly mocked model invocation.

Molecular inputs are copied from the shared Trp-cage/ethanol case. These tests
neither create predictions nor establish historical/GPU numerical equivalence.
"""
import contextlib
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import batch
import run_test as runner
from batch_native import runtime

IMAGE_ID = 'sha256:'+'a'*64
GPU = 'GPU-01234567-89ab-cdef-0123-456789abcdef'


class HistoricalCaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = self.root/'cases'
        self.directory = self.cases/'cyp'
        (self.directory/'boltz2').mkdir(parents=True)
        self.case = json.loads((ROOT.parent/'test-cases/trpcage-ethanol/input.json').read_text())
        self.case.update(seed=11, samples=5)
        self.settings = dict(record_id='cyp', recycling_steps=3, sampling_steps=200,
                             max_parallel_samples=5, no_kernels=False, threads=8, preprocessing_threads=4)
        self.checkpoint = self.root/'checkpoint'
        self.checkpoint.write_bytes(b'synthetic checkpoint identity, not weights')
        self.cache = self.root/'cache'; (self.cache/'mols').mkdir(parents=True)
        self.output = self.root/'output'
        self.payloads = {name: ('synthetic bookkeeping only: '+name).encode()
                         for name in runner.native_names('cyp', 5)}
        self.expected = dict(historical_run='synthetic-bookkeeping-only', image_id=IMAGE_ID,
                             checkpoint_sha256=batch.sha(self.checkpoint),
                             native_files={name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                                           for name, data in self.payloads.items()},
                             rank_to_original_sample=[2, 0, 4, 1, 3])
        self.save()
        self.argv = ['--case', 'cyp', '--output', str(self.output), '--cache', str(self.cache),
                     '--checkpoint', str(self.checkpoint), '--gpu', GPU]

    def save(self):
        for name, value in (('input.json', self.case), ('boltz2/settings.json', self.settings),
                            ('boltz2/expected.json', self.expected)):
            (self.directory/name).write_text(json.dumps(value))

    def synthetic_model_invocation(self, command, **kwargs):
        """Explicit mock: write meaningless bytes solely to exercise comparison."""
        if '--dry-run' in command:
            return subprocess.CompletedProcess(command, 0)
        stage = self.output/'prediction/seed-11'
        pred = stage/'native/boltz_results_inputs/predictions/cyp'
        pred.mkdir(parents=True, exist_ok=True)
        for name, payload in self.payloads.items():
            (pred/name).write_bytes(payload)
        (stage/'verified.json').write_text(json.dumps({'status': 'PASS', 'models': 5,
            'rank_to_original_sample': {'cyp': self.expected['rank_to_original_sample'] or list(range(5))}}))
        return subprocess.CompletedProcess(command, 0)

    def invoke(self, extra=(), effect=None):
        with patch.object(runner, 'CASES', self.cases), \
             patch.object(runner.subprocess, 'run', side_effect=effect or self.synthetic_model_invocation) as model, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = runner.main(self.argv+list(extra))
        return status, model

    def test_full_options_handoff_and_exact_pass(self):
        status, model = self.invoke()
        self.assertEqual(status, 0)
        command = model.call_args.args[0]
        for flag, value in (('--image', batch.IMAGE), ('--threads', '8'), ('--preprocessing-threads', '4'),
                            ('--seeds', '11'), ('--diffusion_samples', '5'), ('--recycling_steps', '3'),
                            ('--sampling_steps', '200'), ('--max_parallel_samples', '5')):
            self.assertEqual(command[command.index(flag)+1], value)
        self.assertNotIn('--no_kernels', command)
        self.assertNotIn(IMAGE_ID, command)  # Never silently substitute historical image.
        self.assertEqual(json.loads((self.output/'comparison.json').read_text())['status'], 'PASS')
        status, model = self.invoke(['--resume', '--image', 'registry.example/boltz:diagnosis'])
        self.assertEqual(status, 0)
        self.assertIn('--resume', model.call_args.args[0])
        self.assertIn('registry.example/boltz:diagnosis', model.call_args.args[0])

    def test_dry_run_and_kernels_are_forwarded_without_comparison_pass(self):
        self.settings['no_kernels'] = True; self.save()
        status, model = self.invoke(['--dry-run'])
        self.assertEqual(status, 0)
        self.assertIn('--dry-run', model.call_args.args[0]); self.assertIn('--no_kernels', model.call_args.args[0])
        self.assertFalse((self.output/'comparison.json').exists())

    def test_missing_extra_different_and_wrong_ranks_fail_exactly(self):
        self.assertEqual(self.invoke()[0], 0)
        stage = self.output/'prediction/seed-11'
        pred = stage/'native/boltz_results_inputs/predictions/cyp'
        missing, differing = sorted(self.payloads)[:2]
        (pred/missing).unlink(); (pred/differing).write_bytes(b'changed')
        (pred/'unexpected').write_bytes(b'extra')
        (stage/'verified.json').write_text(json.dumps({'status': 'PASS', 'models': 5,
            'rank_to_original_sample': {'cyp': list(range(5))}}))
        status, _ = self.invoke(['--resume'], effect=lambda *a, **k: None)
        self.assertEqual(status, 1)
        result = json.loads((self.output/'comparison.json').read_text())
        self.assertEqual(result['missing'], [missing]); self.assertEqual(result['extra'], ['unexpected'])
        self.assertEqual(result['differing'], [differing]); self.assertFalse(result['ranks_match'])
        self.assertEqual(result['status'], 'FAIL')

    def test_bad_checkpoint_or_model_failure_invalidates_old_pass(self):
        self.assertEqual(self.invoke()[0], 0)
        status, _ = self.invoke(effect=subprocess.CalledProcessError(1, ['explicit mocked model']))
        self.assertEqual(status, 1)
        self.assertEqual(json.loads((self.output/'comparison.json').read_text())['status'], 'FAIL')
        self.checkpoint.write_bytes(b'wrong checkpoint')
        status, model = self.invoke()
        self.assertEqual(status, 1); model.assert_not_called()

    def test_msa_hash_containment_and_native_entity_order(self):
        data = ('>query\n'+self.case['protein']['sequence']+'\n').encode()
        (self.directory/'msa.a3m.gz').write_bytes(gzip.compress(data))
        self.case['msa'] = {'file': 'msa.a3m.gz', 'sha256': hashlib.sha256(data).hexdigest()}
        self.case['cofactors'] = [{'id': 'H', 'smiles': 'C'}]; self.save()
        self.assertEqual(self.invoke(['--dry-run'])[0], 0)
        native = json.loads((self.output/'inputs/cyp.yaml').read_text())
        self.assertEqual([next(iter(e.values()))['id'] for e in native['sequences']],
                         [self.case['protein']['id'], 'H', self.case['ligand']['id']])
        self.assertEqual((self.output/'msa/msa.a3m').read_bytes(), data)
        self.case['msa']['sha256'] = '0'*64; self.save()
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): runner.load_case(self.directory)
        self.case['msa']['file'] = '../outside.a3m.gz'; self.save()
        with self.assertRaises(ValueError): runner.load_case(self.directory)
        self.case['msa']['file'] = 'link.a3m.gz'; self.save()
        (self.directory/'link.a3m.gz').symlink_to(self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'escapes'): runner.load_case(self.directory)

    def test_existing_incompatible_inputs_never_overwritten(self):
        self.assertEqual(self.invoke(['--dry-run'])[0], 0)
        target = self.output/'inputs/cyp.yaml'; target.write_bytes(b'preserve me')
        status, model = self.invoke()
        self.assertEqual(status, 1); model.assert_not_called()
        self.assertEqual(target.read_bytes(), b'preserve me')

    def test_strict_types_keys_and_expected_inventory(self):
        original = copy.deepcopy(self.case)
        for key, value in (('seed', True), ('samples', 1.0), ('msa', False), ('cofactors', {}), ('extra', 1)):
            self.case = dict(original, **{key: value}); self.save()
            with self.subTest(key=key), self.assertRaises(ValueError): runner.load_case(self.directory)
        self.case = original; self.settings['no_kernels'] = 0; self.save()
        with self.assertRaises(ValueError): runner.load_case(self.directory)
        self.settings['no_kernels'] = False
        self.expected['rank_to_original_sample'][0] = True; self.save()
        with self.assertRaises(ValueError): runner.load_case(self.directory)
        self.expected['rank_to_original_sample'] = list(range(5))
        self.expected['native_files'].pop(next(iter(self.payloads))); self.save()
        with self.assertRaises(ValueError): runner.load_case(self.directory)

    def test_duplicate_and_nonfinite_json_rejected(self):
        for content in ('{"id":1,"id":2}', '{"id":NaN}'):
            (self.directory/'input.json').write_text(content)
            with self.assertRaises(ValueError): runner.load_case(self.directory)

    def test_pxr_selection_uses_same_batch_route(self):
        import shutil
        shutil.copytree(self.directory, self.cases/'pxr')
        argv = list(self.argv); argv[argv.index('--case')+1] = 'pxr'
        with patch.object(runner, 'CASES', self.cases), \
             patch.object(runner.subprocess, 'run', side_effect=self.synthetic_model_invocation) as model:
            self.assertEqual(runner.main(argv+['--dry-run']), 0)
        self.assertIn('--dry-run', model.call_args.args[0])

    def test_registry_image_local_resolution_no_pull(self):
        reference = 'ghcr.io/example/boltz2@sha256:'+'b'*64
        with patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, IMAGE_ID+'\n')) as call:
            self.assertEqual(batch.resolve_image(reference), IMAGE_ID)
            self.assertEqual(call.call_args.args[0], ['docker', 'image', 'inspect', '--format', '{{.Id}}', reference])

    def test_preprocessing_threads_separate_and_compatibility_default(self):
        for preprocessing, expected in ((None, '8'), (4, '4')):
            settings = dict(self.settings, diffusion_samples=5)
            if preprocessing is None: settings.pop('preprocessing_threads')
            argv = runtime.native_args(Path('/run/seed-11'), 11, settings)
            self.assertEqual(argv[argv.index('--preprocessing-threads')+1], expected)
        # Exercise real batch dry-run validation/command path, no container/model.
        self.invoke(['--dry-run'])
        command = [str(ROOT/'batch.py'), '--input', str(self.output/'inputs'), '--output', str(self.root/'native'),
                   '--cache', str(self.cache), '--checkpoint', str(self.checkpoint), '--gpu', GPU, '--dry-run', '--threads', '8']
        for flags, expected in (([], 8), (['--preprocessing-threads', '4'], 4)):
            with patch.object(sys, 'argv', command+flags), patch.object(batch, 'resolve_image', return_value=IMAGE_ID), \
                 contextlib.redirect_stdout(io.StringIO()) as stdout:
                batch.main()
            report = json.loads(stdout.getvalue())
            self.assertEqual(report['request']['settings']['preprocessing_threads'], expected)
            self.assertIn('OMP_NUM_THREADS=8', report['commands'][0])
        with patch.object(sys, 'argv', command+['--preprocessing-threads', '0']), self.assertRaises(ValueError):
            batch.main()
        zero_threads = command[:command.index('--threads')]+['--threads', '0', '--num-workers', '2']
        with patch.object(sys, 'argv', zero_threads), patch.object(batch, 'resolve_image', return_value=IMAGE_ID), \
             patch.object(batch.os, 'cpu_count', return_value=24), contextlib.redirect_stdout(io.StringIO()) as stdout:
            batch.main()
        report = json.loads(stdout.getvalue())
        self.assertEqual(report['request']['settings']['preprocessing_threads'], 24)
        self.assertEqual(report['request']['settings']['num_workers'], 2)
        self.assertFalse(any('NUM_THREADS=' in arg for arg in report['commands'][0]))
        with patch.object(sys, 'argv', command+['--num-workers', '-1']), self.assertRaises(ValueError):
            batch.main()

    def test_pxr_pdb_potentials_no_embeddings_and_nullable_historical_ranks(self):
        self.settings.update(output_format='pdb', use_potentials=True, write_embeddings=False,
                             num_workers=2, threads=0, preprocessing_threads=24)
        self.payloads = {name: ('synthetic bookkeeping only: '+name).encode()
                         for name in runner.native_names('cyp', 5, 'pdb', False)}
        self.expected['native_files'] = {
            name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
            for name, data in self.payloads.items()}
        self.expected['rank_to_original_sample'] = None
        self.save()
        self.assertEqual(len(self.payloads), 25)
        self.assertTrue(all(not name.endswith('.cif') and not name.startswith('embeddings_')
                            for name in self.payloads))
        status, model = self.invoke()
        self.assertEqual(status, 0)
        command = model.call_args.args[0]
        self.assertIn('--use-potentials', command)
        self.assertIn('--no-write-embeddings', command)
        self.assertEqual(command[command.index('--output-format')+1], 'pdb')
        parsed = batch.parser().parse_args(command[2:])
        settings = vars(parsed)
        argv = runtime.native_args(Path('/run/seed-11'), 11, settings)
        self.assertEqual(argv[argv.index('--model')+1], 'boltz2')
        self.assertEqual(argv[argv.index('--output_format')+1], 'pdb')
        self.assertEqual(argv[argv.index('--num_workers')+1], '2')
        self.assertEqual(argv[argv.index('--preprocessing-threads')+1], '24')
        self.assertIn('--use_potentials', argv)
        self.assertIn('--write_full_pae', argv); self.assertIn('--write_full_pde', argv)
        self.assertNotIn('--write_embeddings', argv); self.assertNotIn('--no_kernels', argv)
        container = batch.command(parsed, self.output, 'observe.py', 11, True)
        self.assertFalse(any(name in value for value in container
                             for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')))
        result = json.loads((self.output/'comparison.json').read_text())
        self.assertFalse(result['historical_ranks_checked'])
        self.assertTrue(result['ranks_match'])
        # Unknown original indices must never weaken ranked-filename byte checks.
        pred = self.output/'prediction/seed-11/native/boltz_results_inputs/predictions/cyp'
        name = 'cyp_model_0.pdb'; (pred/name).write_bytes(b'changed')
        status, _ = self.invoke(['--resume'], effect=lambda *a, **k: None)
        self.assertEqual(status, 1)
        self.assertEqual(json.loads((self.output/'comparison.json').read_text())['differing'], [name])
        self.synthetic_model_invocation([])
        receipt = self.output/'prediction/seed-11/verified.json'
        receipt.write_text(json.dumps({'status': 'PASS', 'models': 5,
                                      'rank_to_original_sample': {'cyp': None}}))
        self.assertEqual(self.invoke(['--resume'], effect=lambda *a, **k: None)[0], 1)

    def test_output_setting_defaults_and_invalid_types(self):
        _, settings, _, _ = runner.load_case(self.directory)
        self.assertEqual([settings[k] for k in ('output_format', 'use_potentials', 'write_embeddings')],
                         ['mmcif', False, True])
        argv = runtime.native_args(Path('/run/seed-11'), 11, dict(self.settings, diffusion_samples=5))
        self.assertEqual(argv[argv.index('--output_format')+1], 'mmcif')
        self.assertIn('--write_embeddings', argv); self.assertNotIn('--use_potentials', argv)
        original = dict(self.settings)
        for key, value in (('output_format', 'cif'), ('output_format', []),
                           ('use_potentials', 1), ('write_embeddings', 'false'),
                           ('num_workers', -1), ('num_workers', True), ('threads', -1)):
            self.settings = dict(original, **{key: value}); self.save()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                runner.load_case(self.directory)


if __name__ == '__main__':
    unittest.main()

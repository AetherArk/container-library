"""CPU-only checks for harness bookkeeping; no model/GPU evidence."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'scripts/test_local.py'
spec = importlib.util.spec_from_file_location('local_test_harness', path)
assert spec is not None and spec.loader is not None
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

class CoverageTests(unittest.TestCase):
    def test_image_checks_use_writable_caches_for_nondefault_uid(self):
        with patch.object(harness.os, 'getuid', return_value=1001), patch.object(harness.os, 'getgid', return_value=1001), patch.object(harness.launcher, 'resolve_image', return_value='sha256:' + 'a' * 64), patch.object(harness, 'run') as run:
            self.assertEqual(harness.main(['--suite', 'image', '--image', 'unit-only', '--cpus', '2']), 0)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index('--user') + 1], '1001:1001')
            self.assertEqual(command[command.index('--workdir') + 1], '/tmp')
            self.assertIn('/tmp:rw,nosuid,nodev,mode=1777,size=1g', command)
            for value in ('HOME=/tmp', 'NUMBA_CACHE_DIR=/tmp/numba', 'XDG_CACHE_HOME=/tmp/cache', 'TRITON_CACHE_DIR=/tmp/triton', 'HF_HOME=/tmp/hf'):
                self.assertIn(value, command)
            self.assertNotIn('--gpus', command)

    def test_duplicate_seed_occurrences_are_preserved(self):
        jobs = [{'input': '/inputs/a.yaml', 'seed': seed} for seed in [12, 11, 12]]
        self.assertTrue(harness.coverage_matches(jobs, ['a.yaml'], [12, 11, 12]))
        self.assertFalse(harness.coverage_matches(jobs[:-1], ['a.yaml'], [12, 11, 12]))
        self.assertFalse(harness.coverage_matches(jobs + [jobs[0]], ['a.yaml'], [12, 11, 12]))

    def test_every_input_gets_the_requested_seed_list(self):
        jobs = [{'input': name, 'seed': seed} for name in ['a.yaml', 'b.yaml'] for seed in [11, 12]]
        self.assertTrue(harness.coverage_matches(jobs, ['a.yaml', 'b.yaml'], [11, 12]))
        jobs[-1]['seed'] = 13
        self.assertFalse(harness.coverage_matches(jobs, ['a.yaml', 'b.yaml'], [11, 12]))

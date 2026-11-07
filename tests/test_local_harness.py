"""CPU-only checks for harness bookkeeping; no model/GPU evidence."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/test_local.py'
spec = importlib.util.spec_from_file_location('local_test_harness', path)
assert spec is not None and spec.loader is not None
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

class CoverageTests(unittest.TestCase):
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

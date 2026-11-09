"""Worker command-line contract: shared flags are read from cofolding/cli-contract.json."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

import resident.worker as worker  # noqa: E402

CONTRACT = json.loads((REPO / 'cofolding/cli-contract.json').read_text())
CONTAINER = 'esmfold2'

_launcher_spec = importlib.util.spec_from_file_location('cli_contract_launcher', REPO / 'run_cofold.py')
assert _launcher_spec is not None and _launcher_spec.loader is not None
launcher = importlib.util.module_from_spec(_launcher_spec)
_launcher_spec.loader.exec_module(launcher)


def parse(argv):
    """Run the real parser and settings construction without starting inference."""
    captured = {}

    def stop_before_run(*args, **kwargs):
        captured['args'] = args
        return 0

    original = worker.run
    worker.run = stop_before_run
    try:
        original_argv = sys.argv
        sys.argv = ['cofold-run'] + argv
        try:
            code = worker.main()
        finally:
            sys.argv = original_argv
    finally:
        worker.run = original
    args = captured['args']
    # run(inputs, output, assets, checkpoint, seeds, settings, resume, ...)
    return code, args[4], args[5]


BASE = ['--input', 'in', '--output', 'out', '--assets', 'assets', '--checkpoint', 'ckpt',
        '--seeds', '11']


class ContractTests(unittest.TestCase):
    """The one machine-readable table, checked against the installed parser."""

    def action(self, flag):
        return worker.parser()._option_string_actions.get(flag)

    def test_every_table_flag_is_implemented_with_its_declared_domain(self):
        for flag, spec in CONTRACT['flags'].items():
            status = spec['containers'][CONTAINER]
            if status == 'unknown':
                continue
            with self.subTest(flag=flag, status=status):
                action = self.action(flag)
                if status == 'rejected' and action is None:
                    self.assertEqual(spec.get('container_domain', {}).get(CONTAINER, []), [])
                    self.assertTrue(spec['reason'][CONTAINER])
                    continue
                self.assertIsNotNone(action, flag + ' is ' + status + ' in cli-contract.json '
                                     'but missing from the installed parser')
                self.assertEqual(action.required, spec['required'])
                if spec['type'] == 'enum':
                    domain = spec.get('container_domain', {}).get(CONTAINER, spec['domain'])
                    self.assertEqual(list(action.choices), list(domain),
                                     flag + ' advertises a different domain than cli-contract.json')

    def test_every_accepted_worker_value_is_reachable_through_the_launcher(self):
        launched = launcher.parser()
        for flag, spec in CONTRACT['flags'].items():
            worker_action = self.action(flag)
            launcher_action = launched._option_string_actions.get(flag)
            if worker_action is None or launcher_action is None or spec['type'] != 'enum':
                continue
            with self.subTest(flag=flag):
                self.assertEqual(list(launcher_action.choices), list(spec['domain']))
                self.assertLessEqual(set(worker_action.choices), set(launcher_action.choices))

    def test_output_format_cif_is_not_a_shared_value(self):
        decision = CONTRACT['output_format_decision']
        self.assertNotIn('cif', decision['containers'][CONTAINER])
        self.assertEqual(list(self.action('--output-format').choices), ['mmcif'])


class CliTests(unittest.TestCase):
    def test_defaults_match_the_engine(self):
        code, seeds, settings = parse(BASE)
        self.assertEqual(code, 0)
        self.assertEqual(seeds, [11])
        self.assertEqual((settings.samples, settings.loops, settings.steps, settings.backend,
                          settings.max_parallel_samples, settings.output_format),
                         (1, 20, 200, 'fused', 1, 'mmcif'))

    def test_recycles_means_num_loops(self):
        _, _, settings = parse(BASE + ['--recycles', '5'])
        self.assertEqual(settings.loops, 5)

    def test_loops_alias(self):
        _, _, settings = parse(BASE + ['--loops', '7'])
        self.assertEqual(settings.loops, 7)

    def test_no_kernels_selects_reference_backend(self):
        _, _, settings = parse(BASE + ['--no-kernels'])
        self.assertEqual(settings.backend, 'reference')

    def test_explicit_backend_is_preserved(self):
        _, _, settings = parse(BASE + ['--backend', 'reference'])
        self.assertEqual(settings.backend, 'reference')

    def test_pdb_output_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse(BASE + ['--output-format', 'pdb'])

    def test_cif_output_is_rejected(self):
        # cif used to be accepted here while the launcher could never send it.
        with self.assertRaises(SystemExit):
            parse(BASE + ['--output-format', 'cif'])

    def test_parallel_samples_above_one_is_rejected_before_inference(self):
        with self.assertRaises(SystemExit):
            parse(BASE + ['--max-parallel-samples', '4'])

    def test_settings_reject_parallel_samples_and_seed_overflow(self):
        with self.assertRaisesRegex(ValueError, 'max-parallel-samples'):
            worker.run(Path('in'), Path('out'), None, Path('ckpt'), [11],
                       worker.Settings(max_parallel_samples=2), engine_factory=object)
        with self.assertRaisesRegex(ValueError, 'signed 32-bit'):
            worker.run(Path('in'), Path('out'), None, Path('ckpt'), [2147484],
                       worker.Settings(samples=5), engine_factory=object)
        with self.assertRaisesRegex(ValueError, 'invalid loop/sampling'):
            worker.run(Path('in'), Path('out'), None, Path('ckpt'), [11],
                       worker.Settings(steps=0), engine_factory=object)

    def test_help_lists_the_compatibility_flags(self):
        proc = subprocess.run([sys.executable, '-m', 'resident.worker', '--help'],
                              cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for flag in ('--recycles', '--steps', '--samples', '--seeds', '--backend', '--no-kernels',
                     '--max-parallel-samples', '--output-format', '--resume'):
            self.assertIn(flag, proc.stdout)

    def test_throughput_cli_reads_the_packaged_examples(self):
        import throughput_benchmark as tb
        items = tb.panel(ROOT / 'examples')
        self.assertEqual([item['system'] for item in items],
                         ['cyp.yaml', 'trpcage-ethanol.yaml', 'two-chain.yaml'])
        self.assertTrue(all(item['sha256'] for item in items))


if __name__ == '__main__':
    unittest.main()

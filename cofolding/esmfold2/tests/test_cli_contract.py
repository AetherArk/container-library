"""Worker command-line contract: Boltz-compatible flags mapped onto ESMFold2 settings."""
import argparse
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import resident.worker as worker  # noqa: E402


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

"""Sweep-side contract for a directory checkpoint and Chai-1 sampling limits."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import throughput_benchmark as tb  # noqa: E402
from resident.identity import directory_identity  # noqa: E402
from resident.worker import Settings  # noqa: E402


def parse(argv):
    """Parse sweep arguments without executing a worker."""
    captured = {}

    def stop(args):
        captured['args'] = args
        return {'status': 'completed'}

    original = tb.sweep
    tb.sweep = stop
    try:
        tb.main(['sweep'] + argv)
    finally:
        tb.sweep = original
    return captured['args']


class SweepContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.model = root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{"model": "fixture"}')
        (self.model / 'weights.bin').write_bytes(b'not a real checkpoint')
        self.assets = root / 'assets'
        self.assets.mkdir()
        self.base = ['--input', str(ROOT / 'examples'), '--output', str(root / 'out'),
                     '--assets', str(self.assets), '--checkpoint', str(self.model),
                     '--seeds', '11']

    def test_directory_checkpoint_uses_the_worker_scheme(self):
        self.assertEqual(tb.checkpoint_digest(self.model), directory_identity(self.model)['sha256'])
        before = tb.checkpoint_digest(self.model)
        (self.model / 'weights.bin').write_bytes(b'changed bytes')
        self.assertNotEqual(before, tb.checkpoint_digest(self.model))

    def test_file_checkpoint_still_hashes(self):
        single = Path(self.tmp.name) / 'single.ckpt'
        single.write_bytes(b'bytes')
        self.assertEqual(tb.checkpoint_digest(single), tb.sha(single))

    def test_missing_checkpoint_is_named(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint missing'):
            tb.checkpoint_digest(Path(self.tmp.name) / 'absent')

    def test_defaults_follow_the_engine(self):
        args = parse(self.base)
        self.assertEqual(args.levels, [1])
        self.assertEqual((args.steps, args.recycles), (200, 3))

    def test_parallel_level_above_one_is_refused(self):
        with self.assertRaises(SystemExit):
            parse(self.base + ['--levels', '1', '2'])

    def test_use_potentials_is_refused(self):
        with self.assertRaises(SystemExit):
            parse(self.base + ['--use-potentials'])

    def test_no_kernels_is_refused(self):
        with self.assertRaises(SystemExit):
            parse(self.base + ['--no-kernels'])

    def test_worker_command_carries_the_mapped_flags(self):
        args = parse(self.base + ['--steps', '64', '--recycles', '12', '--samples', '3',
                                  '--worker-executable', '/bin/echo'])
        command = tb.worker_command(args, Path('/staged'), Path('/config'), 1, 3, self.model)
        self.assertEqual(command[0], '/bin/echo')
        for flag, value in (('--steps', '64'), ('--recycles', '12'), ('--samples', '3'),
                            ('--max-parallel-samples', '1'), ('--output-format', 'mmcif'),
                            ('--derived', 'tensors')):
            index = command.index(flag)
            self.assertEqual(command[index + 1], value, command)
        self.assertNotIn('--no-kernels', command)
        self.assertNotIn('--backend', command)
        self.assertEqual(command[command.index('--checkpoint') + 1], str(self.model))

    def test_sweep_settings_match_the_worker_settings(self):
        args = parse(self.base + ['--samples', '2'])
        settings = {**Settings().__dict__, 'samples': 2, 'max_parallel_samples': 1}
        self.assertEqual(set(settings), set(Settings().__dataclass_fields__))


if __name__ == '__main__':
    unittest.main()

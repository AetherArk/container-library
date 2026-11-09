"""Boltz2 worker command line checked against the one shared cofolding/cli-contract.json."""
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
CONTAINER = 'boltz2'

_launcher_spec = importlib.util.spec_from_file_location('cli_contract_launcher', REPO / 'run_cofold.py')
assert _launcher_spec is not None and _launcher_spec.loader is not None
launcher = importlib.util.module_from_spec(_launcher_spec)
_launcher_spec.loader.exec_module(launcher)


class ContractTests(unittest.TestCase):
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

    def test_help_lists_the_shared_flags(self):
        proc = subprocess.run([sys.executable, '-m', 'resident.worker', '--help'],
                              cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for flag in ('--recycles', '--steps', '--samples', '--seeds', '--no-kernels',
                     '--use-potentials', '--max-parallel-samples', '--output-format',
                     '--capture', '--resume'):
            self.assertIn(flag, proc.stdout)


if __name__ == '__main__':
    unittest.main()

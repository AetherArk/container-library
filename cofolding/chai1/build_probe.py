"""Build-time probe: every chai_lab submodule must import.

chai_lab is installed with --no-deps because its metadata pins torch<2.7, which
cannot drive sm_120. An incomplete requirements.lock would otherwise surface as a
ModuleNotFoundError in the middle of a fold, on a rented GPU, minutes in.
"""
import importlib
import pkgutil
import sys

import chai_lab

failures = []
for module in pkgutil.walk_packages(chai_lab.__path__, 'chai_lab.'):
    try:
        importlib.import_module(module.name)
    except Exception as exc:  # noqa: BLE001 - the point is to report every one
        failures.append('{}: {}: {}'.format(module.name, type(exc).__name__, exc))
if failures:
    print('\n'.join(failures), file=sys.stderr)
    raise SystemExit('chai_lab submodule imports failed')
print('all chai_lab submodules import')

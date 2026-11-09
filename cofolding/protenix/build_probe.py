"""Build-time import probe for the Protenix container.

`protenix==2.0.0` is installed with `--no-deps` on top of a pinned `pip freeze`, so an
incomplete lock would only surface as a runtime failure in the middle of a fold.  Import every
module this container reaches, compile the bundled fast-layer-norm CUDA extension once (the
model imports it), and assert the resolved versions.  No GPU, no weights, no network.
"""
import importlib
import subprocess
import sys
from pathlib import Path

import torch

EXPECTED = {'protenix': '2.0.0', 'torch': '2.7.1', 'biotite': '1.4.0', 'rdkit': '2025.9.3'}


def normalised(version):
    parts = []
    for piece in version.split('.'):
        parts.append(str(int(piece)) if piece.isdigit() else piece)
    return '.'.join(parts)

for module in ('configs.configs_base', 'configs.configs_data', 'configs.configs_inference',
               'configs.configs_model_type', 'protenix.config.config',
               'protenix.data.inference.infer_dataloader', 'protenix.data.constants',
               'protenix.model.protenix', 'protenix.model.layer_norm',
               'protenix.utils.seed', 'protenix.utils.torch_utils',
               'runner.inference', 'runner.dumper', 'protenix.data.utils',
               'cofolding.common.derived', 'resident.worker', 'resident.engine',
               'resident.verify'):
    importlib.import_module(module)

import protenix
import biotite
import rdkit
import importlib.metadata
import sysconfig
from protenix.model.layer_norm.layer_norm import fast_layer_norm_cuda_v2
from protenix.model.layer_norm import layer_norm as layer_norm_module

resolved = {'protenix': protenix.__version__,
            'torch': torch.__version__.split('+')[0],
            'biotite': biotite.__version__,
            'rdkit': rdkit.__version__}
assert {k: normalised(v) for k, v in resolved.items()} == EXPECTED, resolved
assert importlib.metadata.version('protenix') == '2.0.0'
assert fast_layer_norm_cuda_v2 is not None
# The kernel must already be compiled next to the module, so a run never needs nvcc.
# torch.utils.cpp_extension names the artifact `fast_layer_norm_cuda_v2.so` beside the
# module (no ABI suffix); accept the ABI-suffixed name too, in case a future build differs.
base = Path(layer_norm_module.__file__).parent
candidates = [base / 'fast_layer_norm_cuda_v2.so',
              base / ('fast_layer_norm_cuda_v2' + sysconfig.get_config_var('EXT_SUFFIX'))]
compiled = next((c for c in candidates if c.is_file()), None)
assert compiled is not None, candidates
# The build step pins sm_120 explicitly; assert the cubin is really there, because a
# kernel built for older architectures only fails on the GPU with "no kernel image".
architectures = sorted({part.split('.')[-2] for part in subprocess.run(
    ['cuobjdump', '-lelf', str(compiled)], capture_output=True, text=True).stdout.split()
    if '.sm_' in part})
assert 'sm_120' in architectures, architectures
print('kernel cubins:', architectures)
print('imports ok:', resolved, 'cuda build:', torch.version.cuda, 'layernorm:', compiled.name)

"""Build Protenix's fast-layer-norm CUDA kernel for sm_120 at image build time.

Protenix compiles this kernel on first import through torch.utils.cpp_extension and
falls back to its own architecture list when no GPU is visible; on a build host that
list stops at compute_100, so the resulting binary has no sm_120 cubin and the first
forward pass on an RTX 50-series card fails with "no kernel image is available for
execution on the device". Compiling here with an explicit gencode list removes that
failure mode and removes a one-to-two minute compile from every container run.

The commands mirror what torch's cpp_extension emitted for this package; the only
difference is the architecture list.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME

import protenix.model.layer_norm as layer_norm_package

package_file = layer_norm_package.__file__
if package_file is None:
    sys.exit('protenix.model.layer_norm has no __file__')
module_dir = Path(package_file).parent
kernel_dir = module_dir / 'kernel'
build_dir = Path('/tmp/fast_layer_norm_build')
build_dir.mkdir(parents=True, exist_ok=True)

cuda_home = Path(CUDA_HOME) if CUDA_HOME else Path('/usr/local/cuda')
nvcc = shutil.which('nvcc') or str(cuda_home / 'bin' / 'nvcc')
python_include = subprocess.run(
    ['python', '-c', 'import sysconfig; print(sysconfig.get_paths()["include"])'],
    capture_output=True, text=True).stdout.strip()

torch_include = Path(torch.__file__).parent / 'include'
torch_lib = Path(torch.__file__).parent / 'lib'

ARCHITECTURES = [70, 80, 86, 89, 90, 100, 120]

cpp_obj = build_dir / 'layer_norm_cuda.o'
cuda_obj = build_dir / 'layer_norm_cuda_kernel.cuda.o'
artifact = module_dir / 'fast_layer_norm_cuda_v2.so'

common_defines = ['-DVERSION_GE_1_1', '-DVERSION_GE_1_3', '-DVERSION_GE_1_5']
includes = ['-isystem', str(torch_include),
            '-isystem', str(torch_include / 'torch' / 'csrc' / 'api' / 'include'),
            '-isystem', str(cuda_home / 'include'), '-isystem', python_include]
abi = ['-DPYBIND11_COMPILER_TYPE="_gcc"', '-DPYBIND11_STDLIB="_libstdcpp"',
       '-DPYBIND11_BUILD_ABI="_cxxabi1016"', '-DTORCH_API_INCLUDE_EXTENSION_H']

cpp_cmd = ['c++', '-MMD', '-MF', str(cpp_obj) + '.d',
           '-DTORCH_EXTENSION_NAME=fast_layer_norm_cuda_v2', *abi,
           '-I' + str(kernel_dir), *includes,
           '-D_GLIBCXX_USE_CXX11_ABI=1', '-fPIC', '-std=c++17', '-O3', *common_defines,
           '-c', str(kernel_dir / 'layer_norm_cuda.cpp'), '-o', str(cpp_obj)]

cuda_cmd = [nvcc, '--generate-dependencies-with-compile',
            '--dependency-output', str(cuda_obj) + '.d',
            '-DTORCH_EXTENSION_NAME=fast_layer_norm_cuda_v2', *abi,
            '-I' + str(kernel_dir), *includes,
            '-D_GLIBCXX_USE_CXX11_ABI=1',
            '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_BFLOAT16_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__',
            '--compiler-options', '-fPIC', '-O3', '--use_fast_math', *common_defines,
            '-std=c++17', '-maxrregcount=32',
            '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
            '--expt-relaxed-constexpr', '--expt-extended-lambda']
for arch in ARCHITECTURES:
    cuda_cmd += ['-gencode', 'arch=compute_{0},code=sm_{0}'.format(arch)]
cuda_cmd += ['-c', str(kernel_dir / 'layer_norm_cuda_kernel.cu'), '-o', str(cuda_obj)]

link_cmd = ['c++', str(cpp_obj), str(cuda_obj), '-shared',
            '-L' + str(torch_lib), '-lc10', '-lc10_cuda',
            '-ltorch_cpu', '-ltorch_cuda', '-ltorch', '-ltorch_python',
            '-L' + str(cuda_home / 'lib64'), '-lcudart', '-o', str(artifact)]

for command in (cpp_cmd, cuda_cmd, link_cmd):
    print('+', command[0], '...', flush=True)
    if subprocess.run(command).returncode != 0:
        sys.exit('build step failed: ' + ' '.join(command[:3]))

listing = subprocess.run(['cuobjdump', '-lelf', str(artifact)],
                         capture_output=True, text=True).stdout
architectures = sorted({part.split('.')[-2] for part in listing.split() if '.sm_' in part})
print('kernel cubins:', architectures)
if 'sm_120' not in architectures:
    sys.exit('kernel artifact has no sm_120 cubin: ' + ', '.join(architectures))
print('wrote', artifact)

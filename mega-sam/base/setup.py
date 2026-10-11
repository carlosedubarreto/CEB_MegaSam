import os
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

# MSVC-compatible C++17 flags on Windows, GCC/Clang flags on Linux
if os.name == 'nt':
    cxx_args = ['/O2', '/std:c++17', '/wd4819', '/DUSE_CUDA']
    nvcc_args = [
        '-O3',
        '-DUSE_CUDA',
        '-Xcompiler', '/EHsc',
        '-Xcompiler', '/std:c++17',
        '-Xcompiler', '/DUSE_CUDA',
        '-gencode=arch=compute_75,code=sm_75',
        '-gencode=arch=compute_80,code=sm_80',
        '-gencode=arch=compute_86,code=sm_86',
        '-gencode=arch=compute_89,code=sm_89',   # RTX 40xx
        '-gencode=arch=compute_120,code=sm_120', # RTX 50xx
        '-gencode=arch=compute_120,code=compute_120',
    ]
else:
    cxx_args = ['-O3', '-std=c++17']
    nvcc_args = [
        '-O3',
        '-std=c++17',
        '-gencode=arch=compute_75,code=sm_75',
        '-gencode=arch=compute_80,code=sm_80',
        '-gencode=arch=compute_86,code=sm_86',
        '-gencode=arch=compute_89,code=sm_89',
        '-gencode=arch=compute_120,code=sm_120',
        '-gencode=arch=compute_120,code=compute_120',
    ]

cuda_home = os.environ.get('CUDA_HOME', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8')
cuda_include = osp.join(cuda_home, 'include')

setup(
    name='droid_backends',
    version='0.2',
    description='DROID and LieTorch Backends',
    packages=['lietorch'],
    package_dir={'': 'thirdparty/lietorch'},
    ext_modules=[
        CUDAExtension(
            'droid_backends',
            include_dirs=[
                osp.join(ROOT, 'thirdparty/eigen'),
                cuda_include,
            ],
            sources=[
                'src/droid.cpp',
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': nvcc_args,
            },
        ),
        CUDAExtension(
            'lietorch_backends',
            include_dirs=[
                osp.join(ROOT, 'thirdparty/lietorch/lietorch/include'),
                osp.join(ROOT, 'thirdparty/eigen'),
                cuda_include,
            ],
            sources=[
                'thirdparty/lietorch/lietorch/src/lietorch.cpp',
                'thirdparty/lietorch/lietorch/src/lietorch_gpu.cu',
                'thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp',
            ],
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': nvcc_args,
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)},
)
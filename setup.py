from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# LINUS NOTE:
# Don't make me guess where your header files are.
# We explicitly include the 'include' directory.

setup(
    name='libortho_ops',
    ext_modules=[
        CUDAExtension(
            name='libortho_ops', 
            sources=[
                'src/torch_binding.cpp',
                'src/kernel_fusion.cu',
            ],
            include_dirs=[os.path.abspath('include')],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


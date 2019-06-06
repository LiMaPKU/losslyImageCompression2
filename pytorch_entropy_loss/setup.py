from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='entropy_loss_cuda',
    ext_modules=[
        CUDAExtension('entropy_loss_cuda', [
            'entropy_loss_cuda.cpp',
            'entropy_loss_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

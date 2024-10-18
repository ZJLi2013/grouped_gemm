import os
from pathlib import Path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cwd = Path(os.path.dirname(os.path.abspath(__file__)))

hipcc_flags = [
    "-std=c++17",  # NOTE: CUTLASS requires c++17
    "-DENABLE_BF16", # Enable BF16 for cuda_version >= 11
    # "-DENABLE_FP8",  # Enable FP8 for cuda_version >= 11.8
]

ext_modules = [
    CUDAExtension(
        "grouped_gemm_backend",
        ["csrc/ops.cpp", "csrc/grouped_gemm.cpp", "csrc/sinkhorn.cpp", "csrc/permute.cpp"],
        include_dirs = ["/opt/rocm/include/"],
        extra_compile_args={
            "cxx": [
                "-fopenmp", "-fPIC", "-Wno-strict-aliasing"
            ],
            "hipcc": hipcc_flags,
        }
    )
]

setup(
    name="grouped_gemm",
    version="1.1.4",
    author="Trevor Gale, Jiang Shao, Shiqing Fan",
    author_email="tgale@stanford.edu, jiangs@nvidia.com, shiqingf@nvidia.com",
    description="GEMM Grouped",
    url="https://github.com/ZJLi2013/grouped_gemm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["absl-py", "numpy", "torch"],
)

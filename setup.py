import os
from pathlib import Path
from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import BuildExtension
import torch 
from setuptools.command.build_ext import build_ext
import subprocess

hipcc_flags = [
    "--amdgpu-target=gfx942",  
    "-fPIC",
    "-DENABLE_BF16", # Enable BF16 for cuda_version >= 11
    # "-DENABLE_FP8",  # Enable FP8 for cuda_version >= 11.8
]


class HIPBuildExt(BuildExtension):
    def build_extensions(self):
        for ext in self.extensions:
            self.build_extension(ext)

    # enforce manually compile with hipcc 
    def build_extension(self, ext):
        output_path = self.get_ext_fullpath(ext.name)
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_dir)
        
        hip_compile_flags = ext.extra_compile_args
        hip_link_flags = ['-shared', '-fPIC'] + ext.extra_link_args
        include_dirs = [f'-I{dir}' for dir in ext.include_dirs]        

        # Compile HIP source files manually
        object_files = []
        for source in ext.sources:
            obj_file = source.replace(".cpp", ".o")
            compile_command = ['hipcc', '-c', '-o', obj_file, source] + hip_compile_flags + include_dirs
            print(f"Compiling {source} -> {obj_file}")
            subprocess.check_call(compile_command)
            object_files.append(obj_file)

        # Link object files to create the shared library
        link_command = ['hipcc', '-o', output_path] + object_files + hip_link_flags
        print(f"Linking to create {output_path}")
        subprocess.check_call(link_command)


hip_extension_modules = [
    Extension(
        "grouped_gemm_backend",
        ["csrc/ops.cpp", "csrc/grouped_gemm.cpp", "csrc/sinkhorn.cpp", "csrc/permute.cpp"],
        include_dirs = ["/opt/rocm/include/", torch.utils.cpp_extension.include_paths()],
        library_dirs = ["/opt/rocm/lib/", os.path.join(torch.utils.cpp_extension.library_paths()[0], 'torch/lib') ],
        libraries = ["hiprtc", "hipblas", "torch", "c10"],  
        extra_compile_args=hipcc_flags; 
        extra_link_args = ["-L/opt/rocm/lib", "-lhiprtc",  "-lamdhip64", "-lhipblas"], 
    )
]

setup(
    name="grouped_gemm",
    version="1.1.4",
    description="GEMM Grouped",
    url="https://github.com/ZJLi2013/grouped_gemm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    ext_modules=hip_extension_modules,
    cmdclass={"build_ext": HIPBuildExt},
    install_requires=["absl-py", "numpy", "torch"],
)

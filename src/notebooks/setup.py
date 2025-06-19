from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup
import glob
import os

# Current directory of setup.py (should be src/notebooks/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # go up two levels

# Source directories
autodiff_src = os.path.join(project_root, "src", "autodiff")
optimizers_src = os.path.join(project_root, "src", "optimizers")
loss_src = os.path.join(project_root, "src", "loss")

# Bindings file
bindings_file = os.path.join(current_dir, "bindings.cpp")

# Collect all .cpp files recursively in src/autodiff
cpp_files = [bindings_file]
cpp_files += glob.glob(os.path.join(autodiff_src, "**", "*.cpp"), recursive=True)

ext_modules = [
    Pybind11Extension(
        "gradientdescent",
        sources=cpp_files,
        include_dirs=[
            pybind11.get_include(),
            os.path.join(project_root, "src")
        ],
        language="c++",
        cxx_std=20,
    )
]

setup(
    name="gradientdescent",
    version="0.1.0",
    description="Gradient descent optimization and automatic differentiation module",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
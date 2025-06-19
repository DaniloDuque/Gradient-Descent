from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup
import glob
import os

# Current directory of setup.py (should be src/notebooks/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # go up two levels

autodiff_src = os.path.join(project_root, "src", "autodiff")

# Collect all .cpp files recursively in src/autodiff
cpp_files = [os.path.join(current_dir, "bindings.cpp")]  # your bindings source file
cpp_files += glob.glob(os.path.join(autodiff_src, "**", "*"), recursive=True)

print("Compiling these source files:")
for f in cpp_files:
    print(f)

ext_modules = [
    Pybind11Extension(
        "autodiff",
        sources=cpp_files,
        language="c++",
        cxx_std=20,
    )
]

setup(
    name="autodiff",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automatic differentiation library",
    long_description="A C++ automatic differentiation library with Python bindings",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=["pybind11>=2.6.0"],
)

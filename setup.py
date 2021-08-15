from setuptools import Extension, setup
from Cython.Build import cythonize
import os
import platform

if platform.system() == 'Darwin':
    os.environ['CC'] = "gcc-11"
    os.environ['CXX'] = "g++-11"

ext_modules = [
    Extension(
        "*",
        ["*.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='hello-parallel-world',
    ext_modules=cythonize(ext_modules),
)

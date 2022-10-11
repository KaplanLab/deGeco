from setuptools import Extension, setup
import os
import platform

USE_CYTHON=0

if USE_CYTHON:
    from Cython.Build import cythonize
    ext = 'pyx'
else:
    ext = 'c'

if platform.system() == 'Darwin':
    os.environ['CC'] = "gcc-12"
    os.environ['CXX'] = "g++-12"

ext_modules = [
    Extension(
        "loglikelihood",
        [f"src/loglikelihood.{ext}"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension("ssum", [f"src/ssum.{ext}"]),
    Extension("logsumexp", [f"src/logsumexp.{ext}"]),
    Extension("balance_counts", [f"src/balance_counts.{ext}"],),
    Extension("zero_sampler", [f"src/zero_sampler.{ext}"]),
    Extension("sumheap", [f"src/sumheap.{ext}"]),
]

if USE_CYTHON:
    ext_modules = cythonize(ext_modules)
setup(
    name='deGeco',
    ext_modules=ext_modules,
)

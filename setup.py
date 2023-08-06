# https://stackoverflow.com/questions/60717795/python-setup-py-sdist-with-cython-extension-pyx-doesnt-match-any-files

from setuptools import setup, find_packages, Extension
from os.path import join
import numpy as np
from Cython.Build import cythonize


extension = Extension(
    name="pyDecimation.cython._decimation",
    sources=[join("pyDecimation", "cython", "_decimation.pyx")],
    include_dirs=[np.get_include()],
)


setup(
    name="pyDecimation",
    version="0.0.1",
    description="Mesh decimation in python",
    author="Louis Pujol",
    setup_requires=["cython", "numpy", "numba"],
    install_requires=["numpy", "numba"],
    # packages=["pyDecimation", "pyDecimation.cython", "pyDecimation.numba"],
    packages=find_packages(),
    package_data={"pyDecimation.cython": ["*.pyx", "*.pxd"]},
    ext_modules=cythonize([extension]),
)

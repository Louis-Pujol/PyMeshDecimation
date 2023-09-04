# https://stackoverflow.com/questions/60717795/python-setup-py-sdist-with-cython-extension-pyx-doesnt-match-any-files

from setuptools import setup, find_packages, Extension
from os.path import join
import numpy as np
from Cython.Build import cythonize


extension = Extension(
    name="pymeshdecimation.cython._decimation",
    sources=[
        join("pymeshdecimation", "cython", "_decimation.pyx"),
    ],
    include_dirs=[np.get_include(), join("pymeshdecimation", "cython")],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name="pymeshdecimation",
    version="0.0.2",
    description="Mesh decimation in python",
    author="Louis Pujol",
    setup_requires=["cython", "numpy", "numba"],
    install_requires=["numpy", "numba"],
    packages=find_packages(),
    package_data={
        "pymeshdecimation.cython": ["*.pyx", "*.pxd"],
    },
    ext_modules=cythonize([extension]),
)

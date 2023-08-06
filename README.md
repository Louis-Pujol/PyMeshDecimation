# pyDecimation
The goal of this package is to provide a pure python implementation of the quadric decimation algorithm of VTK that keep track of the collapse operations.

# Current state :

Core functions are implemented in numba and cython and the project has started to be organize as a distribution package.

Install with :
- pip uninstall pyDecimation (if necessary)
- python -m build
- cd dist
- pip install *.whl

Then to check if the installation is correct (from the root folder of the project):
- cd test_script
- python script.py


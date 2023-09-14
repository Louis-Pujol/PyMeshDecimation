# PyMeshDecimation

Quadric Decimation for triangular meshes in Cython.

![alt text](pmd_comparison.png)

This repository is now deprecated, the goal was to implement a fast decimation algorithm
that was able to record the mapping from the vertices of the original mesh to the ones of
the decimated mesh. This implementation remained slower than VTK while trying to implement
the same algorithm.

The solution has been to contribute to [fast-simplification](https://github.com/pyvista/fast-simplification), a python wrapper of [Fast-Quadric-Mesh-Simplification](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification) to make available the required funcitonalities.

from time import time
import pyvista
import numpy as np
import pyvista.examples

import pymeshdecimation


mesh = pyvista.examples.download_bunny()
# mesh = pyvista.Sphere()
points = np.array(mesh.points, dtype=np.float64)
triangles = mesh.faces.reshape(-1, 4)[:, 1:].T


output_points, collapses, newpoints = pymeshdecimation.numba.decimate(
    points.copy(),
    triangles.copy(),
    target_reduction=0.9,
)


start = time()
output_points, collapses, newpoints = pymeshdecimation.cython.decimate(
    points.copy(),
    triangles.copy(),
    target_reduction=0.9,
)
print(f"Cython implementation : {time() - start}")

p = pyvista.Plotter()
p.add_points(output_points, color="red", point_size=10)
p.add_mesh(mesh, color="grey", opacity=0.5)
p.show()

start = time()
output_points2, collapses, newpoints = pymeshdecimation.numba.decimate(
    points.copy(),
    triangles.copy(),
    target_reduction=0.9,
)
print(f"Numba implementation : {time() - start}")

p = pyvista.Plotter()
p.add_points(output_points, color="red", point_size=10)
p.add_points(output_points2, color="green", point_size=10)
p.add_mesh(mesh, color="grey", opacity=0.5)
p.show()

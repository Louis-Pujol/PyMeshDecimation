import pymeshdecimation
import pyvista
import numpy as np

# mesh = pyvista.examples.download_cow().triangulate()
mesh = pyvista.Sphere()
points = np.array(mesh.points, dtype=np.float64)
triangles = mesh.faces.reshape(-1, 4)[:, 1:].T


#############################################################

output_points, collapses, newpoints = pymeshdecimation.cython.decimate(
    points.copy(),
    triangles.copy(),
    target_reduction=0.9,
)


output_points2 = pymeshdecimation.cython.replay_decimation(
    points.copy(),
    triangles.copy(),
    collapses,
)

p = pyvista.Plotter()
p.add_points(output_points, color="red", point_size=10)
p.add_points(output_points2, color="green", point_size=10)
p.add_mesh(mesh, color="grey", opacity=0.5)
p.show()

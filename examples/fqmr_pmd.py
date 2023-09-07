import fast_simplification
import pymeshdecimation
import pyvista
import numpy as np

mesh = pyvista.Sphere()


def triangles_to_faces(triangles):
    tmp = 3 * np.ones((len(triangles), 4), dtype=triangles.dtype)
    tmp[:, 1:] = triangles
    return tmp.copy().reshape(-1)


def faces_to_triangles(faces):
    return np.array(faces.reshape(-1, 4)[:, 1:], dtype=np.int64)


points = mesh.points
faces = mesh.faces.reshape(-1, 4)[:, 1:]

points_out, faces_out, collapses = fast_simplification.simplify(
    points, faces, 0.9, return_collapses=True
)

# recreate a mesh
mesh = pyvista.PolyData(points_out, faces=triangles_to_faces(faces_out))

# Compute mapping
mapping = pymeshdecimation.cython._compute_indice_mapping(collapses, points.shape[0])

# Replay decimation
output = pymeshdecimation.cython.replay_decimation(
    points, faces, np.array(collapses, dtype=np.int64), stop_after=420
)

# recreate a mesh
mesh2 = pyvista.PolyData(
    output.decimated_points, faces=triangles_to_faces(output.decimated_triangles)
)

p = pyvista.Plotter(shape=(1, 2))
p.subplot(0, 0)
p.add_mesh(mesh)
p.subplot(0, 1)
p.add_mesh(mesh2)
p.show()

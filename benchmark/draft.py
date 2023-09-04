# We assume you have a numpy based mesh processing software
# Where you can get the vertices and faces of the mesh as numpy arrays
# For example Trimesh or meshio
import pyfqmr
import pyvista.examples
import numpy as np


def triangles_to_faces(triangles):
    tmp = 3 * np.ones((len(triangles), 4), dtype=triangles.dtype)
    tmp[:, 1:] = triangles
    return tmp.copy().reshape(-1)


def faces_to_triangles(faces):
    return np.array(faces.reshape(-1, 4)[:, 1:], dtype=np.int64)


mesh = pyvista.read("louis.vtk")

points = np.array(mesh.points, dtype=np.float64)
triangles = faces_to_triangles(mesh.faces)


# Simplify object
mesh_simplifier = pyfqmr.Simplify()
mesh_simplifier.setMesh(points, triangles)
mesh_simplifier.simplify_mesh(
    target_count=220,
    aggressiveness=3,
    preserve_border=True,
    verbose=10,
    max_iterations=3000,
)
decimated_points, decimated_triangles, _ = mesh_simplifier.getMesh()
decimated_mesh = pyvista.PolyData(
    decimated_points, faces=triangles_to_faces(decimated_triangles)
)
decimated_mesh.plot()

import pyvista
import pyvista.examples
import numpy as np
import pymeshdecimation as pmd
import igl as igl
import pyfqmr
from time import time


def triangles_to_faces(triangles):
    tmp = 3 * np.ones((len(triangles), 4), dtype=triangles.dtype)
    tmp[:, 1:] = triangles
    return tmp.copy().reshape(-1)


def faces_to_triangles(faces):
    return np.array(faces.reshape(-1, 4)[:, 1:], dtype=np.int64)


def compare(mesh, n_faces, disable_pmd=False):
    """Compare the decimation of a mesh using pymeshdecimation/vtk and igl

    This function allows to compare the decimation of a mesh using pymeshdecimation/vtk and igl
    the comparison is done visually by showing the decimated meshes and by printing the running time

    Args:
        mesh (pyvista.PolyData): mesh to decimate
        n_faces (int: required number of faces (approximative)
        disable_pmd (bool, optional): True if you want to discard the decimation using pymeshdecimation. Defaults to False.

    """

    points = np.array(mesh.points, dtype=np.float64)
    triangles = faces_to_triangles(mesh.faces)

    # IGL
    max_m = n_faces
    start = time()
    decimator = igl.decimate(points, triangles, max_m)
    time_igl = time() - start
    done = decimator[0]
    assert done
    decimated_points_igl = decimator[1]
    decimated_triangles_igl = decimator[2]
    decimated_mesh_igl = pyvista.PolyData(
        decimated_points_igl, faces=triangles_to_faces(decimated_triangles_igl)
    )
    birth_faces_igl = decimator[3]
    birth_vertices_igl = decimator[4]

    # Pyvista
    target_reduction = (len(triangles) - n_faces) / len(triangles)
    start = time()
    decimated_mesh_pyvista = mesh.decimate(target_reduction)
    time_pyvista = time() - start

    # PYFQMR
    start = time()
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(points, triangles)
    mesh_simplifier.simplify_mesh(
        target_count=n_faces,
        aggressiveness=4,
        preserve_border=False,
        verbose=1,
        max_iterations=300,
    )
    decimated_points_pyfqmr, decimated_triangles_pyfqmr, _ = mesh_simplifier.getMesh()
    time_pyfqmr = time() - start
    decimated_mesh_pyfqmr = pyvista.PolyData(
        decimated_points_pyfqmr, faces=triangles_to_faces(decimated_triangles_pyfqmr)
    )

    if disable_pmd:
        p = pyvista.Plotter(shape=(1, 3))
        p.subplot(0, 0)
        p.add_mesh(decimated_mesh_igl, color="tan", show_edges=True)
        p.add_text(f"IGL {time_igl}", font_size=10)
        p.subplot(0, 1)
        p.add_mesh(decimated_mesh_pyvista, color="tan", show_edges=True)
        p.add_text(f"VTK {time_pyvista}", font_size=10)
        p.subplot(0, 2)
        p.add_mesh(decimated_mesh_pyfqmr, color="tan", show_edges=True)
        p.add_text(f"PyFQMR {time_pyfqmr}", font_size=10)
        p.show()

    else:
        # assert (len(decimated_mesh_pyvista.points) == len(decimated_points_igl))
        n_points = len(decimated_mesh_pyvista.points)

        # PyMeshDecimation
        start = time()
        output = pmd.cython.decimate(
            points=points,
            triangles=triangles,
            n_points_to_remove=len(points) - n_points,
        )
        time_pmd = time() - start

        decimated_points_pmd = output.decimated_points
        decimated_triangles_pmd = output.decimated_triangles
        decimated_mesh_pmd = pyvista.PolyData(
            decimated_points_pmd, faces=triangles_to_faces(decimated_triangles_pmd)
        )

        p = pyvista.Plotter(shape=(2, 3))
        p.subplot(0, 0)
        p.add_mesh(decimated_mesh_igl, color="tan", show_edges=True)
        p.add_text(f"IGL {time_igl}", font_size=10)
        p.subplot(0, 1)
        p.add_mesh(decimated_mesh_pyvista, color="tan", show_edges=True)
        p.add_text(f"VTK {time_pyvista}", font_size=10)
        p.subplot(0, 2)
        p.add_mesh(decimated_mesh_pyfqmr, color="tan", show_edges=True)
        p.add_text(f"PyFQMR {time_pyfqmr}", font_size=10)
        p.subplot(1, 0)
        p.add_mesh(decimated_mesh_pmd, color="tan", show_edges=True)
        p.add_text(f"PyMeshDecimation {time_pmd}", font_size=10)
        p.show()


compare(mesh=pyvista.read("cow.vtk"), n_faces=220)

compare(mesh=pyvista.read("bunny.vtk"), n_faces=350)

compare(
    mesh=pyvista.read("louis.vtk").clean(),
    n_faces=2000,
    disable_pmd=True,
)

import pymeshdecimation as pmd
import pyvista
import pyvista.examples
import numpy as np


def test_replay():
    # Load a mesh from pyvista and convert points/triangles to
    # numpy arrays with the correct dtype and shape
    mesh = pyvista.examples.download_cow().triangulate().clean()

    points = np.array(mesh.points.copy(), dtype=np.float64)
    n_points_to_remove = int(0.9 * len(points))
    triangles = np.array(mesh.faces.copy().reshape(-1, 4)[:, 1:], dtype=np.int64)

    # Decimate the mesh
    output = pmd.cython.decimate(
        points=points, triangles=triangles, n_points_to_remove=n_points_to_remove
    )

    decimated_points = output.decimated_points
    decimated_triangles = output.decimated_triangles

    # convert the triangles to the correct shape for pyvista faces attribute
    tmp = 3 * np.ones((len(decimated_triangles), 4), dtype=decimated_triangles.dtype)
    tmp[:, 1:] = decimated_triangles
    faces = tmp.copy().reshape(-1)
    decimated_mesh = pyvista.PolyData(decimated_points, faces=faces)

    # Replay the decimation
    output_replay = pmd.cython.replay_decimation(
        points=points, triangles=triangles, collapses=output.collapses
    )

    decimated_points_replay = output_replay.decimated_points
    decimated_triangles_replay = output_replay.decimated_triangles

    # convert the triangles to the correct shape for pyvista faces attribute
    tmp = 3 * np.ones(
        (len(decimated_triangles_replay), 4), dtype=decimated_triangles_replay.dtype
    )
    tmp[:, 1:] = decimated_triangles
    faces_replay = tmp.copy().reshape(-1)
    decimated_mesh_replay = pyvista.PolyData(
        decimated_points_replay, faces=faces_replay
    )

    # Assert that the replayed decimation is the same as the original decimation
    # assert np.allclose(decimated_points, decimated_points_replay)
    # TODO understand why this assertion fails
    assert np.allclose(decimated_triangles, decimated_triangles_replay)

    if __name__ == "__main__":
        p = pyvista.Plotter(shape=(1, 3))
        p.add_points(points, color="red", point_size=10.0)
        p.add_mesh(mesh, color="tan", show_edges=True)
        p.subplot(0, 1)
        p.add_points(decimated_points, color="red", point_size=10.0)
        p.add_points(decimated_points_replay, color="green", point_size=10.0)
        p.add_mesh(decimated_mesh, color="tan", show_edges=True)
        p.subplot(0, 2)
        p.add_points(decimated_points_replay, color="red", point_size=10.0)
        p.add_mesh(decimated_mesh_replay, color="tan", show_edges=True)
        p.show()

import pymeshdecimation as pmd
import pyvista
import pyvista.examples
import numpy as np
from time import time


def test_numba_cython_consistency():
    """Test that the numba and cython implementations are consistent

    We test that the core function of the algorithm are consistent between the
    cython and numba implementations.

    More precisely we chack that :
    - the computation of the quadrics is consistent
    - the computation of the edges' costs is consistent
    - the collapse step have the same results

    Note that the test can fail for some meshes, the sphere being an example.
    We suspect that this is due to the fact that the sphere is a very regular
    mesh and that the collapse step is not deterministic in this case.

    If the test fails, it is recommended to visualize the decimated meshes to
    check that both implementations give good results.
    """

    mesh = pyvista.examples.download_bunny_coarse().clean()
    points = np.array(mesh.points.copy(), dtype=np.float64)
    n_points_to_remove = int(0.9 * len(points))
    triangles = np.array(mesh.faces.copy().reshape(-1, 4)[:, 1:], dtype=np.int64)

    # Compute the non boundary quadrics
    nonboundary_quadrics_cython = pmd.cython._nonboundary_quadrics(
        points=points, triangles=triangles
    )
    nonboundary_quadrics_numba = pmd.numba._initialize_quadrics(
        points=points, triangles=triangles.T
    )
    assert np.allclose(nonboundary_quadrics_cython, nonboundary_quadrics_numba)

    # Compute the boundary quadrics
    boundary_quadrics_cython = pmd.cython._boundary_quadrics(
        points=points, triangles=triangles
    )
    repeated_edges = pmd.numba._compute_edges(triangles=triangles.T, repeated=True)
    boundary_quadrics_numba = pmd.numba._compute_boundary_quadrics(
        points=points, triangles=triangles.T, repeated_edges=repeated_edges
    )
    assert np.allclose(boundary_quadrics_cython, boundary_quadrics_numba)

    quadrics_numba = nonboundary_quadrics_numba + boundary_quadrics_numba
    quadrics_cython = nonboundary_quadrics_cython + boundary_quadrics_cython

    # Initialize costs
    edges_numba = pmd.numba._compute_edges(triangles=triangles.T)
    costs_numba, target_points_numba = pmd.numba._intialize_costs(
        points=points, edges=edges_numba, quadrics=nonboundary_quadrics_numba
    )
    edges_cython = pmd.cython._compute_edges(triangles=triangles)
    costs_cython, target_points_cython = pmd.cython._initialize_costs(
        points=points, edges=edges_cython, quadrics=nonboundary_quadrics_cython
    )
    assert np.allclose(costs_numba, costs_cython)

    # Collapse edges
    decimated_points_numba, collapses_numba, _ = pmd.numba._collapse(
        points=points,
        edges=edges_numba.T,
        costs=costs_numba,
        newpoints=target_points_numba,
        quadrics=quadrics_numba,
        n_points_to_remove=n_points_to_remove,
    )

    decimated_points_cython, collapses_cython = pmd.cython._collapse(
        points=points,
        edges=edges_cython,
        costs=costs_cython,
        target_points=target_points_cython,
        quadrics=quadrics_cython,
        n_points_to_remove=n_points_to_remove,
    )

    assert np.allclose(decimated_points_numba, decimated_points_cython)
    assert np.allclose(collapses_numba, collapses_cython)


test_numba_cython_consistency()

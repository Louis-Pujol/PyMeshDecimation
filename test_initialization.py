"""This file defines functionnal tests for the initialization of the quadricDecimation algorithm.

The initialization if the computatios that are done before starting collapsing edges.
The variables that are computed are:
    - The quadrics for each vertex
    - The cost for each edge
    - The target point for each edge

The test is done by comparing these variables obtained with the C++ VTK implementation and ours. To be runned,
the vtk module must be compiled from the code at https://github.com/Louis-Pujol/VTK/ to have access to the
GetInitialTargetPoints, GetInitialEdges and GetInitialEdgeCosts methods of vtk. Using the standard VTK module,
these variables are not accessible.
"""

import pyvista
import numpy as np
from pyvista.core.filters import _get_output, _update_alg
import vtk
from QuadricDecimation_numba import *


def foo(file):
    mesh = pyvista.read(file)

    # With VTK
    alg = vtk.vtkQuadricDecimation()
    alg.SetVolumePreservation(False)
    alg.SetAttributeErrorMetric(False)
    alg.SetScalarsAttribute(True)
    alg.SetVectorsAttribute(True)
    alg.SetNormalsAttribute(False)
    alg.SetTCoordsAttribute(True)
    alg.SetTensorsAttribute(True)
    alg.SetScalarsWeight(0.1)
    alg.SetVectorsWeight(0.1)
    alg.SetNormalsWeight(0.1)
    alg.SetTCoordsWeight(0.1)
    alg.SetTensorsWeight(0.1)
    alg.SetTargetReduction(0.5)
    alg.SetInputData(mesh)

    progress_bar = False
    _update_alg(alg, progress_bar, "Decimating Mesh")
    decimated_mesh = _get_output(alg)

    initialPoints_vtk = pyvista.convert_array(alg.GetInitialPoints().GetData())
    initialTargetPoints_vtk = pyvista.convert_array(alg.GetInitialTargetPoints())
    edges_vtk = pyvista.convert_array(alg.GetInitialEdges())
    edgeCosts_vtk = pyvista.convert_array(alg.GetInitialEdgeCosts())
    initialQuadrics_vtk = pyvista.convert_array(alg.GetInitialQuadrics())

    # Sort edges_vtk (and consequently edgeCosts_vtk and newpoints_vtk) by lexicographic order to match the order of edges in numba implementation
    edges_vtk.sort(axis=1)
    ordering = np.lexsort((edges_vtk[:, 1], edges_vtk[:, 0]))
    edges_vtk = edges_vtk[ordering]
    edgeCosts_vtk = edgeCosts_vtk[ordering]
    initialTargetPoints_vtk = initialTargetPoints_vtk[ordering]

    vtk_data = {
        "initialPoints": initialPoints_vtk,
        "initialTargetPoints": initialTargetPoints_vtk,
        "initialEdges": edges_vtk,
        "initialEdgeCosts": edgeCosts_vtk,
        "initialQuadrics": initialQuadrics_vtk,
    }

    # With Numba
    points = mesh.points
    triangles = mesh.faces.reshape(-1, 4)[:, 1:].T
    quadrics = initialize_quadrics_numba(points, triangles)
    # Are there boundary edges?
    repeated_edges = compute_edges(triangles, repeated=True)
    check_boundary_constraints_numba(repeated_edges)
    # Compute the cost for each edge
    edges = compute_edges(triangles)
    costs, target_points = intialize_costs(edges, quadrics, points)

    numba_data = {
        "initialPoints": points,
        "initialTargetPoints": target_points,
        "initialEdges": edges.T,
        "initialEdgeCosts": costs,
        "initialQuadrics": quadrics,
    }

    return vtk_data, numba_data


def test_on_scape():
    for i in range(6):
        vtk_data, numba_data = foo("data/mesh00{}.ply".format(i))
        for key in numba_data.keys():
            assert np.allclose(
                vtk_data[key], numba_data[key], rtol=1e-1, atol=1e-1
            ), "Error in {}".format(key)


def test_on_bunny_coarse():
    mesh = pyvista.read("data/bunny_coarse.ply")
    mesh.plot(show_edges=True)
    vtk_data, numba_data = foo("data/bunny_coarse.ply")
    for key in numba_data.keys():
        assert np.allclose(
            vtk_data[key], numba_data[key], rtol=1e-1, atol=1e-1
        ), "Error in {}".format(key)


def test_on_bunny():
    mesh = pyvista.read("data/bunny.ply")
    mesh.plot(show_edges=True)
    vtk_data, numba_data = foo("data/bunny.ply")
    for key in numba_data.keys():
        assert np.allclose(
            vtk_data[key], numba_data[key], rtol=1e-1, atol=1e-1
        ), "Error in {}".format(key)


if __name__ == "__main__":
    test_on_scape()
    test_on_bunny()

"""This script compares the running time of the initialization of the quadrics using numpy and numba.
The numpy implementation takes benefit of the vectorization of numpy, while the numba implementation
is closer to the original C++ code.

A first run of both functions is done to assert that the results are the same. Then, the running time
of both functions is compared.

The results show that the numba implementation is faster than the numpy implementation.
"""

import numpy as np
import numba as nb


@nb.jit(nopython=True, fastmath=True, cache=True)
def initialize_quadrics_numba(vertices, triangles):
    quadrics = np.zeros((vertices.shape[0], 11))

    for i in range(triangles.shape[1]):
        p0, p1, p2 = vertices[triangles[:, i], :]
        n = np.cross(p1 - p0, p2 - p0)
        area2 = np.sqrt((np.sum(n * n))) / 2

        n /= 2 * area2
        assert np.isclose(np.sum(n * n), 1)

        d = -np.sum(n * p0)

        # Compute the quadric for this triangle
        tmp = np.zeros(4)
        tmp[0:3] = n
        tmp[3] = d

        Q = np.zeros(11 + 4 * 3)
        Q[0] = n[0] * n[0]
        Q[1] = n[0] * n[1]
        Q[2] = n[0] * n[2]
        Q[3] = n[0] * d
        Q[4] = n[1] * n[1]
        Q[5] = n[1] * n[2]
        Q[6] = n[1] * d
        Q[7] = n[2] * n[2]
        Q[8] = n[2] * d
        Q[9] = d * d
        Q[10] = 1
        Q = Q * area2

        for j in triangles[:, i]:
            quadrics[j, 0:11] += Q

    # Normalize the quadrics
    return quadrics


def intitialize_quadrics_numpy(vertices, triangles, debug=False):
    n_points = vertices.shape[0]
    n_triangles = triangles.shape[1]

    A = vertices[triangles[0]]
    B = vertices[triangles[1]]
    C = vertices[triangles[2]]

    ns = np.cross(B - A, C - A)  # Normals of the triangles
    assert ns.shape == (n_triangles, 3)  # Check that the shape is correct

    i = np.random.randint(n_triangles)  # Pick a random triangle
    p0, p1, p2 = vertices[triangles[:, i], :]  # Get the vertices of the triangle
    n = np.cross(p1 - p0, p2 - p0)  # Compute the normal
    assert np.allclose(ns[i], n)  # Check that the i's entry of ns is correct

    areas = np.linalg.norm(ns, axis=1) / 2  # Areas of the triangles
    assert areas.shape == (n_triangles,)  # Check that the shape is correct

    ns /= 2 * areas[:, None]  # Normalize the normals
    assert np.allclose(
        np.linalg.norm(ns, axis=1), 1
    )  # Assert that the normals are actually normalized

    ds = -np.sum(ns * A, axis=1)  # Distance of the plane to the origin

    tmp = np.concatenate([ns, ds[:, None]], axis=1)
    assert np.allclose(ns[0], tmp[0, :3]) and np.isclose(
        ds[0], tmp[0, 3]
    )  # Check that the concatenation is correct

    Qs = (
        tmp[:, None, :] * tmp[:, :, None] * areas[:, None, None]
    )  # The quadric for each triangle (n_triangles, 4, 4)
    assert Qs.shape == (n_triangles, 4, 4)  # Check that the shape is correct

    i = np.random.randint(n_triangles)  # Pick a random triangle
    j, k = np.random.randint(4, size=2)  # Pick two random indices
    assert np.isclose(
        Qs[i, j, k], tmp[i, j] * tmp[i, k] * areas[i]
    )  # Check that the outer product is correct for the random triangle and indices

    # Compute the quadric for each vertex
    Quadrics = np.zeros((vertices.shape[0], 4, 4))
    # scatter summation (Quadrics[triangles[0]] += Qs does not work because of the repeated indices)
    np.add.at(Quadrics, triangles[0], Qs)
    np.add.at(Quadrics, triangles[1], Qs)
    np.add.at(Quadrics, triangles[2], Qs)

    ############################
    # Flat version of the quadrics (VTK implementation)
    ############################

    Qs_flat = np.zeros((triangles.shape[1], 11))
    # Qs_flat[:, :10] = Qs[:, *np.triu_indices(4)] # Extract the upper triangular part of the quadrics
    triu_indices = (
        np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]),
        np.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]),
    )
    Qs_flat[:, :10] = Qs[:, triu_indices[0], triu_indices[1]]
    Qs_flat[:, 10] = areas

    i = np.random.randint(n_triangles)  # Pick a random triangle
    # Check that the flat version is correct for the random triangle
    if debug:
        assert np.isclose(Qs_flat[i, 0], ns[i, 0] * ns[i, 0] * areas[i])
        assert np.isclose(Qs_flat[i, 1], ns[i, 0] * ns[i, 1] * areas[i])
        assert np.isclose(Qs_flat[i, 2], ns[i, 0] * ns[i, 2] * areas[i])
        assert np.isclose(Qs_flat[i, 3], ns[i, 0] * ds[i] * areas[i])
        assert np.isclose(Qs_flat[i, 4], ns[i, 1] * ns[i, 1] * areas[i])
        assert np.isclose(Qs_flat[i, 5], ns[i, 1] * ns[i, 2] * areas[i])
        assert np.isclose(Qs_flat[i, 6], ns[i, 1] * ds[i] * areas[i])
        assert np.isclose(Qs_flat[i, 7], ns[i, 2] * ns[i, 2] * areas[i])
        assert np.isclose(Qs_flat[i, 8], ns[i, 2] * ds[i] * areas[i])
        assert np.isclose(Qs_flat[i, 9], ds[i] * ds[i] * areas[i])
        assert np.isclose(Qs_flat[i, 10], areas[i])

    # Compute the quadric for each vertex
    Quadrics_flat = np.zeros((vertices.shape[0], 11))
    # scatter summation (Quadrics_flat[triangles[0]] += Qs_flat does not work because of the repeated indices)
    np.add.at(Quadrics_flat, triangles[0], Qs_flat)
    np.add.at(Quadrics_flat, triangles[1], Qs_flat)
    np.add.at(Quadrics_flat, triangles[2], Qs_flat)

    i = np.random.randint(n_points)  # Pick a random vertex
    # Check that the flat version is correct for the random vertex
    if debug:
        assert np.allclose(
            np.sum(Qs_flat[np.where(triangles == i)[1]], axis=0), Quadrics_flat[i]
        )

    return Quadrics_flat


vertices = np.random.rand(100, 3)
triangles = np.random.randint(0, 100, (3, 20))

import pyvista

mesh = pyvista.read("data/mesh000.ply")

vertices = mesh.points
triangles = mesh.faces.reshape(-1, 4)[:, 1:].T

a = initialize_quadrics_numba(vertices, triangles)
b = intitialize_quadrics_numpy(vertices, triangles)

assert np.allclose(a, b)

import time

numba_time = []
numpy_time = []
for i in range(10):
    start = time.time()
    initialize_quadrics_numba(vertices, triangles)
    end = time.time()
    numba_time.append(end - start)

    start = time.time()
    intitialize_quadrics_numpy(vertices, triangles)
    end = time.time()
    numpy_time.append(end - start)


print("Numba time: ", np.mean(numba_time))
print("Numpy time: ", np.mean(numpy_time))

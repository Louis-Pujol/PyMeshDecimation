from time import time
import pyvista

start = time()

print(f"Import cython implementation : {time() - start}")
import numpy as np

import pyvista.examples

import pyDecimation


# mesh = pyvista.examples.download_bunny()
mesh = pyvista.Sphere()
points = np.array(mesh.points, dtype=np.float64)
triangles = mesh.faces.reshape(-1, 4)[:, 1:].T


#############################################################

print("Initialize quadrics")


quadrics1 = pyDecimation.cython._initialize_quadrics(
    points=points.copy(), triangles=triangles.copy()
)
quadrics2 = pyDecimation.numba._initialize_quadrics(
    points=points.copy(), triangles=triangles.copy()
)

assert np.allclose(quadrics1, quadrics2)

start = time()
for _ in range(1):
    pyDecimation.cython._initialize_quadrics(
        points=points.copy(), triangles=triangles.copy()
    )
print(f"Cython implementation : {time() - start}")

start = time()
for _ in range(1):
    pyDecimation.numba._initialize_quadrics(
        points=points.copy(), triangles=triangles.copy()
    )
print(f"Numba implementation : {time() - start}")
print()

##########################################################################

print("Compute boundary quadrics")

repeated_edges = pyDecimation.numba._compute_edges(
    triangles=triangles.copy(), repeated=True
)
edges = pyDecimation.numba._compute_edges(triangles=triangles.copy(), repeated=False)

boundary_quadrics1 = pyDecimation.cython._compute_boundary_quadrics(
    points=points.copy(),
    repeated_edges=repeated_edges.copy(),
    triangles=triangles.copy(),
)
boundary_quadrics2 = pyDecimation.numba._compute_boundary_quadrics(
    points=points.copy(),
    repeated_edges=repeated_edges.copy(),
    triangles=triangles.copy(),
)

assert np.allclose(boundary_quadrics1, boundary_quadrics2)

start = time()
for _ in range(1):
    pyDecimation.cython._compute_boundary_quadrics(
        points=points.copy(),
        repeated_edges=repeated_edges.copy(),
        triangles=triangles.copy(),
    )
print(f"Cython implementation : {time() - start}")

start = time()
for _ in range(1):
    pyDecimation.numba._compute_boundary_quadrics(
        points=points.copy(),
        repeated_edges=repeated_edges.copy(),
        triangles=triangles.copy(),
    )
print(f"Numba implementation : {time() - start}")
print()

#########################################"
print("Det 3x3")
A = np.random.rand(9).reshape((3, 3))
A = A + A.T
A_line = A.reshape(-1)
pyDecimation.numba.det3x3(A)
det = np.linalg.det(A)

start = time()
for i in range(1000):
    pyDecimation.cython.det3x3(A)
print(f"Cython implementation : {time() - start}")
assert np.allclose(det, pyDecimation.cython.det3x3(A))

start = time()
for i in range(1000):
    pyDecimation.numba.det3x3(A)
print(f"Numba implementation : {time() - start}")
assert np.allclose(det, pyDecimation.numba.det3x3(A))

print()
print("Solver 3x3")
b = np.array([1, 2, 3], dtype=np.float64)
sol = np.linalg.solve(A, b)
pyDecimation.numba.solve3x3(A, b)

start = time()
for i in range(1000):
    x = pyDecimation.cython.solve3x3(A, b)
print(f"Cython implementation : {time() - start}")
print(x)
print(sol)

start = time()
for i in range(1000):
    x = pyDecimation.numba.solve3x3(A, b)
print(f"Numba implementation : {time() - start}")
assert np.allclose(x, sol)
print()


#########################################################
print("Compute cost")

Q1 = quadrics1 + boundary_quadrics1
Q2 = quadrics2 + boundary_quadrics2
edge = edges[:, 2]
pyDecimation.numba._compute_cost(
    edge=edge.copy(), quadrics=Q2.copy(), points=points.copy()
)

start = time()
cost, x = pyDecimation.cython._compute_cost(edge=edge, quadrics=Q1, points=points)
print(f"Cython implementation : {time() - start}")
start = time()
cost, x = pyDecimation.numba._compute_cost(
    edge=edge.copy(), quadrics=Q2.copy(), points=points.copy()
)
print(f"Numba implementation : {time() - start}")

print()

#############################################################
print("Initialize costs")

costs1, newpoints1 = pyDecimation.cython._intialize_costs(
    edges=edges.copy(), quadrics=Q1.copy(), points=points.copy()
)
costs2, newpoints2 = pyDecimation.numba._intialize_costs(
    edges=edges.copy(), quadrics=Q2.copy(), points=points.copy()
)


assert np.allclose(costs1, costs2)
assert np.allclose(newpoints1, newpoints2)

start = time()
for _ in range(1):
    pyDecimation.cython._intialize_costs(
        edges=edges.copy(), quadrics=Q1.copy(), points=points.copy()
    )
print(f"Cython implementation : {time() - start}")

start = time()
for _ in range(1):
    pyDecimation.numba._intialize_costs(
        edges=edges.copy(), quadrics=Q2.copy(), points=points.copy()
    )
print(f"Numba implementation : {time() - start}")

print()

#############################################################
print("Compute collapses")
target_reduction = 0.9
n_points_to_remove = int(target_reduction * points.shape[0])


points_copy = points.copy()
edges_copy = edges.copy()
costs_copy = costs2.copy()
newpoints_copy = newpoints2.copy()
Q2_copy = Q2.copy()

new_vertices2, collapses2, newpoints_history2 = pyDecimation.numba._collapse(
    edges=edges.copy().T,
    costs=costs2.copy(),
    newpoints=newpoints2.copy(),
    quadrics=Q2.copy(),
    points=points.copy(),
    n_points_to_remove=n_points_to_remove,
)


start = time()
new_vertices1, collapses1, newpoints_history1 = pyDecimation.cython._collapse(
    edges=np.ascontiguousarray(edges.T).copy(),
    costs=costs1.copy(),
    newpoints=newpoints1.copy(),
    quadrics=Q1.copy(),
    points=points.copy(),
    n_points_to_remove=n_points_to_remove,
)
print(f"Cython implementation : {time() - start}")

start = time()
new_vertices2, collapses2, newpoints_history2 = pyDecimation.numba._collapse(
    edges=edges.T,
    costs=costs2,
    newpoints=newpoints2,
    quadrics=Q2,
    points=points,
    n_points_to_remove=n_points_to_remove,
)
print(f"Numba implementation : {time() - start}")

print()

#############################################################
print("Total decimation")

output_points, collapses, newpoints = pyDecimation.numba.decimate(
    points.copy(),
    triangles.copy(),
    target_reduction=0.9,
)


start = time()
output_points, collapses, newpoints = pyDecimation.cython.decimate(
    points.copy(),
    triangles.copy(),
    target_reduction=0.9,
)
print(f"Cython implementation : {time() - start}")

start = time()
output_points, collapses, newpoints = pyDecimation.numba.decimate(
    points.copy(),
    triangles.copy(),
    target_reduction=0.9,
)
print(f"Numba implementation : {time() - start}")

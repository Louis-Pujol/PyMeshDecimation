# import pymeshdecimation
# import pyvista
# import numpy as np


# def test_consistency():

#     mesh = pyvista.Sphere()
#     # mesh = pyvista.examples.download_bunny()
#     mesh = pyvista.Sphere()
#     points = np.array(mesh.points, dtype=np.float64)
#     triangles = mesh.faces.reshape(-1, 4)[:, 1:].T

#     quadrics1 = pymeshdecimation.cython._initialize_quadrics(
#         points=points.copy(), triangles=triangles.copy()
#     )
#     quadrics2 = pymeshdecimation.numba._initialize_quadrics(
#         points=points.copy(), triangles=triangles.copy()
#     )

#     assert np.allclose(quadrics1, quadrics2)



#     print("Compute boundary quadrics")

#     repeated_edges = pymeshdecimation.numba._compute_edges(
#         triangles=triangles.copy(), repeated=True
#     )
#     repeated_edges2 = pymeshdecimation.cython._compute_edges(
#         triangles=triangles.copy(), repeated=True
#     )

#     edges = pymeshdecimation.numba._compute_edges(triangles=triangles.copy(), repeated=False)
#     edges2 = pymeshdecimation.cython._compute_edges(triangles=triangles.copy(), repeated=False)

#     assert np.allclose(repeated_edges, repeated_edges2)
#     assert np.allclose(edges, edges2)

#     boundary_quadrics1 = pymeshdecimation.cython._compute_boundary_quadrics(
#         points=points.copy(),
#         repeated_edges=repeated_edges.copy(),
#         triangles=triangles.copy(),
#     )
#     boundary_quadrics2 = pymeshdecimation.numba._compute_boundary_quadrics(
#         points=points.copy(),
#         repeated_edges=repeated_edges.copy(),
#         triangles=triangles.copy(),
#     )

#     assert np.allclose(boundary_quadrics1, boundary_quadrics2)


#     Q1 = quadrics1 + boundary_quadrics1
#     Q2 = quadrics2 + boundary_quadrics2



#     costs1, newpoints1 = pymeshdecimation.cython._intialize_costs(
#         edges=edges.copy(), quadrics=Q1.copy(), points=points.copy()
#     )
#     costs2, newpoints2 = pymeshdecimation.numba._intialize_costs(
#         edges=edges.copy(), quadrics=Q2.copy(), points=points.copy()
#     )


#     assert np.allclose(costs1, costs2)
#     assert np.allclose(newpoints1, newpoints2)
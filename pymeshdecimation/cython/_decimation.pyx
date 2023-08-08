cimport cython


from libc.math cimport sqrt
cimport numpy as cnp
cnp.import_array()
import numpy as np


INT_DTYPE = np.int64
FLOAT_DTYPE = np.double

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef cnp.int64_t INT_DTYPE_t
ctypedef cnp.double_t FLOAT_DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def det3x3(FLOAT_DTYPE_t[:, :] mat):
    cdef FLOAT_DTYPE_t det

    det = (
        mat[0, 0] * (mat[1, 1] * mat[2, 2] - mat[2, 1] * mat[1, 2])
        - mat[0, 1] * (mat[1, 0] * mat[2, 2] - mat[1, 2] * mat[2, 0])
        + mat[0, 2] * (mat[1, 0] * mat[2, 1] - mat[1, 1] * mat[2, 0])
    )
    return det

@cython.boundscheck(False)
@cython.wraparound(False)
def solve3x3(FLOAT_DTYPE_t[:, :] A, FLOAT_DTYPE_t[:] b):

    cdef FLOAT_DTYPE_t d, d1, d2, d3
    cdef cnp.ndarray[FLOAT_DTYPE_t, ndim=1] x = np.zeros([3], dtype=FLOAT_DTYPE)

    cdef FLOAT_DTYPE_t a00 = A[0, 0]
    cdef FLOAT_DTYPE_t a01 = A[0, 1]
    cdef FLOAT_DTYPE_t a02 = A[0, 2]
    cdef FLOAT_DTYPE_t a11 = A[1, 1]
    cdef FLOAT_DTYPE_t a12 = A[1, 2]
    cdef FLOAT_DTYPE_t a22 = A[2, 2]
    cdef FLOAT_DTYPE_t b0 = b[0]
    cdef FLOAT_DTYPE_t b1 = b[1]
    cdef FLOAT_DTYPE_t b2 = b[2]

    d = det3x3(A)

    d1 = (
        b0 * (a11 * a22 - a12 * a12)
        - b1 * (a01 * a22 - a02 * a12)
        + b2 * (a01 * a12 - a02 * a11)
    )
    
    d2 = (
        a00 * (b1 * a22 - b2 * a12)
        - a01 * (b0 * a22 - b2 * a02)
        + a02 * (b0 * a12 - b1 * a02)
    )


    d3 = (
        a00 * (a11 * b2 - a12 * b1)
        - a01 * (a01 * b2 - a12 * b0)
        + a02 * (a01 * b1 - a11 * b0)
    )

    x[0] = d1 / d
    x[1] = d2 / d
    x[2] = d3 / d

    return x

@cython.boundscheck(False)
@cython.wraparound(False)
def _initialize_quadrics(FLOAT_DTYPE_t[:, :] points, INT_DTYPE_t[:, :] triangles):
    """Compute the quadrics for the vertices of the mesh ignoring boundaries.

    Args:
        points (Nx3 float64 array): The points of the mesh.
        triangles (Kx3 int64 array): The triangles of the mesh.

    Returns:
        Nx11 float64 array: The quadrics for the vertices of the mesh.
    """

    cdef int n_points = points.shape[0]
    cdef int n_triangles = triangles.shape[0]
    cdef FLOAT_DTYPE_t[:, :] quadrics = np.zeros([n_points, 11], dtype=FLOAT_DTYPE)
    # cdef FLOAT_DTYPE_t[:, :] quadrics_view = quadrics
    cdef cnp.ndarray[FLOAT_DTYPE_t, ndim=1] Q = np.zeros([11], dtype=FLOAT_DTYPE)
    cdef int i, j, k
    cdef FLOAT_DTYPE_t d

    cdef cnp.ndarray[FLOAT_DTYPE_t, ndim=1] p0 = np.zeros([3], dtype=FLOAT_DTYPE)
    cdef cnp.ndarray[FLOAT_DTYPE_t, ndim=1] p1 = np.zeros([3], dtype=FLOAT_DTYPE)
    cdef cnp.ndarray[FLOAT_DTYPE_t, ndim=1] p2 = np.zeros([3], dtype=FLOAT_DTYPE)
    cdef cnp.ndarray[FLOAT_DTYPE_t, ndim=1] n = np.zeros([3], dtype=FLOAT_DTYPE)


    for i in range(n_triangles):

        # Get the points of the triangle
        for j in range(3):
            p0[j] = points[triangles[i, 0], j]
            p1[j] = points[triangles[i, 1], j]
            p2[j] = points[triangles[i, 2], j]

        # Compute the normal of the triangle
        n[0] = (p1[1] - p0[1]) * (p2[2] - p0[2]) - (p1[2] - p0[2]) * (p2[1] - p0[1])
        n[1] = (p1[2] - p0[2]) * (p2[0] - p0[0]) - (p1[0] - p0[0]) * (p2[2] - p0[2])
        n[2] = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])

        # Compute the area of the triangle
        area2 = sqrt((n[0] * n[0] + n[1] * n[1] + n[2] * n[2])) / 2

        # Normalize the normal
        n[0] /= 2 * area2
        n[1] /= 2 * area2
        n[2] /= 2 * area2
        d = -(n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2])


        # Compute the quadric for this triangle
        Q[0] = n[0] * n[0] * area2
        Q[1] = n[0] * n[1] * area2
        Q[2] = n[0] * n[2] * area2
        Q[3] = n[0] * d * area2
        Q[4] = n[1] * n[1] * area2
        Q[5] = n[1] * n[2] * area2
        Q[6] = n[1] * d * area2
        Q[7] = n[2] * n[2] * area2
        Q[8] = n[2] * d * area2
        Q[9] = d * d * area2
        Q[10] = area2

        for j in range(3):
            for k in range(11):
                quadrics[triangles[i, j], k] += Q[k]

    return np.asarray(quadrics)

@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_boundary_quadrics(FLOAT_DTYPE_t[:, :] points, INT_DTYPE_t[:, :] repeated_edges, INT_DTYPE_t[:, :] triangles):
    """Compute the quadrics for the boundary edges of the mesh.

    Args:
        points (Nx3 float64 array): The points of the mesh.
        repeated_edges (M'x2 int64 array): The repeated edges of the mesh.
        triangles (Kx3 int64 array): The triangles of the mesh.

    Returns:
        boundary_quadrics (Nx11 float64 array): The quadrics associated with the boundary.
    """

    cdef int n_points = points.shape[0]
    cdef int n_edges = repeated_edges.shape[0]
    cdef int n_triangles = triangles.shape[0]

    cdef FLOAT_DTYPE_t[:, :] boundary_quadrics = np.zeros((n_points, 11), dtype=FLOAT_DTYPE)
    # cdef FLOAT_DTYPE_t[:, :] boundary_quadrics_view = boundary_quadrics

    cdef bint boundary = 1
    cdef int e0, e1

    cdef int i, j, k, l
    cdef FLOAT_DTYPE_t c
    cdef INT_DTYPE_t[:] t = np.zeros([3], dtype=INT_DTYPE)
    cdef FLOAT_DTYPE_t[:] t0 = np.zeros([3], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:] t1 = np.zeros([3], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:] t2 = np.zeros([3], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:] u = np.zeros([3], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:] v = np.zeros([3], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:] n = np.zeros([3], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:] Q = np.zeros([11], dtype=FLOAT_DTYPE)

    e0 = repeated_edges[0, 0]
    e1 = repeated_edges[0, 1]

    for i in range(1, n_edges):
        if repeated_edges[i, 0] == e0 and repeated_edges[i, 1] == e1:
            boundary = 0

        else:
            if boundary == 1:
                for j in range(n_triangles):
                    t[0] = triangles[j, 0]
                    t[1] = triangles[j, 1]
                    t[2] = triangles[j, 2]

                    if (
                        (t[0] == e0 and t[1] == e1)
                        or (t[1] == e0 and t[2] == e1)
                        or (t[0] == e0 and t[2] == e1)
                        or (t[0] == e1 and t[1] == e0)
                        or (t[1] == e1 and t[2] == e0)
                        or (t[0] == e1 and t[2] == e0)
                    ):

                    #########
                        for k in range(3):
                            l = t[k]
                            if l != e0 and l != e1:
                                t0[0] = points[l][0]
                                t0[1] = points[l][1]
                                t0[2] = points[l][2]

                        t1[0] = points[e0][0]
                        t1[1] = points[e0][1]
                        t1[2] = points[e0][2]
                        t2[0] = points[e1][0]
                        t2[1] = points[e1][1]
                        t2[2] = points[e1][2]

                        #u = t2 - t1
                        u[0] = t2[0] - t1[0]
                        u[1] = t2[1] - t1[1]
                        u[2] = t2[2] - t1[2]

                        #v = t1 - t0
                        v[0] = t1[0] - t0[0]
                        v[1] = t1[1] - t0[1]
                        v[2] = t1[2] - t0[2]


                        c = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
                        n[0] = v[0] - c * u[0]
                        n[1] = v[1] - c * u[1]
                        n[2] = v[2] - c * u[2]

                        c = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
                        n[0] = n[0] / c
                        n[1] = n[1] / c
                        n[2] = n[2] / c

                        w = u[0] * u[0] + u[1] * u[1] + u[2] * u[2]

                        d = - (n[0] * t1[0] + n[1] * t1[1] + n[2] * t1[2])
                        Q[0] = n[0] * n[0] * w
                        Q[1] = n[0] * n[1] * w
                        Q[2] = n[0] * n[2] * w
                        Q[3] = n[0] * d * w
                        Q[4] = n[1] * n[1] * w
                        Q[5] = n[1] * n[2] * w
                        Q[6] = n[1] * d * w
                        Q[7] = n[2] * n[2] * w
                        Q[8] = n[2] * d * w
                        Q[9] = d * d * w
                        Q[10] = 1 * w

                        for l in range(11):
                            boundary_quadrics[e0, l] += Q[l]
                            boundary_quadrics[e1, l] += Q[l]

            e0 = repeated_edges[i, 0]
            e1 = repeated_edges[i, 1]
            boundary = 1

    return np.asarray(boundary_quadrics)

cdef FLOAT_DTYPE_t[:] pt0 = np.zeros([3], dtype=FLOAT_DTYPE)
cdef FLOAT_DTYPE_t[:] pt1 = np.zeros([3], dtype=FLOAT_DTYPE)
cdef FLOAT_DTYPE_t[:] tmp = np.zeros([3], dtype=FLOAT_DTYPE)
cdef FLOAT_DTYPE_t[:] tmp2 = np.zeros([3], dtype=FLOAT_DTYPE)
cdef FLOAT_DTYPE_t[:] v = np.zeros([3], dtype=FLOAT_DTYPE)
cdef FLOAT_DTYPE_t[:] tmpQuad = np.zeros([11], dtype=FLOAT_DTYPE)

@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_cost(INT_DTYPE_t[:] edge, FLOAT_DTYPE_t[:, :] quadrics, FLOAT_DTYPE_t[:, :] points):
    
    cdef double error = 0.0000000001
    cdef double norm
    cdef double c
    cdef double coast
    cdef double tmp_float
    cdef int e0, e1

    cdef cnp.ndarray[FLOAT_DTYPE_t, ndim=1] x = np.zeros([3], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:] xview = x
    cdef FLOAT_DTYPE_t[:] newpoint = np.zeros([4], dtype=FLOAT_DTYPE)

    cdef int i, j, counter
    
    e0 = edge[0]
    e1 = edge[1]

    for i in range(11):
        tmpQuad[i] = quadrics[e0, i] + quadrics[e1, i]


    #compute the max manually
    norm = sqrt(tmpQuad[0] ** 2 + tmpQuad[1] ** 2 + tmpQuad[2] ** 2)
    tmp_float = sqrt(tmpQuad[1] ** 2 + tmpQuad[4] ** 2 + tmpQuad[5] ** 2)
    if tmp_float > norm:
        norm = tmp_float
    tmp_float = sqrt(tmpQuad[2] ** 2 + tmpQuad[5] ** 2 + tmpQuad[7] ** 2)
    if tmp_float > norm:
        norm = tmp_float


    det = (
        tmpQuad[0] * (tmpQuad[4] * tmpQuad[7] - tmpQuad[5] * tmpQuad[5])
        - tmpQuad[1] * (tmpQuad[1] * tmpQuad[7] - tmpQuad[5] * tmpQuad[2])
        + tmpQuad[2] * (tmpQuad[1] * tmpQuad[5] - tmpQuad[4] * tmpQuad[2])
    )


    if det / (norm ** 3) > error:
        # x = solve3x3(A, b)

        d1 = (
            - tmpQuad[3] * (tmpQuad[4] * tmpQuad[7] - tmpQuad[5] * tmpQuad[5])
            - tmpQuad[1] * (-1.0 * tmpQuad[6] * tmpQuad[7] + tmpQuad[5] * tmpQuad[8])
            + tmpQuad[2] * (-1.0 * tmpQuad[6] * tmpQuad[5] + tmpQuad[4] * tmpQuad[8])
            )

        d2 = (
            tmpQuad[0] * (-1.0 * tmpQuad[6] * tmpQuad[7] + tmpQuad[8] * tmpQuad[5])
            + tmpQuad[3] * (tmpQuad[1] * tmpQuad[7] - tmpQuad[5] * tmpQuad[2])
            + tmpQuad[2] * (-1.0 * tmpQuad[1] * tmpQuad[8] + tmpQuad[6] * tmpQuad[2])
            )

        d3 = (
            tmpQuad[0] * (-1.0 * tmpQuad[4] * tmpQuad[8] + tmpQuad[5] * tmpQuad[6])
            - tmpQuad[1] * (-1.0 * tmpQuad[1] * tmpQuad[8] + tmpQuad[6] * tmpQuad[2])
            + -tmpQuad[3] * (tmpQuad[1] * tmpQuad[5] - tmpQuad[4] * tmpQuad[2])
            )

        x[0] = d1 / det
        x[1] = d2 / det
        x[2] = d3 / det

    else:

        
        for i in range(3):
            pt0[i] = points[e0, i]
            pt1[i] = points[e1, i]
            v[i] = pt1[i] - pt0[i]


        tmp2[0] = tmpQuad[0] * v[0] + tmpQuad[1] * v[1] + tmpQuad[2] * v[2]
        tmp2[1] = tmpQuad[1] * v[0] + tmpQuad[4] * v[1] + tmpQuad[5] * v[2]
        tmp2[2] = tmpQuad[2] * v[0] + tmpQuad[5] * v[1] + tmpQuad[7] * v[2]

        if (tmp2[0] ** 2 + tmp2[1] ** 2 + tmp2[2] ** 2) > error:
            tmp[0] = tmpQuad[0] * pt0[0] + tmpQuad[1] * pt0[1] + tmpQuad[2] * pt0[2]
            tmp[1] = tmpQuad[1] * pt0[0] + tmpQuad[4] * pt0[1] + tmpQuad[5] * pt0[2]
            tmp[2] = tmpQuad[2] * pt0[0] + tmpQuad[5] * pt0[1] + tmpQuad[7] * pt0[2]

            tmp[0] = - tmpQuad[3] - tmp[0]
            tmp[1] = - tmpQuad[6] - tmp[1]
            tmp[2] = - tmpQuad[8] - tmp[2]

            c = (tmp[0] * tmp2[0] + tmp[1] * tmp2[1] + tmp[2] * tmp2[2]) / (tmp2[0] ** 2 + tmp2[1] ** 2 + tmp2[2] ** 2)

            for i in range(3):
                xview[i] = pt0[i] + c * v[i]

        else:
            for i in range(3):
                xview[i] = 0.5 * (pt0[i] + pt1[i])

    cost = 0.0
    for i in range(3):
        newpoint[i] = x[i]
    newpoint[3] = 1.0
    
    counter = 0
    for i in range(4):
        cost += newpoint[i] * newpoint[i] * tmpQuad[counter]
        counter += 1
        for j in range(i + 1, 4):
            cost += 2 * newpoint[i] * newpoint[j] * tmpQuad[counter]
            counter += 1

    return cost, x


@cython.boundscheck(False)
@cython.wraparound(False)
def _initialize_costs(INT_DTYPE_t[:, :] edges, FLOAT_DTYPE_t[:, :]  quadrics, FLOAT_DTYPE_t[:, :] points):
    """Compute the cost of collapsing each edge in the mesh.

    Args:
        edges (Mx2 int64 array): Array of edges to collapse.
        quadrics (Nx11 float64 array): Array of quadrics for each vertex.
        points (Nx3 array): Array of points for each vertex.
    
    Returns:
        costs (M float64 array): Array of costs for each edge.
        newpoints (Mx3 float64 array): Array of new points for each edge.
    """

    cdef int n_edges = edges.shape[0]
    cdef FLOAT_DTYPE_t[:] costs = np.zeros([n_edges], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:, :] newpoints = np.zeros([n_edges, 3], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t cost = 0.0
    cdef FLOAT_DTYPE_t[:] newpoint = np.zeros([3], dtype=FLOAT_DTYPE)

    cdef int i

    for i in range(n_edges):
        costs[i], newpoint = _compute_cost(edges[i, :], quadrics, points)
        # costs[i] = cost
        newpoints[i, :] = newpoint

    return np.asarray(costs), np.asarray(newpoints)


@cython.boundscheck(False)
@cython.wraparound(False)
def _collapse(
    INT_DTYPE_t[:, :] edges,
    FLOAT_DTYPE_t[:] costs,
    FLOAT_DTYPE_t[:, :] newpoints,
    FLOAT_DTYPE_t[:, :] quadrics,
    FLOAT_DTYPE_t[:, :] points,
    INT_DTYPE_t n_points_to_remove=5000):

    cdef INT_DTYPE_t[:] edge = np.zeros([2], dtype=INT_DTYPE)

    cdef INT_DTYPE_t[:] indices_to_remove = np.zeros([n_points_to_remove], dtype=INT_DTYPE)
    cdef INT_DTYPE_t[:, :] collapses = np.zeros([n_points_to_remove, 2], dtype=INT_DTYPE)
    cdef FLOAT_DTYPE_t[:, :] newpoints_history = np.zeros([n_points_to_remove, 3], dtype=FLOAT_DTYPE)
    cdef INT_DTYPE_t n_points_removed = 0
    cdef int n_points = points.shape[0]
    cdef FLOAT_DTYPE_t[:, :] new_vertices = np.zeros([points.shape[0] - n_points_to_remove, 3], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:] newpoint = np.zeros([3], dtype=FLOAT_DTYPE)

    cdef int e0, e1
    cdef int i, j, k, indice, counter

    # the edges with infinite cost will be at the end of the array
    cdef INT_DTYPE_t noninf_limit = edges.shape[0]

    while n_points_removed < n_points_to_remove:

        ## FAST #####
        indice = 0
        for i in range(noninf_limit):
            if costs[i] < costs[indice]:
                indice = i

        ##############
        e0 = edges[indice, 0]
        e1 = edges[indice, 1]
        ### FAST #####

        for k in range(11):
            quadrics[e0, k] += quadrics[e1, k]

        
        points[e0, :] = newpoints[indice, :]
        collapses[n_points_removed, 0] = e0
        collapses[n_points_removed, 1] = e1
        newpoints_history[n_points_removed, :] = points[e0, :]

        indices_to_remove[n_points_removed] = e1
        ##############

        n_points_removed = n_points_removed + 1

        # Update the edges
        i = 0
        # the edges with indices > noninf_limit are the ones with infinite cost
        # they are not considered
        while i < noninf_limit:

            # Update the connectivity e0 <- e1
            if edges[i, 0] == e1:
                edges[i, 0] = e0
            if edges[i, 1] == e1:
                edges[i, 1] = e0

            # Update the cost of the impacted edges (they have e0 as vertex)
            if (edges[i, 0] == e0 or edges[i, 1] == e0) and edges[i, 0] != edges[i, 1]:
                
                costs[i], newpoint = _compute_cost(edge=edges[i, :], quadrics=quadrics, points=points)
                newpoints[i, :] = newpoint


            # If the edge is degenerated, remove it
            if edges[i, 0] == edges[i, 1]:
                noninf_limit -= 1
                costs[i] = costs[noninf_limit]
                edges[i, :] = edges[noninf_limit, :]
                newpoints[i, :] = newpoints[noninf_limit, :]
                i -= 1

            i += 1
    

    np.asarray(indices_to_remove).sort()
    j = 0
    counter = 0
    for i in range(n_points):
        if i == indices_to_remove[j]:
            j += 1
        else:
            new_vertices[counter, 0] = points[i, 0]
            new_vertices[counter, 1] = points[i, 1]
            new_vertices[counter, 2] = points[i, 2]
            counter += 1

    return np.asarray(new_vertices), np.asarray(collapses), np.asarray(newpoints_history)


####### Python functions #######

def _compute_edges(triangles, repeated=False):
    repeated_edges = np.concatenate(
        [
            triangles[[0, 1], :],
            triangles[[1, 2], :],
            triangles[[0, 2], :],
        ],
        axis=1,
    )

    repeated_edges.sort(axis=0)

    repeated_edges
    # Remove the duplicates and return
    if not repeated:
        return np.unique(repeated_edges, axis=1)

    else:
        ordering = np.lexsort(repeated_edges)
        return repeated_edges[:, ordering]


def decimate(
    points,
    triangles,
    target_reduction: float = 0.5,
    running_time: bool = False,
):
    """Apply the quadric decimation algorithm to a mesh.

    Args:
        points (_type_): _description_
        triangles (_type_): _description_
        target_reduction (float, optional): _description_. Defaults to 0.5.
        print_compute_time (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    assert target_reduction > 0.0 and target_reduction < 1.0
    quadrics = _initialize_quadrics(points, triangles.T)
    repeated_edges = _compute_edges(triangles, repeated=True)
    boundary_quadrics = _compute_boundary_quadrics(points, repeated_edges.T, triangles.T)
    quadrics += boundary_quadrics
    edges = _compute_edges(triangles)
    costs, target_points = _initialize_costs(edges.T, quadrics, points)
    n_points_to_remove = int(target_reduction * points.shape[0])
    output_points, collapses, newpoints = _collapse(
        edges=edges.T,
        costs=costs,
        newpoints=target_points,
        quadrics=quadrics,
        points=points,
        n_points_to_remove=n_points_to_remove,
    )


    return output_points, collapses, newpoints




def _replay_loop(FLOAT_DTYPE_t[:, :] points, FLOAT_DTYPE_t[:, :] quadrics, INT_DTYPE_t[:, :] collapses_history):

    cdef INT_DTYPE_t n_collapses = collapses_history.shape[0]
    cdef INT_DTYPE_t[:] edge = np.zeros([2], dtype=INT_DTYPE)
    cdef FLOAT_DTYPE_t[:] newpoint = np.zeros([3], dtype=FLOAT_DTYPE)
    cdef INT_DTYPE_t e0, e1
    cdef INT_DTYPE_t i, k
    cdef FLOAT_DTYPE_t cost

    for i in range(n_collapses):
        e0, e1 = collapses_history[i, :]
        for k in range(11):
            quadrics[e0, k] += quadrics[e1, k]
        edge[0] = e0
        edge[1] = e1
        cost, newpoint = _compute_cost(edge, quadrics, points)
        points[e0, :] = newpoint

    return np.asarray(points)


def replay_decimation(
    points,
    triangles,
    collapses_history,
):

    quadrics = _initialize_quadrics(points, triangles)
    repeated_edges = _compute_edges(triangles, repeated=True)
    boundary_quadrics = _compute_boundary_quadrics(points, repeated_edges, triangles)
    quadrics += boundary_quadrics

    newpoints = _replay_loop(points.copy(), quadrics, collapses_history)
    
    keep = np.setdiff1d(
        np.arange(len(points)), np.array(collapses_history[:, 1])
    )  # Indices of the points that must be kept after decimation

    return newpoints[keep]



if False:

    DecimationOutput = namedtuple(
            "DecimationOutput",
            [
                "decimated_points",
                "decimated_triangles",
                "indice_mapping",
                "collapses_history",
            ],
        )

    # Newpoints history is no longer needed with
    # the new implementation of replay

    def decimate2(points, triangle, n_points_to_remove):
        """Decimate a mesh to a given number of points.

        Args:
            points (_type_): _description_
            triangle (_type_): _description_
            n_points_removed (_type_): _description_
        """

        # Initialize the quadrics as the sum of boundary quadrics and non-boundary quadrics
        quadrics = _nonboundary_quadrics(points, triangles) + _boundary_quadrics(points, triangles)
        # Now we have a quadric for each vertex, one can forget about the triangles and focus on
        # the edges to decimate. Each edge will have a cost associated to it, and a target point.
        # After executing the decimation, we will compute the new triangles.

        # Compute the edges of the mesh
        edges = _get_edges(triangles)
        # Initialize the costs and target points
        costs, target_points = _initialize_costs(edges, quadrics, points)
        # Decimate the mesh to the desired number of points
        decimated_points, collapses_history = _collapse(
            edges, costs, target_points, quadrics, points, n_points_to_remove
        )

        # At this point, we did the intensive computations. We have the decimated points, the
        # history of collapses, and the history of new points. We can now compute the new triangles
        # and the mapping between the old and new indices from these informations.

        # Compute the mapping between the old and new indices
        indice_mapping = _compute_indice_mapping(collapses_history)
        # Compute the new triangles
        decimated_triangles = _compute_decimated_triangles(triangles, indice_mapping)

        # Return a named tuple with all the informations
        DecimationOutput = namedtuple(
            "DecimationOutput",
            [
                "decimated_points",
                "decimated_triangles",
                "indice_mapping",
                "collapses_history",
            ],
        )

        return DecimationOutput(
            decimated_points=decimated_points,
            decimated_triangles=decimated_triangles,
            indice_mapping=indice_mapping,
            collapses_history=collapses_history,
            newpoints_history=newpoints_history,
        )


    def replay_decimation2(points, triangles, collapses_history, n_points_to_remove):
        """Replay a decimation from a history of collapses.

        You can replay a decimation
        """
        assert 0 <= n_points_to_remove <= collapses_history.shape[0], "n_points_to_remove must be between 0 and the number of collapses"
        collapses = collapses_history[:n_points_to_remove, :]

        # Initialize the quadrics as the sum of boundary quadrics and non-boundary quadrics
        quadrics = _nonboundary_quadrics(points, triangles) + _boundary_quadrics(points, triangles)

        #Replay the decimation
        decimated_points = _replay_decimation(points, quadrics, collapses)

        # Compute the mapping between the old and new indices
        indice_mapping = _compute_indice_mapping(collapses)

        # Compute the new triangles
        decimated_triangles = _compute_decimated_triangles(triangles, indice_mapping)   

        return DecimationOutput(
            decimated_points=decimated_points,
            decimated_triangles=decimated_triangles,
            indice_mapping=indice_mapping,
            collapses_history=collapses_history,
        )

# distutils: language = c++

from .Rectangle cimport Rectangle
# from libcpp.vector cimport vector

cdef class PyRectangle:
    cdef Rectangle*c_rect  # hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self):
        self.c_rect = new Rectangle()

    def __init__(self, int x0, int y0, int x1, int y1):

        self.c_rect.setCoordinates(x0, y0, x1, y1)

        self.c_rect.x0 = x0
        self.c_rect.y0 = y0
        self.c_rect.x1 = x1
        self.c_rect.y1 = y1

    def getArea(self):
        return self.c_rect.getArea()

    def __dealloc__(self):
        del self.c_rect


cimport cython

from libc.math cimport sqrt
cimport numpy as cnp
cnp.import_array()
import numpy as np
from collections import namedtuple

INT_DTYPE = np.int64
FLOAT_DTYPE = np.double
ctypedef cnp.int64_t INT_DTYPE_t
ctypedef cnp.double_t FLOAT_DTYPE_t


DecimationOutput = namedtuple(
        typename="DecimationOutput",
        field_names=[
            "decimated_points",
            "decimated_triangles",
            "indice_mapping",
            "collapses",
        ]
    )

##################################################################################################

# Utility functions for dim 3 vectors
@cython.boundscheck(False)
@cython.wraparound(False)
def dot3(FLOAT_DTYPE_t[:] x, FLOAT_DTYPE_t[:] y) -> FLOAT_DTYPE_t:
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]

##################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
def squared_norm3(FLOAT_DTYPE_t[:] x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

##################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
def norm3(FLOAT_DTYPE_t[:] x):
    return sqrt(squared_norm3(x))

##################################################################################################

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

##################################################################################################

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def solve3x3(FLOAT_DTYPE_t[:, :] A, FLOAT_DTYPE_t[:] b, FLOAT_DTYPE_t[:] x):

#     d = det3x3(A)

#     x[0] = (
#         b0 * (A[1, 1] * A[2, 2] - A[1, 2] * A[1, 2])
#         - b1 * (A[0, 1] * A[2, 2] - A[0, 2] * A[1, 2])
#         + b2 * (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1])
#     ) / d
    
#     x[1] = (
#         A[0, 0] * (b1 * A[2, 2] - b2 * A[1, 2])
#         - A[0, 1] * (b0 * A[2, 2] - b2 * A[0, 2])
#         + A[0, 2] * (b0 * A[1, 2] - b1 * A[0, 2])
#     ) / d


#     x[3] = (
#         A[0, 0] * (A[1, 1] * b2 - A[1, 2] * b1)
#         - A[0, 1] * (A[0, 1] * b2 - A[1, 2] * b0)
#         + A[0, 2] * (A[0, 1] * b1 - A[1, 1] * b0)
#     ) / d

##################################################################################################

def _compute_edges(triangles, repeated=False):
    repeated_edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [0, 2]],
        ],
        axis=0,
    )

    repeated_edges.sort(axis=1)
    # Remove the duplicates and return
    if not repeated:
        return np.unique(repeated_edges, axis=0)

    else:
        # return repeated_edges
        ordering = np.lexsort(repeated_edges.T)
        return repeated_edges[ordering]

##################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
def _nonboundary_quadrics(FLOAT_DTYPE_t[:, :] points, INT_DTYPE_t[:, :] triangles):

    cdef int n_points = points.shape[0]
    cdef int n_triangles = triangles.shape[0]
    cdef FLOAT_DTYPE_t[:, :] quadrics = np.zeros([n_points, 11], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:] Q = np.zeros([11], dtype=FLOAT_DTYPE)
    cdef int i, j, k
    cdef FLOAT_DTYPE_t d

    cdef FLOAT_DTYPE_t[:] p0 = np.zeros([3], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:] p1 = np.zeros([3], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:] p2 = np.zeros([3], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:] n = np.zeros([3], dtype=FLOAT_DTYPE)


    for i in range(n_triangles):

        # Get the points of the triangle
        # p0[:] = points[triangles[i, 0], :] is slower
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


##################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
def _boundary_quadrics(FLOAT_DTYPE_t[:, :] points, INT_DTYPE_t[:, :] triangles):

    cdef INT_DTYPE_t[:, :] repeated_edges = _compute_edges(triangles=np.asarray(triangles), repeated=True)
    cdef int n_triangles = triangles.shape[0]
    cdef int i, j, k, l

    cdef int n_points = points.shape[0]
    cdef int n_edges = repeated_edges.shape[0]

    cdef FLOAT_DTYPE_t[:, :] boundary_quadrics = np.zeros((n_points, 11), dtype=FLOAT_DTYPE)

    cdef bint boundary = 1
    cdef int e0, e1

    
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


##############################################################

@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_cost(
    INT_DTYPE_t[:] edge,
    FLOAT_DTYPE_t[:, :] quadrics,
    FLOAT_DTYPE_t[:, :] points,
    FLOAT_DTYPE_t[:] pt0 = np.zeros([3], dtype=FLOAT_DTYPE),
    FLOAT_DTYPE_t[:] pt1 = np.zeros([3], dtype=FLOAT_DTYPE),
    FLOAT_DTYPE_t[:] tmp = np.zeros([3], dtype=FLOAT_DTYPE),
    FLOAT_DTYPE_t[:] tmp2 = np.zeros([3], dtype=FLOAT_DTYPE),
    FLOAT_DTYPE_t[:] v = np.zeros([3], dtype=FLOAT_DTYPE),
    FLOAT_DTYPE_t[:] tmpQuad = np.zeros([11], dtype=FLOAT_DTYPE),
    FLOAT_DTYPE_t[:] x = np.zeros([3], dtype=FLOAT_DTYPE),
    FLOAT_DTYPE_t[:] newpoint = np.zeros([4], dtype=FLOAT_DTYPE)):

    cdef double error = 0.0000000001
    cdef double norm
    cdef double c, cost
    cdef double tmp_float
    cdef int e0, e1

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
                x[i] = pt0[i] + c * v[i]

        else:
            for i in range(3):
                x[i] = 0.5 * (pt0[i] + pt1[i])

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

    return cost, np.asarray(x)


##############################################################

@cython.boundscheck(False)
@cython.wraparound(False)
def _initialize_costs(INT_DTYPE_t[:, :] edges, FLOAT_DTYPE_t[:, :]  quadrics, FLOAT_DTYPE_t[:, :] points):

    # Temporary variables for _compute_cost routine
    # there are defined here to avoid re-initialize them a lot of time
    # cdef FLOAT_DTYPE_t[:] _pt0 = np.zeros([3], dtype=FLOAT_DTYPE)
    # cdef FLOAT_DTYPE_t[:] _pt1 = np.zeros([3], dtype=FLOAT_DTYPE)
    # cdef FLOAT_DTYPE_t[:] _tmp = np.zeros([3], dtype=FLOAT_DTYPE)
    # cdef FLOAT_DTYPE_t[:] _tmp2 = np.zeros([3], dtype=FLOAT_DTYPE)
    # cdef FLOAT_DTYPE_t[:] _v = np.zeros([3], dtype=FLOAT_DTYPE)
    # cdef FLOAT_DTYPE_t[:] _tmpQuad = np.zeros([11], dtype=FLOAT_DTYPE)
    # cdef FLOAT_DTYPE_t[:] _x = np.zeros([3], dtype=FLOAT_DTYPE)
    # cdef FLOAT_DTYPE_t[:] _tmpNewpoint = np.zeros([4], dtype=FLOAT_DTYPE)

    cdef int n_edges = edges.shape[0]
    cdef FLOAT_DTYPE_t[:] costs = np.zeros([n_edges], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:, :] newpoints = np.zeros([n_edges, 3], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t cost = 0.0
    cdef FLOAT_DTYPE_t[:] newpoint = np.zeros([3], dtype=FLOAT_DTYPE)

    cdef int i

    for i in range(n_edges):
        costs[i], newpoint = _compute_cost(edge=edges[i, :], quadrics=quadrics, points=points, pt0=_pt0, pt1=_pt1, tmp=_tmp, tmp2=_tmp2, v=_v, tmpQuad=_tmpQuad, x=_x, newpoint=_tmpNewpoint)
        newpoints[i, :] = newpoint

    return np.asarray(costs), np.asarray(newpoints)


######################################################################


# Temporary variables for _compute_cost routine
# there are defined here to avoid re-initialize them a lot of time
cdef FLOAT_DTYPE_t[:] _pt0 = np.zeros([3], dtype=FLOAT_DTYPE)
cdef FLOAT_DTYPE_t[:] _pt1 = np.zeros([3], dtype=FLOAT_DTYPE)
cdef FLOAT_DTYPE_t[:] _tmp = np.zeros([3], dtype=FLOAT_DTYPE)
cdef FLOAT_DTYPE_t[:] _tmp2 = np.zeros([3], dtype=FLOAT_DTYPE)
cdef FLOAT_DTYPE_t[:] _v = np.zeros([3], dtype=FLOAT_DTYPE)
cdef FLOAT_DTYPE_t[:] _tmpQuad = np.zeros([11], dtype=FLOAT_DTYPE)
cdef FLOAT_DTYPE_t[:] _x = np.zeros([3], dtype=FLOAT_DTYPE)
cdef FLOAT_DTYPE_t[:] _tmpNewpoint = np.zeros([4], dtype=FLOAT_DTYPE)
#TODO : why putting these lines in _collapse make the function slower ?

@cython.boundscheck(False)
@cython.wraparound(False)
def _collapse(
    INT_DTYPE_t[:, :] edges,
    FLOAT_DTYPE_t[:] costs,
    FLOAT_DTYPE_t[:, :] target_points,
    FLOAT_DTYPE_t[:, :] quadrics,
    FLOAT_DTYPE_t[:, :] points,
    INT_DTYPE_t n_points_to_remove=5000):

    
    points = points.copy()

    cdef INT_DTYPE_t[:] indices_to_remove = np.zeros([n_points_to_remove], dtype=INT_DTYPE)
    cdef INT_DTYPE_t[:, :] collapses = np.zeros([n_points_to_remove, 2], dtype=INT_DTYPE)
    cdef INT_DTYPE_t n_points_removed = 0
    cdef int n_points = points.shape[0]

    cdef FLOAT_DTYPE_t[:, :] decimated_points = np.zeros([points.shape[0] - n_points_to_remove, 3], dtype=FLOAT_DTYPE)
    cdef FLOAT_DTYPE_t[:] newpoint = np.zeros([3], dtype=FLOAT_DTYPE)

    cdef int e0, e1
    cdef int i, j, k, indice, counter

    # the edges with infinite cost will be at the end of the array
    cdef INT_DTYPE_t noninf_limit = edges.shape[0]

    while n_points_removed < n_points_to_remove:

        indice = 0
        for i in range(noninf_limit):
            if costs[i] < costs[indice]:
                indice = i

        e0 = edges[indice, 0]
        e1 = edges[indice, 1]

        for k in range(11):
            quadrics[e0, k] += quadrics[e1, k]

        for k in range(3):
            points[e0, k] = target_points[indice, k]
        collapses[n_points_removed, 0] = e0
        collapses[n_points_removed, 1] = e1
        # newpoints_history[n_points_removed, :] = points[e0, :]

        indices_to_remove[n_points_removed] = e1
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
                
                costs[i], newpoint = _compute_cost(edge=edges[i, :], quadrics=quadrics, points=points, pt0=_pt0, pt1=_pt1, tmp=_tmp, tmp2=_tmp2, v=_v, tmpQuad=_tmpQuad, x=_x, newpoint=_tmpNewpoint)
                target_points[i, 0] = newpoint[0]
                target_points[i, 1] = newpoint[1]
                target_points[i, 2] = newpoint[2]


            # If the edge is degenerated, remove it
            if edges[i, 0] == edges[i, 1]:
                noninf_limit -= 1
                costs[i] = costs[noninf_limit]
                edges[i, 0] = edges[noninf_limit, 0]
                edges[i, 1] = edges[noninf_limit, 1]
                target_points[i, 0] = target_points[noninf_limit, 0]
                target_points[i, 1] = target_points[noninf_limit, 1]
                target_points[i, 2] = target_points[noninf_limit, 2]
                i -= 1

            i += 1
    

    np.asarray(indices_to_remove).sort()
    j = 0
    counter = 0
    for i in range(n_points):
        if i == indices_to_remove[j]:
            j += 1
        else:
            decimated_points[counter, 0] = points[i, 0]
            decimated_points[counter, 1] = points[i, 1]
            decimated_points[counter, 2] = points[i, 2]
            counter += 1

    return np.asarray(decimated_points), np.asarray(collapses)

def _compute_indice_mapping(collapses, n_points):

    # Compute the mapping from original indices to new indices
    keep = np.setdiff1d(
        np.arange(n_points), collapses[:, 1]
    )  # Indices of the points that must be kept after decimation
    # start with identity mapping
    indice_mapping = np.arange(n_points, dtype=INT_DTYPE)
    # First round of mapping
    origin_indices = collapses[:, 1]
    indice_mapping[origin_indices] = collapses[:, 0]
    previous = np.zeros(len(indice_mapping))
    while not np.array_equal(previous, indice_mapping):
        previous = indice_mapping.copy()
        indice_mapping[origin_indices] = indice_mapping[
            indice_mapping[origin_indices]
        ]
    application = dict([keep[i], i] for i in range(len(keep)))
    indice_mapping = np.array([application[i] for i in indice_mapping])

    return indice_mapping

def _compute_decimated_triangles(triangles, indice_mapping):

    triangles = indice_mapping[triangles.copy()]
    # compute the new triangles
    keep_triangle = (
        (triangles[:, 0] != triangles[:, 1])
        * (triangles[:, 1] != triangles[:, 2])
        * (triangles[:, 0] != triangles[:, 2])
    )
    return triangles[keep_triangle]



def decimate(points, triangles, n_points_to_remove):
    """Decimate a mesh to a given number of points.

    Args:
        points (_type_): _description_
        triangle (_type_): _description_
        n_points_removed (_type_): _description_
    """

    points = np.array(points.copy(), dtype=FLOAT_DTYPE)
    n_points = points.shape[0]
    triangles = np.array(triangles.copy(), dtype=INT_DTYPE)

    # Initialize the quadrics as the sum of boundary quadrics and non-boundary quadrics
    quadrics = (
        _nonboundary_quadrics(points=points, triangles=triangles)
        + _boundary_quadrics(points=points, triangles=triangles)
        )
    # Now we have a quadric for each vertex, one can forget about the triangles and focus on
    # the edges to decimate. Each edge will have a cost associated to it, and a target point.
    # After executing the decimation, we will compute the new triangles.

    # Compute the edges of the mesh
    edges = _compute_edges(triangles)
    # Initialize the costs and target points
    costs, target_points = _initialize_costs(points=points, edges=edges, quadrics=quadrics)
    # Decimate the mesh to the desired number of points
    decimated_points, collapses = _collapse(
        points=points,
        edges=edges,
        costs=costs,
        target_points=target_points,
        quadrics=quadrics,
        n_points_to_remove=n_points_to_remove
    )

    # At this point, we did the intensive computations. We have the decimated points, the
    # history of collapses, and the history of new points. We can now compute the new triangles
    # and the mapping between the old and new indices from these informations.

    # Compute the mapping between the old and new indices
    indice_mapping = _compute_indice_mapping(collapses=collapses, n_points=n_points)
    # Compute the new triangles
    decimated_triangles = _compute_decimated_triangles(triangles=triangles, indice_mapping=indice_mapping)

    # Return a named tuple with all the informations

    return DecimationOutput(
        decimated_points=decimated_points,
        decimated_triangles=decimated_triangles,
        indice_mapping=indice_mapping,
        collapses=collapses,
    )


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
        cost, newpoint = _compute_cost(edge=edge, quadrics=quadrics, points=points, pt0=_pt0, pt1=_pt1, tmp=_tmp, tmp2=_tmp2, v=_v, tmpQuad=_tmpQuad, x=_x, newpoint=_tmpNewpoint)
        points[e0, :] = newpoint

    return np.asarray(points)


def _replay_decimation(
    points,
    triangles,
    collapses,
):

    points = np.array(points.copy(), dtype=FLOAT_DTYPE) 
    triangles = np.array(triangles.copy(), dtype=INT_DTYPE)

    quadrics = (
        _nonboundary_quadrics(points=points, triangles=triangles)
        + _boundary_quadrics(points=points, triangles=triangles)
        )

    decimated_points = _replay_loop(points, quadrics, collapses)
    
    keep = np.sort(np.setdiff1d(
        np.arange(len(points)), np.array(collapses[:, 1])
    ))  # Indices of the points that must be kept after decimation

    return decimated_points[keep]

def replay_decimation(points, triangles, collapses, stop_after=None):
        """Replay a decimation from a history of collapses.

        You can replay a decimation
        """
        points = np.array(points.copy(), dtype=FLOAT_DTYPE)
        n_points = points.shape[0]
        triangles = np.array(triangles.copy(), dtype=INT_DTYPE)

        if stop_after is not None:
            assert 0 <= stop_after <= collapses.shape[0], "stop_after must be between 0 and the number of collapses"
        else:
            stop_after = collapses.shape[0]

        collapses = collapses[0:stop_after, :]
        # Initialize the quadrics as the sum of boundary quadrics and non-boundary quadrics

        #Replay the decimation
        decimated_points = _replay_decimation(points=points, triangles=triangles, collapses=collapses)
        # Compute the mapping between the old and new indices
        indice_mapping = _compute_indice_mapping(collapses=collapses, n_points=n_points)
        # Compute the new triangles
        decimated_triangles = _compute_decimated_triangles(triangles=triangles, indice_mapping=indice_mapping)   

        return DecimationOutput(
            decimated_points=decimated_points,
            decimated_triangles=decimated_triangles,
            indice_mapping=indice_mapping,
            collapses=collapses,
        )
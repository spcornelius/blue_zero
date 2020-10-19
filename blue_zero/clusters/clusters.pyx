# cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
ctypedef np.uint8_t uint8
from libc.stdlib cimport calloc, free

__all__ = ['find_clusters']

cdef inline int int_max(int a, int b): return a if a >= b else b

cdef int find(int x, int *alias):
    cdef int z, y = x
    while alias[y] != y:
        y = alias[y]

    while alias[x] != x:
        z = alias[x]
        alias[x] = y
        x = z
    return y


cdef int unite(int x, int y, int *alias):
    cdef int a = find(y, alias)
    alias[find(x, alias)] = a
    return a


def find_clusters(uint8[:, ::1] grid):
    """ Hoshen-Kopelman algorithm for identifying clusters in 2D square
        lattice. """
    cdef Py_ssize_t h = grid.shape[0]
    cdef Py_ssize_t w = grid.shape[1]

    # cdef int [::1] alias = np.arange(h * w + 1, dtype=np.intc)
    cdef int* alias = <int *> calloc(sizeof(int), h * w + 1)
    alias[0] = 0

    # for x in range(h * w + 1):
    #    alias[x] = x

    labels_arr = np.zeros((h, w), dtype=np.intc)
    cdef int[:, ::1] labels = labels_arr

    cdef Py_ssize_t i, j
    cdef int left, up, x

    for i in range(h):
        for j in range(w):
            if grid[i, j]:
                left = labels[i - 1, j] if i > 0 else 0
                up = labels[i, j - 1] if j > 0 else 0

                x = (left != 0) + (up != 0)
                if x == 2:
                    labels[i, j] = unite(left, up, alias)
                elif x == 1:
                    labels[i, j] = max(up, left)
                else:
                    alias[0] += 1
                    alias[alias[0]] = alias[0]
                    labels[i, j] = alias[0]


    # relabel each cluster using only the smallest equivalent label alias
    cdef int* new_labels = <int *> calloc(sizeof(int), h * w + 1)
    new_labels[0] = 0

    for i in range(h):
        for j in range(w):
            if labels[i, j]:
                x = find(labels[i, j], alias)
                if new_labels[x] == 0:
                    new_labels[0] += 1
                    new_labels[x] = new_labels[0]
                labels[i, j] = new_labels[x]

    # calculate sizes of all clusters
    cluster_sizes_arr = np.zeros(new_labels[0]+1, dtype=np.intc)
    cdef int[::1] cluster_sizes = cluster_sizes_arr

    for i in range(h):
        for j in range(w):
            if labels[i, j]:
                cluster_sizes[labels[i, j]] += 1

    free(alias)
    free(new_labels)

    return labels_arr, cluster_sizes_arr

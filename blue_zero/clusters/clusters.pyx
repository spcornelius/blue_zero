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

    label_arr = np.zeros((h, w), dtype=np.intc)
    cdef int[:, ::1] label = label_arr

    cdef Py_ssize_t i, j
    cdef int left, up, x

    for i in range(h):
        for j in range(w):
            if grid[i, j]:
                left = label[i - 1, j] if i > 0 else 0
                up = label[i, j - 1] if j > 0 else 0

                x = (left != 0) + (up != 0)
                if x == 2:
                    label[i, j] = unite(left, up, alias)
                elif x == 1:
                    label[i, j] = max(up, left)
                else:
                    alias[0] += 1
                    alias[alias[0]] = alias[0]
                    label[i, j] = alias[0]


    # relabel each cluster using only the smallest equivalent label alias
    cdef int* new_label = <int *> calloc(sizeof(int), h * w + 1)
    new_label[0] = 0

    for i in range(h):
        for j in range(w):
            if label[i, j]:
                x = find(label[i, j], alias)
                if new_label[x] == 0:
                    new_label[0] += 1
                    new_label[x] = new_label[0]
                label[i, j] = new_label[x]

    free(alias)
    free(new_label)

    return label_arr

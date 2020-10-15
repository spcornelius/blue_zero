cimport cython
import numpy as np
cimport numpy as np
ctypedef np.uint8_t uint8


__all__ = ['find_clusters']


cdef int find(int x, int[::1] alias):
    cdef int z, y = x
    while alias[y] != y:
        y = alias[y]

    while alias[x] != x:
        z = alias[x]
        alias[x] = y
        x = z
    return y


cdef int unite(int x, int y, int[::1] alias):
    cdef int a = find(y, alias)
    alias[find(x, alias)] = a
    return a


@cython.boundscheck(False)
@cython.wraparound(False)
def find_clusters(uint8[:, ::1] grid):
    """ Hoshen-Kopelman algorithm for identifying clusters in 2D square
        lattice. """
    cdef Py_ssize_t h = grid.shape[0]
    cdef Py_ssize_t w = grid.shape[1]

    alias = np.arange(h * w + 1, dtype=np.intc)
    label = np.zeros((h, w), dtype=np.intc)

    cdef int[:, ::1] label_view = label
    cdef int[::1] alias_view = alias

    cdef Py_ssize_t i, j
    cdef int left, up, x

    for i in range(h):
        for j in range(w):
            if grid[i, j]:
                left = label_view[i - 1, j] if i > 0 else 0
                up = label_view[i, j - 1] if j > 0 else 0

                if left == 0 and up == 0:
                    alias_view[0] += 1
                    label_view[i, j] = alias_view[0]
                elif left > 0 and up == 0:
                    label_view[i, j] = find(left, alias_view)
                elif left == 0 and up > 0:
                    label_view[i, j] = find(up, alias_view)
                else:
                    unite(left, up, alias_view)
                    label_view[i, j] = find(left, alias_view)

    # relabel each cluster using only the smallest equivalent label alias
    new_label = np.zeros(h * w + 1, dtype=np.intc)
    cdef int[::1] new_label_view = new_label

    for i in range(h):
        for j in range(w):
            if label_view[i, j]:
                x = find(label_view[i, j], alias_view)
                if new_label_view[x] == 0:
                    new_label_view[0] += 1
                    new_label_view[x] = new_label_view[0]
                label_view[i, j] = new_label_view[x]

    return label

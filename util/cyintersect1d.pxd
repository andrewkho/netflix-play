#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

cdef inline int min(int left, int right) nogil:
    if left < right:
        return left
    else:
        return right

cdef inline int cyintersect1d(int* ixn, int[:] left, int[:] right) nogil:
    """
    To use this function, left and right must be unique and in sorted order
    Returns the intersection of the two arrays in ixn
    """
    cdef int i = 0
    cdef int j = 0
    cdef int c = 0
    cdef int max_left, max_right
    max_left = left.shape[0]
    max_right = right.shape[0]

    while True:
        if left[i] < right[j]:
            i += 1
        elif left[i] > right[j]:
            j += 1
        else:
            ixn[c] = left[i]
            c += 1
            i += 1
            j += 1
        if i >= max_left or j >= max_right:
            break

    return c


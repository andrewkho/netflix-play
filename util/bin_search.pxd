#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False


cpdef inline int bin_search(int st, int en, int[:] arr, long val) nogil:
    if st >= en:
        return st

    cdef int i = (st + en) / 2

    if val < arr[i]:
        return bin_search(st, i, arr, val)
    elif val > arr[i]:
        return bin_search(i+1, en, arr, val)
    else:
        return i

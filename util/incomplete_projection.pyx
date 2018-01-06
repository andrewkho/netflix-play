#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

import numpy as np
cimport numpy as np

import scipy.sparse
from scipy.sparse import csr_matrix

def incomplete_projection(left, right):
    # type: (scipy.sparse.csr_matrix, np.ndarray) -> np.ndarray
    """
    Performs a projection of incomplete matrix "left" onto the space
    defined by the dense matrix "right".

    :param left: A sparse representation of incomplete matrix, i.e. ratings in csr_matrix format
    :param right: Dense representation of the transformation, i.e. U matrix of SVD with first k-columns
    :return: A dense matrix where the rows are the transformed rows of left into the space defined by right, in coo_format
    """
    assert left.shape[1] == right.shape[0], "Incorrect dimensions! left (%d, %d), right (%d, %d)" %\
                                            (left.shape[0], left.shape[1], right.shape[0], right.shape[1])
    return _incomplete_projection(left, right)


cdef _incomplete_projection(left, right):
    # type: (scipy.sparse.csr_matrix, np.ndarray) -> np.ndarray

    cdef int[:] csr_indptr, csr_indices
    cdef double[:] csr_data
    csr_indptr = left.indptr
    csr_indices = left.indices
    csr_data = left.data

    cdef np.ndarray[np.float64_t, ndim=2] out_mat
    out_mat = np.zeros((left.shape[0], right.shape[1]), dtype=np.float64)

    _inner_loop(out_mat, csr_indices, csr_indptr, csr_data, right,
               left.shape[0], left.shape[1], right.shape[0], right.shape[1])

    return out_mat


cdef int _inner_loop(double[:,:] out_mat, int[:] csr_indices, int[:] csr_indptr, double[:] csr_data, double[:,:] right,
                     int left_num_rows, int left_num_cols, int right_num_rows, int right_num_cols):

    cdef double tot
    cdef int out_row, out_col, k, i
    for out_row in range(left_num_rows):
        user_cols = csr_indices[csr_indptr[out_row]:csr_indptr[out_row + 1]]
        user_rats = csr_data[csr_indptr[out_row]:csr_indptr[out_row + 1]]

        for out_col in range(right_num_cols):
            tot = 0.
            for i in range(user_cols.size):
                k = user_cols[i]
                rating = user_rats[i]
                tot += right[k, out_col] * rating
            tot /= user_cols.size
            out_mat[out_row, out_col] = tot

    return 0

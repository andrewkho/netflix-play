#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

import numpy as np
cimport numpy as np

import scipy.sparse
from scipy.sparse import coo_matrix

from util.cyintersect1d cimport cyintersect1d, min

def mle_cov(ratings_mat):
    # type: (scipy.sparse.coo_matrix) -> scipy.sparse.coo_matrix
    """
    Calculate the "maximum likelihood approximate" item (column)-wise covariance
    matrix of Ratings by only including rated items in the covariance calculation.
    If two items have nothing in common, then returns zero.

    If ratings_mat is of shape (n,m), returns a matrix of shape (m,m)

    :param ratings: A sparse matrix in coo format of ratings_mat
    :return: a sparse matrix in scipy.sparse.coo_matrix format
    """
    return py_get_cov(ratings_mat)


cdef py_get_cov(ratings):
    # type: (scipy.sparse.coo_matrix) -> scipy.sparse.coo_matrix

    cdef int[:] csr_indptr, csr_indices, csc_indptr, csc_indices
    cdef double[:] csr_data, csc_data
    csr_mat = ratings.tocsr()
    if csr_mat.has_sorted_indices == False:
        csr_mat.sort_indices()
    csr_indptr = csr_mat.indptr
    csr_indices = csr_mat.indices
    csr_data = csr_mat.data
    csc_mat = ratings.tocsc()
    if csc_mat.has_sorted_indices == False:
        csc_mat.sort_indices()
    csc_indptr = csc_mat.indptr
    csc_indices = csc_mat.indices
    csc_data = csc_mat.data

    cdef double[:,:] out_mat
    out_mat = np.zeros(shape=(csr_mat.shape[1], csr_mat.shape[1]), dtype=np.float64)

    cdef int out_counter
    cdef int item_col
    cdef int[:] item_rows
    cdef double[:] item_rats, column_means
    cdef double mn

    column_means = np.zeros(ratings.shape[1])
    print("  cy: Building column_means")
    for item_col in range(ratings.shape[1]):
        item_rows = csc_indices[csc_indptr[item_col]:csc_indptr[item_col + 1]]
        item_rats = csc_data[csc_indptr[item_col]:csc_indptr[item_col + 1]]
        mn = 0.
        for rat in item_rats:
            mn += rat
        column_means[item_col] = mn / item_rats.shape[0]
    print "  cy: building covar matrix with size (%d, %d)" % (ratings.shape[1], ratings.shape[1])
    out_counter = _inner_loop(out_mat,
                              csr_indices, csr_indptr, csr_data,
                              csc_indices, csc_indptr, csc_data,
                              ratings.shape[1], column_means)

    print "  cy: done"
    return out_mat


cdef int _inner_loop(double[:,:] out_mat,
                     int[:] csr_indices, int[:] csr_indptr, double[:] csr_data,
                     int[:] csc_indices, int[:] csc_indptr, double[:] csc_data,
                     int num_cols, double[:] item_means) nogil:

    cdef int out_counter = 0
    cdef int[:] other_rows, item_rows
    cdef double[:] other_rats, item_rats
    cdef double cov, item_mean, other_mean,

    cdef int other_col, item_row, item_col, i, j
    for item_col in range(num_cols):
        item_rows = csc_indices[csc_indptr[item_col]:csc_indptr[item_col + 1]]
        item_rats = csc_data[csc_indptr[item_col]:csc_indptr[item_col + 1]]

        for other_col in range(num_cols):
            other_rows = csc_indices[csc_indptr[other_col]:csc_indptr[other_col + 1]]
            other_rats = csc_data[csc_indptr[other_col]:csc_indptr[other_col + 1]]
            item_mean = item_means[item_col]
            other_mean = item_means[other_col]

            cov = _cycovar(item_rows, other_rows,
                           item_mean, other_mean,
                           item_rats, other_rats)

            out_mat[item_col, other_col] = cov
            out_counter += 1

    return out_counter


cdef inline double _cycovar(int[:] left, int[:] right,
                         double item_mean, double other_mean,
                         double[:] item_rats, double[:] other_rats) nogil:
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

    cdef double covar = 0

    while True:
        if left[i] < right[j]:
            i += 1
        elif left[i] > right[j]:
            j += 1
        else:
            covar += (item_rats[i] - item_mean) * (other_rats[j] - other_mean)
            c += 1
            i += 1
            j += 1
        if i >= max_left or j >= max_right:
            break

    if c == 0:
        return 0.
    else:
        return covar / c

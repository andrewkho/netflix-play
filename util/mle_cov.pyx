#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

from libcpp.map cimport map as cppmap
from libcpp.set cimport set as cppset
cdef extern from "math.h":
    double sqrt(double m)

import numpy as np
cimport numpy as np

import scipy.sparse
from scipy.sparse import coo_matrix


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
    csr_indptr = csr_mat.indptr
    csr_indices = csr_mat.indices
    csr_data = csr_mat.data
    csc_mat = ratings.tocsc()
    csc_indptr = csc_mat.indptr
    csc_indices = csc_mat.indices
    csc_data = csc_mat.data

    cdef long output_length
    output_length = ((csr_mat > 0).transpose().dot((csr_mat > 0)) > 0).sum()
    print("predicted output_length: " + str(output_length))

    cdef int[:] out_row, out_col
    cdef double[:] out_data
    cdef long out_counter
    out_row = np.zeros(output_length, dtype=np.int32)
    out_col = np.zeros(output_length, dtype=np.int32)
    out_data = np.zeros(output_length, dtype=np.float64)
    out_counter = 0

    cdef int item_col, i
    cdef int[:] item_rows
    cdef double[:] item_rats, column_means
    #cdef dict di
    cdef cppmap[int, double] di
    cdef list rating_dicts

    rating_dicts = []
    column_means = np.zeros(ratings.shape[1])
    #column_means = np.squeeze(np.asarray(csc_mat.mean(axis=0)))
    print("Building rating_dicts, column_means")
    for item_col in range(ratings.shape[1]):
        item_rows = csc_indices[csc_indptr[item_col]:csc_indptr[item_col + 1]]
        item_rats = csc_data[csc_indptr[item_col]:csc_indptr[item_col + 1]]
        di = cppmap[int, double]()
        mn = 0
        for i in range(item_rows.size):
            di[item_rows[i]] = item_rats[i]
            mn += item_rats[i]

        rating_dicts.append(di)
        column_means[item_col] = mn / item_rows.size
    print("done!")

    out_counter = _inner_loop(out_row, out_col, out_data,
                              csr_indices, csr_indptr, csr_data,
                              csc_indices, csc_indptr, csc_data,
                              ratings.shape[1], column_means, rating_dicts)

    print("out_counter: " + str(out_counter) + " output_length: " + str(output_length))

    return coo_matrix((out_data, (out_row, out_col)), shape=(ratings.shape[1], ratings.shape[1]))


cdef int _inner_loop(int[:] out_row, int[:] out_col, double[:] out_data,
                     int[:] csr_indices, int[:] csr_indptr, double[:] csr_data,
                     int[:] csc_indices, int[:] csc_indptr, double[:] csc_data,
                     int num_cols, double[:] item_means, list rating_dicts):

    cdef int out_counter = 0
    cdef int[:] other_rows, item_rows
    cdef double[:] other_rats, user_rats
    cdef cppmap[int, double] other_rats_dict, user_rats_dict
    cdef int[:] ixn
    cdef double cov, item_mean, other_mean, denom_u, denom_o, udiff, odiff
    cdef int user, ixn_size
    cdef cppset[int] neighbours
    cdef int[:] tmp

    cdef int other_col, item_row, item_col, i, j
    for item_col in range(num_cols):
        item_rows = csc_indices[csc_indptr[item_col]:csc_indptr[item_col + 1]]
        item_rats = csc_data[csc_indptr[item_col]:csc_indptr[item_col + 1]]
        item_rats_dict = rating_dicts[item_col]
        neighbours = cppset[int]()
        for i in range(item_rows.shape[0]):
            item_row = item_rows[i]
            tmp = csr_indices[csr_indptr[item_row]:csr_indptr[item_row+1]]
            for j in range(tmp.shape[0]):
                neighbours.insert(tmp[j])
        for other_col in neighbours:
            other_rows = csc_indices[csc_indptr[other_col]:csc_indptr[other_col + 1]]
            other_rats = csc_data[csc_indptr[other_col]:csc_indptr[other_col + 1]]
            other_rats_dict = rating_dicts[other_col]
            ixn = np.intersect1d(item_rows, other_rows)
            ixn_size = ixn.size

            item_mean = item_means[item_col]
            other_mean = item_means[other_col]
            cov = 0
            denom_u = 0
            denom_o = 0
            for i in range(ixn_size):
                user = ixn[i]
                udiff = item_rats_dict[user] - item_mean
                odiff = other_rats_dict[user] - other_mean
                cov += udiff * odiff

            if ixn_size == 0:
                cov = 0
            else:
                cov /= ixn_size

            out_row[out_counter] = item_col
            out_col[out_counter] = other_col
            out_data[out_counter] = cov
            out_counter += 1

    return out_counter

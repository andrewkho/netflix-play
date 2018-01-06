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

def user_cor(ratings_mat):
    # type: (scipy.sparse.coo_matrix) -> scipy.sparse.coo_matrix
    """
    Calculate the user-wise Pearson correlation of the ratings matrix.
    Returns a sparse matrix in coo_matrix format

    :param ratings: ratings matrix in scipy.sparse.coo_matrix form
    :return: a sparse matrix in scipy.sparse.coo_matrix format
    """
    return py_get_cor(ratings_mat)


cdef py_get_cor(ratings):
    # type: (scipy.sparse.coo_matrix) -> scipy.sparse.coo_matrix

    cdef int[:] csr_indptr, csr_indices, csc_indptr, csc_indices
    cdef double[:] csr_data
    csr_mat = ratings.tocsr()
    csr_indptr = csr_mat.indptr
    csr_indices = csr_mat.indices
    csr_data = csr_mat.data
    csc_mat = ratings.tocsc()
    csc_indptr = csc_mat.indptr
    csc_indices = csc_mat.indices

    cdef long output_length
    output_length = ((csr_mat > 0).dot((csr_mat > 0).transpose()) > 0).sum()
    print("predicted output_length: " + str(output_length))

    cdef int[:] out_row, out_col
    cdef double[:] out_data
    cdef long out_counter
    out_row = np.zeros(output_length, dtype=np.int32)
    out_col = np.zeros(output_length, dtype=np.int32)
    out_data = np.zeros(output_length, dtype=np.float64)
    out_counter = 0

    cdef int user_row, i
    cdef int[:] user_cols
    cdef double[:] user_rats
    #cdef dict di
    cdef cppmap[int, double] di
    cdef list rating_dicts

    rating_dicts = []
    row_means = np.zeros(ratings.shape[0])
    print("Building rating_dicts")
    for user_row in range(ratings.shape[0]):
        user_cols = csr_indices[csr_indptr[user_row]:csr_indptr[user_row + 1]]
        user_rats = csr_data[csr_indptr[user_row]:csr_indptr[user_row + 1]]
        di = cppmap[int, double]()
        mn = 0
        for i in range(user_cols.size):
            di[user_cols[i]] = user_rats[i]
            mn += user_rats[i]

        rating_dicts.append(di)
        row_means[user_row] = mn / user_cols.size
    print("done!")

    out_counter = _inner_loop(out_row, out_col, out_data,
                              csr_indices, csr_indptr, csr_data,
                              csc_indices, csc_indptr, ratings.shape[0], row_means, rating_dicts)

    print("out_counter: " + str(out_counter) + " output_length: " + str(output_length))

    return coo_matrix((out_data, (out_row, out_col)), shape=(ratings.shape[0], ratings.shape[0]))

cdef int _inner_loop(int[:] out_row, int[:] out_col, double[:] out_data,
                     int[:] csr_indices, int[:] csr_indptr, double[:] csr_data,
                     int[:] csc_indices, int[:] csc_indptr, int num_rows,
                     double[:] user_means, list rating_dicts):

    cdef int out_counter = 0
    cdef int[:] other_cols, user_cols
    cdef double[:] other_rats, user_rats
    cdef cppmap[int, double] other_rats_dict, user_rats_dict
    cdef int[:] ixn
    cdef double corr, user_mean, other_mean, denom_u, denom_o, udiff, odiff
    cdef int movie, ixn_size
    cdef cppset[int] neighbours
    cdef int[:] tmp

    cdef int other_row, user_row, user_col, i, j
    for user_row in range(num_rows):
        user_cols = csr_indices[csr_indptr[user_row]:csr_indptr[user_row + 1]]
        user_rats = csr_data[csr_indptr[user_row]:csr_indptr[user_row + 1]]
        user_rats_dict = rating_dicts[user_row]
        neighbours = cppset[int]()
        for i in range(user_cols.shape[0]):
            user_col = user_cols[i]
            tmp = csc_indices[csc_indptr[user_col]:csc_indptr[user_col+1]]
            for j in range(tmp.shape[0]):
                neighbours.insert(tmp[j])
        for other_row in neighbours:
            other_cols = csr_indices[csr_indptr[other_row]:csr_indptr[other_row + 1]]
            other_rats = csr_data[csr_indptr[other_row]:csr_indptr[other_row + 1]]
            other_rats_dict = rating_dicts[other_row]
            ixn = np.intersect1d(user_cols, other_cols)
            ixn_size = ixn.size

            user_mean = user_means[user_row]
            other_mean = user_means[other_row]
            corr = 0
            denom_u = 0
            denom_o = 0
            for i in range(ixn_size):
                movie = ixn[i]
                udiff = user_rats_dict[movie] - user_mean
                odiff = other_rats_dict[movie] - other_mean
                corr += udiff*odiff
                denom_u += udiff*udiff
                denom_o += odiff*odiff

            corr /= sqrt(denom_u)*sqrt(denom_o)

            out_row[out_counter] = user_row
            out_col[out_counter] = other_row
            if np.isnan(corr):
                out_data[out_counter] = 0
            else:
                out_data[out_counter] = corr
            out_counter += 1

    return out_counter

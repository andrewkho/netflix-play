#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

import numpy as np
cimport numpy as np

from scipy.sparse import coo_matrix

from core.ratings import Ratings


def rating_cov(ratings):
    # type: (Ratings) -> scipy.sparse.coo_matrix
    return py_get_cov(ratings)


cdef py_get_cov(ratings):
    # type: (Ratings) -> scipy.sparse.coo_matrix

    cdef int[:] csr_indptr, csr_indices, csc_indptr, csc_indices
    cdef double[:] csr_data
    csr_mat = ratings.get_csr_matrix()
    csr_indptr = csr_mat.indptr
    csr_indices = csr_mat.indices
    csr_data = csr_mat.data
    csc_mat = ratings.get_coo_matrix().tocsc()
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

    cdef int user_row, other_row, user_col, movie, i
    cdef int[:] user_cols, other_cols, ixn
    cdef double[:] user_rats, other_rats
    cdef dict user_rats_dict, other_rats_dict
    cdef double user_mean, other_mean, covariance, rating
    cdef set neighbours
    cdef list rating_dicts

    rating_dicts = []
    print("Building rating_dicts")
    for user_row in range(ratings.num_rows):
        user_cols = csr_indices[csr_indptr[user_row]:csr_indptr[user_row + 1]]
        user_rats = csr_data[csr_indptr[user_row]:csr_indptr[user_row + 1]]
        di = dict()
        for i in range(user_cols.size):
            di[user_cols[i]] = user_rats[i]
        rating_dicts.append(di)
    print("done!")

    out_counter = _inner_loop(out_row, out_col, out_data,
                              csr_indices, csr_indptr, csr_data,
                              csc_indices, csc_indptr, ratings.num_rows, rating_dicts)

    print("out_counter: " + str(out_counter) + " output_length: " + str(output_length))

    return coo_matrix((out_data, (out_row, out_col)), shape=(ratings.num_rows, ratings.num_rows))

cdef int _inner_loop(int[:] out_row, int[:] out_col, double[:] out_data,
                     int[:] csr_indices, int[:] csr_indptr, double[:] csr_data,
                     int[:] csc_indices, int[:] csc_indptr, int num_rows,
                     list rating_dicts):

    cdef int out_counter = 0
    cdef int[:] other_cols, user_cols
    cdef double[:] other_rats, user_rats
    cdef dict other_rats_dict, user_rats_dict
    cdef int[:] ixn
    cdef double covariance, user_mean, other_mean
    cdef int movie
    cdef set neighbours

    cdef int other_row, user_row, user_col
    for user_row in range(num_rows):
        user_cols = csr_indices[csr_indptr[user_row]:csr_indptr[user_row + 1]]
        user_rats = csr_data[csr_indptr[user_row]:csr_indptr[user_row + 1]]
        user_rats_dict = rating_dicts[user_row]
        #user_rats_dict = dict()
        #for i in range(user_cols.size):
        #    user_rats_dict[user_cols[i]] = user_rats[i]
        neighbours = set()
        for user_col in user_cols:
            neighbours = neighbours.union(csc_indices[csc_indptr[user_col]:csc_indptr[user_col + 1]])
        for other_row in neighbours:
            other_cols = csr_indices[csr_indptr[other_row]:csr_indptr[other_row + 1]]
            other_rats = csr_data[csr_indptr[other_row]:csr_indptr[other_row + 1]]
            other_rats_dict = rating_dicts[other_row]
            #other_rats_dict = dict()
            #for i in range(other_cols.size):
            #    other_rats_dict[other_cols[i]] = other_rats[i]
            ixn = np.intersect1d(user_cols, other_cols)
            if ixn.size == 1:
                covariance = 1
            else:
                user_mean = 0
                other_mean = 0
                for movie in ixn:
                    user_mean += user_rats_dict[movie]
                    other_mean += other_rats_dict[movie]
                user_mean /= ixn.size
                other_mean /= ixn.size
                # print("%f, %f" % (user_mean, other_mean))
                covariance = 0
                for movie in ixn:
                    covariance += (user_rats_dict[movie] - user_mean) * (other_rats_dict[movie] - other_mean)
                covariance /= ixn.size - 1

            out_row[out_counter] = user_row
            out_col[out_counter] = other_row
            out_data[out_counter] = covariance
            out_counter += 1

    return out_counter

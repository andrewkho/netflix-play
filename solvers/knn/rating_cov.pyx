#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

import numpy as np
cimport numpy as np

from scipy.sparse import coo_matrix

from core.ratings import Ratings


def rating_cov(ratings):
    # type: (Ratings) -> np.ndarray

    if False:
        # print("Allocating array size: " + str((ratings.num_rows, ratings.num_rows)))
        # cov = np.zeros(shape=(ratings.num_rows, ratings.num_rows), dtype=np.float32)
        print("Filling cov array")
        # _get_cov(cov, ratings.get_coo_matrix().todense(), ratings.num_rows, ratings.num_cols,
        #            ratings.get_coo_matrix().row, ratings.get_coo_matrix().col, ratings.get_coo_matrix().data)
        csr_mat = ratings.get_csr_matrix()
        # _get_cov(cov, ratings.get_coo_matrix().todense(), ratings.num_rows, ratings.num_cols,
        #         csr_mat.data, csr_mat.indices, csr_mat.indptr)

        output_length = ((csr_mat > 0).dot((csr_mat > 0).transpose()) > 0).sum()
        print("predicted output_length: " + str(output_length))

        out_row = np.zeros(output_length, dtype=np.int32)
        out_col = np.zeros(output_length, dtype=np.int32)
        out_data = np.zeros(output_length, dtype=np.float64)
        _get_cov(out_row, out_col, out_data, ratings.num_rows, ratings.num_cols, csr_mat.data, csr_mat.indices,
                 csr_mat.indptr)
        cov = coo_matrix((out_data, (out_row, out_col)), shape=(ratings.num_rows, ratings.num_rows))
        print("Done")
        return cov
    else:
        return py_get_cov(ratings)


def py_get_cov(ratings):
    cdef
    int[:]
    csr_indptr, csr_indices, csc_indptr, csc_indices
    cdef
    double[:]
    csr_data
    csr_mat = ratings.get_csr_matrix()
    csr_indptr = csr_mat.indptr
    csr_indices = csr_mat.indices
    csr_data = csr_mat.data
    csc_mat = ratings.get_coo_matrix().tocsc()
    csc_indptr = csc_mat.indptr
    csc_indices = csc_mat.indices

    cdef
    long
    output_length
    output_length = ((csr_mat > 0).dot((csr_mat > 0).transpose()) > 0).sum()
    print("predicted output_length: " + str(output_length))

    cdef
    int[:]
    out_row, out_col
    cdef
    double[:]
    out_data
    cdef
    long
    out_counter
    out_row = np.zeros(output_length, dtype=np.int32)
    out_col = np.zeros(output_length, dtype=np.int32)
    out_data = np.zeros(output_length, dtype=np.float64)
    out_counter = 0

    cdef
    int
    user_row, other_row, user_col
    cdef
    int[:]
    user_cols, other_cols, ixn
    cdef
    double[:]
    user_rats, other_rats
    cdef
    dict
    user_rats_dict, other_rats_dict
    cdef
    double
    covariance
    cdef
    set
    neighbours
    for user_row in range(ratings.num_rows):
        user_cols = csr_indices[csr_indptr[user_row]:csr_indptr[user_row + 1]]
        user_rats = csr_data[csr_indptr[user_row]:csr_indptr[user_row + 1]]
        user_rats_dict = {movie: rating for movie, rating in zip(user_cols, user_rats)}
        neighbours = set()
        for user_col in user_cols:
            neighbours = neighbours.union(csc_indices[csc_indptr[user_col]:csc_indptr[user_col + 1]])
        for other_row in neighbours:
            other_cols = csr_indices[csr_indptr[other_row]:csr_indptr[other_row + 1]]
            other_rats = csr_data[csr_indptr[other_row]:csr_indptr[other_row + 1]]
            other_rats_dict = {movie: rating for movie, rating in zip(other_cols, other_rats)}
            ixn = np.intersect1d(user_cols, other_cols)
            if ixn.size == 1:
                covariance = 1
            else:
                user_mean = np.mean([user_rats_dict[movie] for movie in ixn])
                other_mean = np.mean([other_rats_dict[movie] for movie in ixn])
                # print("%f, %f" % (user_mean, other_mean))
                covariance = 0
                for movie in ixn:
                    covariance += (user_rats_dict[movie] - user_mean) * (other_rats_dict[movie] - other_mean)
                covariance /= ixn.size - 1

            out_row[out_counter] = user_row
            out_col[out_counter] = other_row
            out_data[out_counter] = covariance
            out_counter += 1

    print("out_counter: " + str(out_counter) + " output_length: " + str(output_length))

    return coo_matrix((out_data, (out_row, out_col)), shape=(ratings.num_rows, ratings.num_rows))

#cdef _get_cov(int num_rows, int num_cols, np.ndarray[np.int32_t, ndim=1] I, np.ndarray[np.int32_t, ndim=1] J, np.ndarray[np.float64_t, ndim=1] R):
# cdef void _get_cov(float[:,:] cov, double[:,:] mat, int num_rows, int num_cols, int[:] I, int[:] J, double[:] R):# nogil:
# cdef void _get_cov(float[:,:] cov, double[:,:] mat, int num_rows, int num_cols, double[:] data, int[:] indices, int[:] indptr):# nogil:
cdef
void
_get_cov(int[:]
out_row, int[:]
out_col, double[:]
out_data, int
num_rows, int
num_cols, double[:]
data, int[:]
indices, int[:]
indptr):
# print("HI")

# ind_mat = (mat > 0)
# print("ind_mat " + str(type(ind_mat)))
# n_mat = ind_mat.dot(ind_mat.transpose()).todense()
# mean_mat = csr_mat.dot(ind_mat.transpose()).todense()
# n_mat_coo = n_mat.tocoo()
# print("HI1")
# for i in range(n_mat_coo.size):
# mean_mat[n_mat_coo.row[i], n_mat_coo.col[i]] /= n_mat_coo.data[i]

# print("HI2")

# cdef double[:] user_row, other_row
cdef
int
user_row, user_col
cdef
double
tot = 0
for user_row in range(num_rows):
    for user_col in range(indptr[user_row], indptr[user_row + 1]):
        print("i: " + str(user_row) + " j: " + str(indices[user_col]) + " r: " + str(data[user_col]))
        tot += data[user_col]


# for i in range(num_rows):
#    user_row = mat[i]
#    #print("user_row shape: " + str(user_row.shape[0]))
#    for j in range(num_rows):
#        other_row = mat[j]
#        tot = 0
#        for k in range(user_row.shape[0]):
#            print("user_row: " + str(i) + " data: " + str(user_row[k]))
#            if user_row[k] != 0 and other_row[k] != 0:
#                tot += 1
#        cov[i,j] = tot


def get_cov(user_row, other_row):
    #ixn = np.intersect1d(user_row.nonzero()[1], other_row.nonzero()[1])

    cdef np.ndarray[np.int32_t, ndim=1] ixn
    ixn = np.intersect1d(user_row.nonzero()[1], other_row.nonzero()[1])

    if ixn.size <= 1:
        return 0

    cdef float mean_user, mean_other
    cdef int i

    mean_user = 0
    mean_other = 0
    for i in ixn:
        mean_user += user_row[0, i]
        mean_other += user_row[0, i]
    mean_user /= ixn.size
    mean_other /= ixn.size

    cdef float cov
    cov = 0
    for i in ixn:
        cov += (user_row[0, i] - mean_user)*(other_row[0, i] - mean_other)

    return cov / ixn.size - 1

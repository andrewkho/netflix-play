#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

import numpy as np
cimport numpy as np

from scipy.sparse import csr_matrix

from core.ratings import Ratings


def rating_cov(ratings):
    # type: (Ratings) -> np.ndarray

    if True:
        return _get_cov(ratings.get_csr_matrix(), ratings.num_rows, ratings.num_cols,
                    ratings.get_coo_matrix().row, ratings.get_coo_matrix().col, ratings.get_coo_matrix().data)

    cdef np.ndarray[np.float32_t, ndim=2] cov
    cdef np.ndarray[np.int32_t, ndim=1] user_row, other_row
    cdef int i, j, num_rows

    cov = np.zeros(shape=(ratings.num_rows, ratings.num_rows), dtype=np.float32)

    csr_mat = ratings.get_csr_matrix()

    ##cdef csr_matrix user_row, other_row
    num_rows = ratings.num_rows

    for i in range(num_rows):
        #    if i % 100 == 0:
        #        print("  Progress: %04d / %04d" % (i, num_rows))
        user_row = csr_mat.getrow(i)

        for j in range(num_rows):
            #print("    Progress j: %04d / %04d" % (j, num_rows))
            other_row = csr_mat.getrow(j)
            #cov[i,j] = get_cov(user_row, other_row)
            cov[i,j] = 1

    return cov


#cdef _get_cov(int num_rows, int num_cols, np.ndarray[np.int32_t, ndim=1] I, np.ndarray[np.int32_t, ndim=1] J, np.ndarray[np.float64_t, ndim=1] R):
cdef _get_cov(csr_mat, int num_rows, int num_cols, int[:] I, int[:] J, double[:] R):
    cdef float[:, :] cov

    cov = np.zeros(shape=(num_rows, num_rows), dtype=np.float32)
    print("HI")

    ind_mat = (csr_mat > 0)
    n_mat = ind_mat.dot(ind_mat.transpose())
    mean_mat = csr_mat.dot(ind_mat.transpose())
    n_mat_coo = n_mat.tocoo()
    cdef int i, j
    print("HI1")
    for i in range(n_mat_coo.size):
        mean_mat[n_mat_coo.row[i], n_mat_coo.col[i]] /= n_mat_coo.data[i]

    print("HI2")

    cdef np.ndarray[np.int32_t, ndim=1] ixn
    for i in range(num_rows):
        i_index = ind_mat.getrow(i)
        for j in range(num_rows):
            #ixn = np.intersect1d(J[I == i], J[I == j])
            cov[i,j] = 1 #ixn.sum()

    return cov

cdef _get_index(int[:] I, int t):
    cdef list l = list()
    cdef int i
    for i in I:
        if i==t:
            l.append(i)

    return np.array(l, dtype=np.int32)

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

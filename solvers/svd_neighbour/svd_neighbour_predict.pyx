#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

cimport numpy as np

from util.bin_search cimport bin_search

cpdef void cy_svdn_predict(
        double[:] preds, int[:] uidxs, int[:] midxs, int knn_k, double[:,:] correlations, long[:,:] neighbours,
        int[:] csc_indptr, int[:] csc_indices, double[:] csc_data) nogil:

    cdef int i
    cdef int user_idx, movie_idx
    cdef int[:] potentials
    cdef double[:] potential_ratings
    for i in range(preds.shape[0]):
        user_idx = uidxs[i]
        movie_idx = midxs[i]

        potentials = csc_indices[csc_indptr[movie_idx]:csc_indptr[movie_idx+1]]
        potential_ratings = csc_data[csc_indptr[movie_idx]:csc_indptr[movie_idx+1]]

        preds[i] = cy_svdn_predict_single(user_idx, correlations, knn_k,
                                          neighbours[user_idx, :], potentials, potential_ratings)


cpdef double cy_svdn_predict_single(int user_idx, double[:,:] correlation, int knn_k,
                      long[:] neighbours, int[:] potentials, double[:] ratings) nogil:
    cdef int counter = 0
    cdef double tot = 0.
    cdef double denom = 0.
    cdef int i, j
    cdef long nei_idx
    cdef double rating

    for j in range(neighbours.shape[0]):
        nei_idx = neighbours[j]
        i = bin_search(0, potentials.shape[0], potentials, nei_idx)
        if i >= potentials.shape[0] or potentials[i] != nei_idx:
            continue
        rating = ratings[i]
        counter += 1
        tot += rating * correlation[user_idx, nei_idx]
        denom += correlation[user_idx, nei_idx]
        if counter >= knn_k:
            break

    tot /= denom
    return tot



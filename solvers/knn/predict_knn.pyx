#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

import numpy as np
cimport numpy as np

import scipy.sparse
from scipy.sparse import csr_matrix

from core.ratings import Ratings
#from solvers.knn.knn_solver import KNNSolver


def predict_ratings(knn_solver, ratings):
    # type: (KNNSolver, Ratings) -> np.array

    cdef int n, i
    cdef double[:] pred
    cdef int uidx, midx, uid, mid
    n = ratings.get_coo_matrix().row.shape[0]
    pred = np.zeros(n, dtype=np.float64)
    ratings_dok = knn_solver._ratings.get_coo_matrix().todok()
    for i in range(n):
        uidx = ratings.get_coo_matrix().row[i]
        midx = ratings.get_coo_matrix().col[i]
        uid, mid = ratings.reverse_translate(uidx, midx)
        pred[i] = predict_single(knn_solver, ratings_dok, uid, mid)

    return pred

def predict_single(knn_solver, ratings_dok, int uid, int mid):
    # type: (KNNSolver, int, int) -> float

    #cdef int user_idx, movie_idx
    user_idx, movie_idx = knn_solver._ratings.translate(uid, mid)

    #cdef int[:] uindices
    #cdef double[:] covar
    uindices = knn_solver._cov.indices[knn_solver._cov.indptr[user_idx]:knn_solver._cov.indptr[user_idx+1]]
    ucovar = knn_solver._cov.data[knn_solver._cov.indptr[user_idx]:knn_solver._cov.indptr[user_idx+1]]

    #cdef dict neighbours
    neighbours = {}

    #cdef int i, counter, neiidx
    #cdef long[:] sorted_neighbours
    #cdef double rating
    counter = 0
    sorted_neighbours = ucovar.argsort()[::-1]
    for i in sorted_neighbours:
        neiidx = uindices[i]
        rating = ratings_dok[neiidx, movie_idx]
        if rating == 0 or rating is None: # Neighbour hasn't rated movie
            continue
        neighbours[neiidx] = rating
        counter += 1
        if counter >= knn_solver.k:
            break

    #if counter < self.k:
    #    print('Warning: not enough neighbours for k, counter: ' + str(counter))

    # Just try an average
    #cdef int neiId
    #cdef double tot
    tot = 0.
    for neiId, rating in neighbours.items():
        tot += rating

    return tot / counter


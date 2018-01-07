import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from itertools import izip

from core.ratings import Ratings
from solvers.recommenderAlgorithm import RecommenderAlgorithm

from util.mle_cov import mle_cov
from util.incomplete_projection import incomplete_projection

from solvers.svd_neighbour.svd_neighbour_predict import cy_svdn_predict, cy_svdn_predict_single


class SvdNeighbourSolver(RecommenderAlgorithm):
    """
    Generates a SVD representation of the ratings matrix
    """

    def __init__(self, svd_k, knn_k):
        # type: (int) -> None
        self.svd_k = svd_k  # type: int
        self.knn_k = knn_k  # type: int
        self._ratings = None  # type: Ratings
        self._cov = None  # type: scipy.sparse.csr_matrix

        self._svd_u = None # type: np.ndarray
        self._svd_s = None # type: np.ndarray
        self._svd_vt = None # type: np.ndarray
        self._eigenratings = None # type:np.ndarray

    def train(self, ratings, seed = None):
        # type: (Ratings) -> None
        self._ratings = ratings

        ## compute and store data needed for predictions
        print "  mle covariance"
        self._cov = mle_cov(self._ratings.get_coo_matrix())
        print "  %d top svd" % self.svd_k
        self._svd_u, self._svd_s, self._svd_vt = scipy.sparse.linalg.svds(self._cov, self.svd_k)
        print "  eigenratings"
        self._eigenratings = incomplete_projection(self._ratings.get_csr_matrix(), self._svd_u)
        print "  ratings correlation in eigen space"
        self._correlation = np.corrcoef(self._eigenratings)
        print "  Nearest neighbour sort"
        self._neighbours = np.argsort(self._correlation, axis=1)[:, ::-1][:, 1:]

    def predict(self, testratings):
        # type: (Ratings) -> np.array
        if True:
            return self.predict_cy(testratings)

        n = testratings.get_coo_matrix().nnz
        pred = np.zeros(n)
        rat_mat = self._ratings.get_coo_matrix().tolil()
        csc_mat = self._ratings.get_csc_matrix()

        for i in range(n):
            uidx = testratings.get_coo_matrix().row[i]
            midx = testratings.get_coo_matrix().col[i]

            uid, mid = testratings.reverse_translate(uidx, midx)
            pred[i] = self.predict_single(uid, mid, csc_mat, rat_mat)

        return pred

    def predict_cy(self, testratings):
        # type: (Ratings) -> np.array
        n = testratings.get_coo_matrix().nnz
        uidxs = np.zeros(n, dtype=np.int32)
        midxs = np.zeros(n, dtype=np.int32)
        preds = np.zeros(n, dtype=np.float64)
        for i in range(n):
            uidx = testratings.get_coo_matrix().row[i]
            midx = testratings.get_coo_matrix().col[i]

            uid, mid = testratings.reverse_translate(uidx, midx)
            try:
                this_user_idx, this_movie_idx = self._ratings.translate(uid, mid)
            except KeyError:
                this_user_idx = None
                this_movie_idx = None
            uidxs[i] = this_user_idx
            midxs[i] = this_movie_idx

        csc_mat = self._ratings.get_csc_matrix()

        cy_svdn_predict(preds, uidxs, midxs, self.knn_k, self._correlation, self._neighbours,
                        csc_mat.indptr, csc_mat.indices, csc_mat.data)

        return preds


    def predict_single(self, uid, mid, csc_mat, rat_mat):
        # type: (int, int, scipy.sparse.csc_matrix) -> float

        try:
            user_idx, movie_idx = self._ratings.translate(uid, mid)
        except KeyError:
            return None

        # neighbours = self._neighbours[user_idx, :]
        # return _py_svdn_predict(user_idx, movie_idx, self._correlation, self.knn_k, rat_mat, neighbours)

        potentials = csc_mat.indices[csc_mat.indptr[movie_idx]:csc_mat.indptr[movie_idx+1]]
        potential_ratings = csc_mat.data[csc_mat.indptr[movie_idx]:csc_mat.indptr[movie_idx+1]]
        return cy_svdn_predict_single(user_idx, self._correlation, self.knn_k, self._neighbours[user_idx, :], potentials, potential_ratings)


def _py_svdn_predict(user_idx, movie_idx, correlation, knn_k, rat_mat, neighbours):
    """
    Deprecated, use cy_svdn_predict_single instead, it's much faster
    :param user_idx:
    :param movie_idx:
    :param correlation:
    :param knn_k:
    :param rat_mat:
    :param neighbours:
    :return:
    """
    counter = 0
    tot = 0.
    denom = 0.

    for nei_idx in neighbours:
        rating = rat_mat[nei_idx, movie_idx]
        if rating == 0:
            continue
        counter += 1
        tot += rating * correlation[user_idx, nei_idx]
        denom += correlation[user_idx, nei_idx]
        if counter >= knn_k:
            break

    #if counter == self.knn_k:
    #    print "found k neighbours!"
    #    #print "Warning, counter less than knn_k (%d, %d)" % (counter, self.knn_k)

    tot /= denom
    return tot




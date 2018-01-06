import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from itertools import izip

from core.ratings import Ratings
from solvers.recommenderAlgorithm import RecommenderAlgorithm

from util.mle_cov import mle_cov
from util.incomplete_projection import incomplete_projection

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
        self._cov = mle_cov(self._ratings.get_coo_matrix())
        self._svd_u, self._svd_s, self._svd_vt = scipy.sparse.linalg.svds(self._cov, self.svd_k)
        self._eigenratings = incomplete_projection(self._ratings.get_csr_matrix(), self._svd_u)
        self._correlation = np.corrcoef(self._eigenratings)
        self._neighbours = np.argsort(self._correlation, axis=1)[:, ::-1][:, 1:]

    def predict(self, testratings):
        # type: (Ratings) -> np.array
        n = testratings.get_coo_matrix().nnz
        print("predicting %d testratigns: " % n)
        pred = np.zeros(n)
        lil_mat = self._ratings.get_lil_matrix()

        for i in range(n):
            uidx = testratings.get_coo_matrix().row[i]
            midx = testratings.get_coo_matrix().col[i]

            uid, mid = testratings.reverse_translate(uidx, midx)
            pred[i] = self.predict_single(uid, mid, lil_mat)

        print("len(pred): " + str(len(pred)))

        return pred

    def predict_single(self, uid, mid, lil_mat):
        # type: (int, int, scipy.sparse.lil_matrix) -> float

        try:
            user_idx, movie_idx = self._ratings.translate(uid, mid)
        except KeyError:
            return None

        neighbours = self._neighbours[user_idx, :]
        counter = 0
        tot = 0.
        denom = 0.
        for nei_idx in neighbours:
            rating = lil_mat[nei_idx, movie_idx]
            if rating == 0:
                continue
            counter += 1
            tot += rating * self._correlation[user_idx, nei_idx]
            denom += self._correlation[user_idx, nei_idx]
            if counter >= self.knn_k:
                break

        if counter < self.knn_k:
            print "Warning, counter less than knn_k (%d, %d)" % (counter, self.knn_k)

        tot /= denom
        return tot






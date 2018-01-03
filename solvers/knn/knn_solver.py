import numpy as np
import scipy.sparse

from core.ratings import Ratings
from rating_cov import rating_cov
from solvers.recommenderAlgorithm import RecommenderAlgorithm


class KNNSolver(RecommenderAlgorithm):
    """
    Get rating for one user-movie by doing regression on his K-nearest neighbours
    """

    def __init__(self, k, dist="cov"):
        # type: (int, str) -> None
        self.k = k  # type: int
        self.dist = dist  # type: str
        self._ratings = None  # type: Ratings
        self._cov = None  # type: scipy.sparse.csr_matrix

    def train(self, ratings):
        # type: (Ratings) -> None

        self._ratings = ratings
        self._cov = rating_cov(self._ratings).tocsr()

    def predict(self, ratings):
        # type: (Ratings) -> np.array
        n = ratings.get_coo_matrix().row.shape[0]
        pred = np.zeros(n)
        #dok_matrix = self._ratings.get_coo_matrix().todok()  ## DoK matrix is faster but different MSE (!@?!?!?!)
        dok_matrix = self._ratings.get_coo_matrix().tocsr()
        for i in range(n):
            uidx = ratings.get_coo_matrix().row[i]
            midx = ratings.get_coo_matrix().col[i]
            uid, mid = ratings.reverse_translate(uidx, midx)
            pred[i] = self.predict_single(dok_matrix, uid, mid)

        return pred

    def predict_single(self, dok_matrix, uid, mid):
        # type: (int, int) -> float

        user_idx, movie_idx = self._ratings.translate(uid, mid)

        uindices = self._cov.indices[self._cov.indptr[user_idx]:self._cov.indptr[user_idx+1]]
        ucovar = self._cov.data[self._cov.indptr[user_idx]:self._cov.indptr[user_idx+1]]

        neighbours = {}

        counter = 0
        sorted_neighbours = ucovar.argsort()[::-1]
        for nei_idx in sorted_neighbours:
            neiidx = uindices[nei_idx]
            rating = dok_matrix[neiidx, movie_idx]
            if rating == 0 or rating is None: # Neighbour hasn't rated movie
                continue
            neighbours[neiidx] = rating
            counter += 1
            if counter >= self.k:
                break

        #if counter < self.k:
        #    print('Warning: not enough neighbours for k, counter: ' + str(counter))

        # Just try an average
        tot = 0.
        for neiId, rating in neighbours.items():
            tot += rating

        return tot / counter

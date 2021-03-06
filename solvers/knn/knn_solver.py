import numpy as np
import scipy.sparse

from itertools import izip

from core.ratings import Ratings
from util.user_cor import user_cor
from solvers.recommenderAlgorithm import RecommenderAlgorithm


class KNNSolver(RecommenderAlgorithm):
    """
    Get rating for one user-movie by doing regression on his K-nearest neighbours
    """

    def __init__(self, k, dist="cor"):
        # type: (int, str) -> None
        self.k = k  # type: int
        self.dist = dist  # type: str
        self._ratings = None  # type: Ratings
        self._cor = None  # type: scipy.sparse.csr_matrix

    def train(self, ratings):
        # type: (Ratings) -> None

        self._ratings = ratings
        self._cor = user_cor(self._ratings.get_coo_matrix()).tocsr()

    def predict(self, testratings):
        # type: (Ratings) -> np.array
        n = testratings.get_coo_matrix().nnz
        print("nnz testratigns: " + str(n))
        pred = np.zeros(n)
        lil_matrix = self._ratings.get_coo_matrix().tolil()

        for i in range(n):
            uidx = testratings.get_coo_matrix().row[i]
            midx = testratings.get_coo_matrix().col[i]
            uid, mid = testratings.reverse_translate(uidx, midx)
            pred[i] = self.predict_single(lil_matrix, uid, mid)

        print("len(pred): " + str(len(pred)))

        return pred

    def predict_single(self, lil_matrix, uid, mid):
        # type: (int, int) -> float

        try:
            user_idx, movie_idx = self._ratings.translate(uid, mid)
        except KeyError:
            return None

        uindices = self._cor.indices[self._cor.indptr[user_idx]:self._cor.indptr[user_idx+1]]
        ucorr = self._cor.data[self._cor.indptr[user_idx]:self._cor.indptr[user_idx+1]]

        neighbours = {}

        counter = 0
        tot = 0.
        denom = 0.
        sorted_neighbours = ucorr.argsort()[::-1]
        for nei_idx in sorted_neighbours:
            neiidx = uindices[nei_idx]
            rating = lil_matrix[neiidx, movie_idx]
            if rating == 0 or rating is None: # Neighbour hasn't rated movie
                continue
            tot += rating * ucorr[nei_idx]
            denom += ucorr[nei_idx]
            counter += 1
            if counter >= self.k:
                break

        #if counter < self.k:
        #    print('Warning: not enough neighbours for k, counter: ' + str(counter))

        if counter == 0 or denom == 0:
            return None

        # Just try an average
        #tot = 0.
        #for neiId, rating in neighbours.items():
        #    tot += rating
        # tot /= counter

        return tot / denom

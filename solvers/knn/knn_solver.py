import numpy as np

from solvers.recommenderAlgorithm import RecommenderAlgorithm
from solvers.testResult import TestResult

from core.ratings import Ratings

from rating_cov import rating_cov

class KNNSolver(RecommenderAlgorithm):
    """
    Get rating for one user-movie by doing regression on his K-nearest neighbours
    """

    def __init__(self, k, dist="cov"):
        # type: (int, str) -> None
        self.k = k
        self.dist = dist
        self._ratings = None
        self._cov = None

    def train(self, ratings):
        # type: (Ratings) -> None

        self._ratings = ratings
        self._cov = rating_cov(self._ratings)

    def get_rating(self, ratings, userId, movieId):
        # type: (Ratings, int, int) -> float
        user_idx = ratings.get_idx_user(userId)
        movie_idx = ratings.get_idx_movie(movieId)

        user_row = self._cov[user_idx]

        neighbours = [-1] * self.k
        counter = 0
        sorted_neighbours = user_row.argsort()[::-1]
        for nei_idx in sorted_neighbours:
            neiId = ratings.get_idx_user(nei_idx)
            rating = ratings.get(neiId, movieId)
            if rating is None:
                continue
            neighbours[counter] = neiId
            counter += 1
            if counter >= self.k:
                break

        print('counter ' + str(counter))

        # Just try an average
        tot = 0.
        for neiId in neighbours:
            if neiId == -1:
                break
            tot = tot + self.data[neiId][movieId]

        return tot / counter

import numpy as np

from solvers.recommenderAlgorithm import RecommenderAlgorithm
from solvers.testResult import TestResult

from core.ratings import Ratings


def rating_cov(ratings):
    # type: (Ratings) -> np.matrix
    cov = np.zeros(shape=(ratings.num_rows, ratings.num_rows), dtype=np.float32)

    for i in range(ratings.num_rows):
        if i % 100 == 0:
            print("  Progress: %04d / %04d" % (i, ratings.num_rows))
        user_row = ratings.get_coo_matrix().getrow(i)
        for j in range(ratings.num_rows):
            other_row = ratings.get_coo_matrix().getrow(j)
            cov[i,j] = get_cov(user_row, other_row)

    return cov


def get_cov(user_row, other_row):
    ixn = np.intersect1d(user_row.nonzero()[1], other_row.nonzero()[1])
    if ixn.size <= 1:
        return 0

    mean_user = 0
    mean_other = 0
    for i in ixn:
        mean_user += user_row[0,i]
        mean_other += user_row[0,i]
    mean_user /= ixn.size
    mean_other /= ixn.size

    cov = 0
    for i in ixn:
        cov += (user_row[0,i] - mean_user)*(other_row[0,i] - mean_other)

    return cov / ixn.size - 1


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

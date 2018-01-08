import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from itertools import izip

from core.ratings import Ratings
from solvers.recommenderAlgorithm import RecommenderAlgorithm

from util.mle_cov import mle_cov
from util.incomplete_projection import incomplete_projection

from solvers.svd.svd_train_feature import svd_train_feature


class SvdSolver(RecommenderAlgorithm):
    """
    Form a full Left-Right factorization of the Ratings matrix

    """

    def __init__(self, num_features, learning_rate = 0.001, epsilon=1e-7, maxiters=1000, gamma = 0.02):
        # type: (int) -> None
        self.k = num_features
        self.learning_rate = learning_rate
        self.eps = epsilon
        self.maxiters = maxiters
        self.gamma = gamma

        self._left = None
        self._right = None

    def train(self, ratings, keep_intermed=False):
        # type: (Ratings) -> None
        if False:
            return self.train_all(ratings)

        self._ratings = ratings

        self._left = 0.1*np.ones(shape=(self._ratings.shape[0], self.k), dtype=np.float64)
        self._right = 0.1*np.ones(shape=(self._ratings.shape[1], self.k), dtype=np.float64)

        coo_mat = ratings.get_coo_matrix()
        y = coo_mat.data


        for k in range(self.k):
            print "  training feature %d" % k
            ## Initialize residual
            resid = y.astype(dtype=np.float32)
            for i in range(resid.shape[0]):
                resid[i] -= self._left[coo_mat.row[i], :].dot(self._right[coo_mat.col[i], :].transpose()).astype(np.float32)

            svd_train_feature(self._left, self._right, resid, k,
                              coo_mat.row, coo_mat.col, coo_mat.data,
                              self.learning_rate, self.eps, self.maxiters, self.gamma)
            print "  done. %f remaining residual" % np.sum(resid*resid)

    def train_all(self, ratings):
        self._ratings = ratings

        self._left = 0.1*np.ones(shape=(self._ratings.shape[0], self.k), dtype=np.float64)
        self._right = 0.1*np.ones(shape=(self._ratings.shape[1], self.k), dtype=np.float64)

        csr_mat = ratings.get_csr_matrix()
        indicator = csr_mat > 0
        print csr_mat.shape
        print indicator.shape
        print type (indicator)
        print (self._left.dot(self._right.transpose())).shape

        left = scipy.sparse.csr_matrix(self._left)
        right = scipy.sparse.csr_matrix(self._right)

        old_resid = 0
        counter = 0
        while True:
            E = csr_mat - left.dot(right.transpose()).multiply(indicator)
            resid = (E.multiply(E)).sum()
            #if counter % 10 == 0:
            print "resid: %e" % resid
            U = left + E.dot(right).multiply(self.learning_rate)
            V = right + E.transpose().dot(left).multiply(self.learning_rate)

            left = U
            right = V

            counter += 1
            if abs(old_resid - resid)/resid < self.eps or counter > self.maxiters:
                break
            old_resid = resid

        self._left = left.todense()
        self._right = right.todense()

    def predict(self, testratings):
        # type: (Ratings) -> np.array
        if False:
            return self.predict_cy(testratings)

        n = testratings.get_coo_matrix().nnz
        pred = np.zeros(n)
        rat_mat = self._ratings.get_coo_matrix().tolil()
        csc_mat = self._ratings.get_csc_matrix()
        csr_mat = self._ratings.get_csr_matrix()

        row_means = np.squeeze(np.array(csr_mat.sum(axis=1) / (csr_mat>0).sum(axis=1)))

        for i in range(n):
            uidx = testratings.get_coo_matrix().row[i]
            midx = testratings.get_coo_matrix().col[i]

            uid, mid = testratings.reverse_translate(uidx, midx)
            pred[i] = self.predict_single(uid, mid, csc_mat, row_means)

        return pred

    def predict_single(self, uid, mid, csc_mat, row_means):
        # type: (int, int, scipy.sparse.csc_matrix) -> float

        try:
            user_idx, movie_idx = self._ratings.translate(uid, mid)
        except KeyError:
            return None

        return self._left[user_idx, :].dot(self._right[movie_idx, :].transpose())


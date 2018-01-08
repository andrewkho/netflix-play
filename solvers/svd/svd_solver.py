from enum import Enum
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
from solvers.svd.svd_train_stochastic import svd_train_sgd

GradientDescentMethod = Enum("GradientDescentMethod", "full incremental stochastic")

class SvdSolver(RecommenderAlgorithm):
    """
    Form a full Left-Right factorization of the Ratings matrix

    """

    def __init__(self, num_features,
                 learning_rate=0.005,
                 epsilon=1e-7,
                 solver=GradientDescentMethod.stochastic,
                 maxiters=1000,
                 gamma=0.02,
                 include_bias=True):
        # type: (int) -> None
        self.k = num_features
        self.learning_rate = learning_rate
        self.eps = epsilon
        self.maxiters = maxiters
        self.gamma = gamma
        self.include_bias = include_bias

        self.solver = solver

        self._left = None
        self._right = None

        self._total_mean = None

    def train(self, ratings):
        # type: (Ratings) -> None
        if self.solver == GradientDescentMethod.full:
            return self.train_all(ratings)
        elif self.solver == GradientDescentMethod.incremental:
            return self.train_incremental(ratings)
        elif self.solver == GradientDescentMethod.stochastic:
            return self.train_stochastic(ratings)
        else:
            raise ValueError("Unknown solver %s!" % self.solver)

    def train_stochastic(self, ratings):
        print "  Mean centering all ratings (generates a new ratings matrix for myself)"
        self._total_mean = ratings.get_coo_matrix().sum() / ratings.get_coo_matrix().nnz
        print "  total_mean: %f" % self._total_mean
        ratings.get_coo_matrix().data -= self._total_mean
        self._ratings = ratings

        self._left = 0.1*np.ones(shape=(self._ratings.shape[0], self.k), dtype=np.float64)
        self._right = 0.1*np.ones(shape=(self._ratings.shape[1], self.k), dtype=np.float64)
        if self.include_bias:
            self._left[:, 0] = 1.
            self._right[:, 1] = 1.

        coo_mat = self._ratings.get_coo_matrix()
        y = coo_mat.data

        resid = y.astype(dtype=np.float32)
        svd_train_sgd(self._left, self._right, resid,
                      np.random.choice(resid.shape[0], size=resid.shape[0], replace=False).astype(dtype=np.int32),
                      coo_mat.row, coo_mat.col, coo_mat.data,
                      self.learning_rate, self.eps, self.maxiters, self.gamma,
                      self.include_bias)

        print "  done. %f remaining residual" % np.sum(resid*resid)

    def train_incremental(self, ratings):
        print "  Mean centering all ratings (generates a new ratings matrix for myself)"
        self._total_mean = ratings.get_coo_matrix().sum() / ratings.get_coo_matrix().nnz
        print "  total_mean: %f" % self._total_mean
        ratings.get_coo_matrix().data -= self._total_mean
        self._ratings = ratings

        self._left = 0.1*np.ones(shape=(self._ratings.shape[0], self.k), dtype=np.float64)
        self._right = 0.1*np.ones(shape=(self._ratings.shape[1], self.k), dtype=np.float64)
        if self.include_bias:
            self._left[:, self.k-1] = 1.
            self._right[:, self.k-2] = 1.

        coo_mat = self._ratings.get_coo_matrix()
        y = coo_mat.data

        for k in range(self.k):
            ignore_left = False
            ignore_right = False
            if self.include_bias and k == 0:
                ignore_left = True
            if self.include_bias and k == 1:
                ignore_right = True

            print "  training feature %d" % k
            resid = y.astype(dtype=np.float32)
            for i in range(resid.shape[0]):
                resid[i] -= self._left[coo_mat.row[i], :].dot(self._right[coo_mat.col[i], :].transpose()).astype(np.float32)

            svd_train_feature(self._left, self._right, resid, k,
                              coo_mat.row, coo_mat.col, coo_mat.data,
                              self.learning_rate, self.eps, self.maxiters, self.gamma,
                              ignore_left, ignore_right)
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
        n = testratings.get_coo_matrix().nnz
        pred = np.zeros(n)

        for i in range(n):
            uidx = testratings.get_coo_matrix().row[i]
            midx = testratings.get_coo_matrix().col[i]

            uid, mid = testratings.reverse_translate(uidx, midx)
            pred[i] = self.predict_single(uid, mid)

        return pred

    def predict_single(self, uid, mid):
        # type: (int, int, scipy.sparse.csc_matrix) -> float

        try:
            user_idx, movie_idx = self._ratings.translate(uid, mid)
        except KeyError:
            return None

        return self._total_mean + self._left[user_idx, :].dot(self._right[movie_idx, :].transpose())


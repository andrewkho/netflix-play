import unittest

import os
import numpy as np

import cPickle

from core.flixdata import FlixData
from solvers.kFoldsCrossValidator import KFolds
from core.ratings import Ratings

from solvers.knn.knn_solver import KNNSolver

flix_data_root = "../../../data/arrays/"
saved_data = "ratings_test.pkl"


class TestStringMethods(unittest.TestCase):

    def setUp(self):
        if os.path.isfile(saved_data):
            print ("Trying to read existing data from " + saved_data)
            with open(saved_data) as f:
                self.ratings, self.kfolds = cPickle.load(f)
            print ("Done!")
        else:
            print ("Couldn't find " + saved_data + ", regenerating")
            fd = FlixData(flix_data_root)

            np.random.seed(12345)
            idx = np.random.choice(fd.userIDsForUsers.size, size=fd.userIDsForUsers.size, replace=False)
            N = int(1e5)
            self.ratings = Ratings(fd.userIDsForUsers[idx[:N]],
                                   fd.movieIDs[idx[:N]],
                                   fd.userRatings[idx[:N]])

            self.kfolds = KFolds(self.ratings.size, 10, 12345)

            with open(saved_data, "wb") as f:
                cPickle.dump((self.ratings, self.kfolds), f, cPickle.HIGHEST_PROTOCOL)

    def test_one_fold(self):
        test, train = self.kfolds.get(0)
        test_set = self.ratings.get_index_split(test)
        train_set = self.ratings.get_index_split(train)

        knn = KNNSolver(k=15, dist="cov")
        knn.train(train_set)

        #print knn._cov.sum(axis=1)
        #print knn._cov.shape

        assert True


if __name__ == '__main__':
    unittest.main()

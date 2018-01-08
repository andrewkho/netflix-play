import cPickle
import os
import unittest

import cProfile

from itertools import izip
import numpy as np

from sklearn.cluster import KMeans

from core.flixdata import FlixData
from core.flixdata_subsampler import FlixDataSubsampler
from core.ratings import Ratings
from solvers.kFoldsCrossValidator import KFolds
from solvers.svd.svd_solver import SvdSolver

flix_data_root = "../../../data/arrays/"
saved_data = "ratings_test.pkl"


class TestSvd(unittest.TestCase):

    def setUp(self):
        if os.path.isfile(saved_data):
            print ("Trying to read existing data from " + saved_data)
            with open(saved_data) as f:
                self.ratings, self.kfolds = cPickle.load(f)
            print ("Done!")
        else:
            print ("Couldn't find " + saved_data + ", regenerating")
            fd = FlixData(flix_data_root)
            print "total ratings: %d" % fd.numratings
            #self.ratings = FlixDataSubsampler.get_all(fd)
            self.ratings = FlixDataSubsampler.random_sample_movies(fd, seed=12345, N=int(1e4), M=int(1e3))
            self.kfolds = KFolds(self.ratings.size, 40, 12345)

            with open(saved_data, "wb") as f:
                cPickle.dump((self.ratings, self.kfolds), f, cPickle.HIGHEST_PROTOCOL)
        print "Using dataset with %d users, %d movies" % (self.ratings.shape[0], self.ratings.shape[1])

    def test_one_fold(self):
        print "Splitting data into train, test"
        test, train = self.kfolds.get(0)
        test_set = self.ratings.get_index_split(test)
        train_set = self.ratings.get_index_split(train)
        print "Training SvdNeighbourSolver..."
        svd = SvdSolver(30, learning_rate=0.005, epsilon=1e-7, maxiters=5000, gamma=0.15)
        svd.train(train_set)

        print("predicting %d testratings: " % test_set.get_coo_matrix().nnz)
        pred = svd.predict(test_set)
        print "Done!"
        print("len(pred): " + str(len(pred)))

        y = test_set.get_coo_matrix().data

        print("%d / %d are None (%f)" % ((pred==None).sum(), pred.shape[0], (pred==None).mean()))
        print("%d / %d are NaN (%f)" % ((np.isnan(pred)).sum(), pred.shape[0], (np.isnan(pred)).mean()))
        #print [(_, __) for _, __ in izip(y, pred)]

        print "Test SSE: %f" % np.nansum((y-pred)**2)
        print "Test MSE: %f" % np.nanmean((y-pred)**2)
        print "Test RMS: %f" % np.sqrt(np.nanmean((y-pred)**2))
        print "mean of error: %f" % np.nanmean(y-pred)
        print "SD of error: %f" % np.nanstd(y-pred)

        assert True


if __name__ == '__main__':
    unittest.main()

import cPickle
import os
import unittest

import cProfile

from itertools import izip
import numpy as np

from core.flixdata import FlixData
from core.flixdata_subsampler import FlixDataSubsampler
from core.ratings import Ratings
from solvers.kFoldsCrossValidator import KFolds
from solvers.svd_neighbour.svd_neighbour_solver import SvdNeighbourSolver

flix_data_root = "../../../data/arrays/"
saved_data = "ratings_test.pkl"


class TestSvdNeighbour(unittest.TestCase):

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
            #self.ratings = FlixDataSubsampler.popularity_sample(fd, 12345, int(1e2), int(1e4))
            #self.ratings = FlixDataSubsampler.random_sample(fd, 12345, int(1e4))
            #self.ratings = FlixDataSubsampler.random_sample_users(fd, 12345, int(1e6), int(1e4), 9)
            self.ratings = FlixDataSubsampler.random_sample_movies(fd, seed=12345, N=int(2e4), M=int(1e3), minratings=9)
            self.kfolds = KFolds(self.ratings.size, 10, 12345)

            with open(saved_data, "wb") as f:
                cPickle.dump((self.ratings, self.kfolds), f, cPickle.HIGHEST_PROTOCOL)
        print "Using dataset with %d users, %d movies" % (self.ratings.shape[0], self.ratings.shape[1])

    def test_one_fold(self):
        test, train = self.kfolds.get(0)
        test_set = self.ratings.get_index_split(test)
        train_set = self.ratings.get_index_split(train)
        svdn = SvdNeighbourSolver(svd_k=15, knn_k=10)
        svdn.train(train_set, seed=54321)

        #y = train_set.get_coo_matrix().data
        #print "Generating prediction"
        #pred = svdn.predict(train_set)
        #print "Done!"

        #print "Train SSE: %f" % np.sum((y-pred)**2)
        #print "Train MSE: %f" % np.mean((y-pred)**2)
        #print "Train RMS: %f" % np.sqrt(np.mean((y-pred)**2))

        pred = svdn.predict(test_set)
        print "Done!"
        y = test_set.get_coo_matrix().data

        print("%d / %d are None (%f)" % ((pred==None).sum(), pred.shape[0], (pred==None).mean()))
        print("%d / %d are NaN (%f)" % ((np.isnan(pred)).sum(), pred.shape[0], (np.isnan(pred)).mean()))
        print [(_, __) for _, __ in izip(y, pred)]

        print "Test SSE: %f" % np.sum((y[~np.isnan(pred)]-pred[~np.isnan(pred)])**2)
        print "Test MSE: %f" % np.mean((y[~np.isnan(pred)]-pred[~np.isnan(pred)])**2)
        print "Test RMS: %f" % np.sqrt(np.mean((y[~np.isnan(pred)]-pred[~np.isnan(pred)])**2))

        assert True


if __name__ == '__main__':
    unittest.main()

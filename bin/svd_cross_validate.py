import sys
import logging

import cPickle
import os
import unittest

from itertools import izip
import numpy as np

from core.flixdata import FlixData
from core.flixdata_subsampler import FlixDataSubsampler

from core.ratings import Ratings

from solvers.kFoldsCrossValidator import KFolds
from solvers.svd.svd_solver import SvdSolver
from solvers.svd.svd_solver import GradientDescentMethod

flix_data_root = "../data/arrays/"
saved_data = "cross_valid_svd.pkl"


class TestSvd(object):

    def setUp(self):
        if os.path.isfile(saved_data):
            logging.info("Trying to read existing data from " + saved_data)
            with open(saved_data) as f:
                self.ratings, self.kfolds = cPickle.load(f)
            print ("Done!")
        else:
            logging.info("Couldn't find " + saved_data + ", regenerating")
            fd = FlixData(flix_data_root)
            logging.info("total ratings: %d" % fd.numratings)
            #self.ratings = FlixDataSubsampler.get_all(fd)
            self.ratings = FlixDataSubsampler.random_sample_movies(fd, seed=12345, N=int(2e4), M=int(1e3))
            self.kfolds = KFolds(self.ratings.size, 10, 12345)

            with open(saved_data, "wb") as f:
                cPickle.dump((self.ratings, self.kfolds), f, cPickle.HIGHEST_PROTOCOL)
        logging.info("Using dataset with %d users, %d movies" % (self.ratings.shape[0], self.ratings.shape[1]))


    def test_perform_cv(self):
        errs = list()
        for fold in range(self.kfolds.k):
            logging.info("Starting fold %d " % fold)
            rmse = self.perform_one_fold(fold)
            errs.append(rmse)
            logging.info("Finished fold %d with rmse: %e" % (fold, rmse))

        logging.info("Finished Cross Validation")
        logging.info("RMS: %s" + str(errs))
        logging.info("NaNs: %d" + np.isnan(rmse).sum())
        logging.info("mean: %e, max: %e, min: %e, sd: %e" % (np.nanmean(errs),
                                                             np.nanmax(errs),
                                                             np.nanmin(errs),
                                                             np.nanstd(errs)))

    def perform_one_fold(self, k):
        logging.info("Splitting data into train, test")
        test, train = self.kfolds.get(k)
        test_set = self.ratings.get_index_split(test)
        train_set = self.ratings.get_index_split(train)
        logging.info("Training SvdNeighbourSolver...")
        svd = SvdSolver(32,
                        learning_rate=0.005,
                        epsilon=1e-7,
                        solver=GradientDescentMethod.stochastic,
                        maxiters=10000,
                        gamma=0.05,
                        include_bias=True)
        svd.train(train_set)

        logging.info(svd._left)
        logging.info(svd._right)

        logging.info("predicting %d testratings: " % test_set.get_coo_matrix().nnz)
        pred = svd.predict(test_set)
        logging.info( "Done!")
        logging.info("len(pred): " + str(len(pred)))

        y = test_set.get_coo_matrix().data

        logging.info("%d / %d are None (%f)" % ((pred==None).sum(), pred.shape[0], (pred==None).mean()))
        logging.info("%d / %d are NaN (%f)" % ((np.isnan(pred)).sum(), pred.shape[0], (np.isnan(pred)).mean()))
        logging.info([(_, __) for _, __ in izip(y, pred)])

        logging.info("Test SSE: %f" % np.nansum((y-pred)**2))
        logging.info("Test MSE: %f" % np.nanmean((y-pred)**2))
        rmse = np.sqrt(np.nanmean((y-pred)**2))
        logging.info("Test RMS: %f" % rmse)
        logging.info("mean of error: %f" % np.nanmean(y-pred))
        logging.info("SD of error: %f" % np.nanstd(y-pred))

        return rmse


if __name__ == '__main__':
    logging.basicConfig(filename="svd_cross_validate.log",
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("*******************************************")
    logging.info("Starting run of Cross Validation with params")

    test_svd = TestSvd()
    test_svd.setUp()
    test_svd.test_perform_cv()

    logging.info("Finished run of Cross Validation")
    logging.info("*******************************************")


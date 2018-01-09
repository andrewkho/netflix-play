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

    def __init__(self, k, learning_rate, gamma, epsilon=1e-7,
                 solver=GradientDescentMethod.stochastic, maxiters=10000, include_bias=True):
        self.k = k
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.solver = solver
        self.maxiters = maxiters
        self.include_bias = include_bias

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
            self.kfolds = KFolds(self.ratings.size, 5, 12345)

            with open(saved_data, "wb") as f:
                cPickle.dump((self.ratings, self.kfolds), f, cPickle.HIGHEST_PROTOCOL)
        logging.info("Using dataset with %d users, %d movies" % (self.ratings.shape[0], self.ratings.shape[1]))


    def test_perform_cv(self):
        logging.info("######################################")
        logging.info("Starting cross validation with params:")
        logging.info("  k:        %d" % self.k)
        logging.info("  learn:    %f" % self.learning_rate)
        logging.info("  gamma:    %f" % self.gamma)
        logging.info("  epsilon:  %e" % self.epsilon)
        logging.info("  solver:   %s" % str(self.solver))
        logging.info("  maxiters: %d" % self.maxiters)
        logging.info("  bias:     %s" % str(self.include_bias))

        errs = list()
        for fold in range(self.kfolds.k):
            logging.info("Starting fold %d " % fold)
            rmse = self.perform_one_fold(fold)
            errs.append(rmse)
            logging.info("Finished fold %d with rmse: %e" % (fold, rmse))

        logging.info("Finished Cross Validation")
        logging.info("RMS: %s" % str(errs))
        logging.info("NaNs: %d" % np.isnan(rmse).sum())
        err_summary = (np.nanmean(errs), np.nanmax(errs), np.nanmin(errs), np.nanstd(errs))

        logging.info("mean: %e, max: %e, min: %e, sd: %e" % err_summary)

        return err_summary

    def perform_one_fold(self, k):
        logging.info("Splitting data into train, test")
        test, train = self.kfolds.get(k)
        test_set = self.ratings.get_index_split(test)
        train_set = self.ratings.get_index_split(train)
        logging.info("Training SvdNeighbourSolver...")
        svd = SvdSolver(self.k,
                        learning_rate=self.learning_rate,
                        epsilon=self.epsilon,
                        solver=self.solver,
                        maxiters=self.maxiters,
                        gamma=self.gamma,
                        include_bias=self.include_bias)

        svd.train(train_set)

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
    logging.basicConfig(filename="svd_cross_validate_grid.log",
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("*******************************************")
    logging.info("Starting run of Cross Validation")

    ks = [20, 25, 30, 40, 50]
    gammas = [0.01, 0.03, 0.1, 0.2]

    err_summaries = []

    for k in ks:
        for gamma in gammas:
            lrn = 0.005

            test_svd = TestSvd(k, lrn, gamma)
            test_svd.setUp()
            es = test_svd.test_perform_cv()
            err_summaries.append(es)

    logging.info("Finished run of Cross Validation")

    i = 0
    for k in ks:
        for gamma in gammas:
            logging.info("  error summaries:")
            logging.info("    k: %d, gamma: %d -- " % (k, gamma))
            logging.info("            mean: %e, max: %e, min: %e, sd: %e" % err_summaries[i])
            i += 1

    logging.info("*******************************************")


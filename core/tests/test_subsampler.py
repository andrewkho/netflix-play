import unittest

import os
import numpy as np

from core.flixdata import FlixData
from core.flixdata_subsampler import FlixDataSubsampler
from solvers.kFoldsCrossValidator import KFolds
from core.ratings import Ratings

flix_data_root = "../../data/arrays/"


class TestSubsampler(unittest.TestCase):

    def test_random_subsample(self):
        print ("Read flixdata")
        fd = FlixData(flix_data_root)
        print ("Generate random sample")
        self.ratings = FlixDataSubsampler.random_sample(12345, 1e5, fd)
        print ("Generat KFolds")
        self.kfolds = KFolds(self.ratings.size, 10, 12345)

        print ("Create Kfolds splits")
        test, train = self.kfolds.get(0)
        self.test_set = self.ratings.get_index_split(test)
        self.train_set = self.ratings.get_index_split(train)

        self.ratings_csr = self.ratings.get_coo_matrix().tocsr()
        self.test_csr = self.test_set.get_coo_matrix().tocsr()
        self.train_csr = self.train_set.get_coo_matrix().tocsr()

        print ("Test consistency")
        for i in range(self.test_set.size):
            t_uid = self.test_set.reverse_translate_user(self.test_set.get_coo_matrix().row[i])
            t_mid = self.test_set.reverse_translate_movie(self.test_set.get_coo_matrix().col[i])
            t_rid = self.test_set.get_coo_matrix().data[i]

            r_uidx, r_midx = self.ratings.translate(t_uid, t_mid)
            assert self.ratings_csr[r_uidx, r_midx] == t_rid, "%d, %d rating in test is not equal to ratings" % (
                t_uid, t_mid)

            try:
                train_uidx, train_midx = self.train_set.translate(t_uid, t_mid)
            except KeyError:
                continue  # It's possible the movie or rating doesn't exist in training set
            assert self.train_csr[train_uidx, train_midx] == 0, "%d, %d rating exists in train set but shouldn't!" % (
                t_uid, t_mid)

    def test_popularity_subsample(self):
        print ("Read flixdata")
        fd = FlixData(flix_data_root)
        print ("Generate popularity subsample")
        self.ratings = FlixDataSubsampler.popularity_sample(12345, 500, 50000, fd)
        print ("Generate KFolds")
        self.kfolds = KFolds(self.ratings.size, 10, 12345)

        print ("Create Kfolds splits")
        test, train = self.kfolds.get(0)
        self.test_set = self.ratings.get_index_split(test)
        self.train_set = self.ratings.get_index_split(train)

        self.ratings_csr = self.ratings.get_coo_matrix().tocsr()
        self.test_csr = self.test_set.get_coo_matrix().tocsr()
        self.train_csr = self.train_set.get_coo_matrix().tocsr()

        print ("Test consistency")
        for i in range(self.test_set.size):
            t_uid = self.test_set.reverse_translate_user(self.test_set.get_coo_matrix().row[i])
            t_mid = self.test_set.reverse_translate_movie(self.test_set.get_coo_matrix().col[i])
            t_rid = self.test_set.get_coo_matrix().data[i]

            r_uidx, r_midx = self.ratings.translate(t_uid, t_mid)
            assert self.ratings_csr[r_uidx, r_midx] == t_rid, "%d, %d rating in test is not equal to ratings" % (
                t_uid, t_mid)

            try:
                train_uidx, train_midx = self.train_set.translate(t_uid, t_mid)
            except KeyError:
                continue  # It's possible the movie or rating doesn't exist in training set
            assert self.train_csr[train_uidx, train_midx] == 0, "%d, %d rating exists in train set but shouldn't!" % (
                t_uid, t_mid)


if __name__ == '__main__':
    unittest.main()

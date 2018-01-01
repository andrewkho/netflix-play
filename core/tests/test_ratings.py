import unittest

import os
import numpy as np

import cPickle

from core.flixdata import FlixData
from solvers.kFoldsCrossValidator import KFolds
from core.ratings import Ratings

flix_data_root = "../../data/arrays/"
saved_data = "ratings_test.pkl"


class TestStringMethods(unittest.TestCase):

    def setUp(self):
        print os.getcwd()
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
            N = int(1e6)
            self.ratings = Ratings(fd.userIDsForUsers[idx[:N]],
                                   fd.movieIDs[idx[:N]],
                                   fd.userRatings[idx[:N]])

            self.kfolds = KFolds(self.ratings.size, 10, 12345)

            with open(saved_data, "wb") as f:
                cPickle.dump((self.ratings, self.kfolds), f, cPickle.HIGHEST_PROTOCOL)

        test, train = self.kfolds.get(0)
        self.test_set = self.ratings.get_index_split(test)
        self.train_set = self.ratings.get_index_split(train)

        self.ratings_csr = self.ratings.get_coo_matrix().tocsr()
        self.test_csr = self.test_set.get_coo_matrix().tocsr()
        self.train_csr = self.train_set.get_coo_matrix().tocsr()

    def test_test_set(self):
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

    def test_train_set(self):
        for i in range(self.train_set.size):
            t_uid = self.train_set.reverse_translate_user(self.train_set.get_coo_matrix().row[i])
            t_mid = self.train_set.reverse_translate_movie(self.train_set.get_coo_matrix().col[i])
            t_rid = self.train_set.get_coo_matrix().data[i]

            r_uidx, r_midx = self.ratings.translate(t_uid, t_mid)
            assert self.ratings_csr[r_uidx, r_midx] == t_rid, "%d, %d rating in test is not equal to ratings" % (
                t_uid, t_mid)

            try:
                test_uidx, test_midx = self.test_set.translate(t_uid, t_mid)
            except KeyError:
                continue  # It's possible the movie or rating doesn't exist in training set
            assert self.test_csr[test_uidx, test_midx] == 0, "%d, %d rating exists in train set but shouldn't!" % (
                t_uid, t_mid)


if __name__ == '__main__':
    unittest.main()

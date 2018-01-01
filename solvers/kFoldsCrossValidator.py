import numpy as np

from core.ratings import Ratings
from solvers.recommenderAlgorithm import RecommenderAlgorithm
from solvers.testResult import TestResult


class KFoldsCrossValidator(object):
    def __init__(self):
        pass

    def cv(self, recommender, ratings, seed, get_average, k, metric):
        # type: (RecommenderAlgorithm, Ratings, int, function, int, function) -> TestResult
        """
        Perform cross validation using "k" folds

        :param recommender: An instance of RecommenderAlgorithm
        :param ratings: An instance of Ratings
        :param seed: A seed for randomizer
        :param get_average: A function to average TestResults
        :param k: how many folds to use
        :param metric: A cost function
        :return: the result of get_average([ ... list of TestResults ... ])
        """

        kfolds = KFolds(ratings.size, k, seed)

        results = [None] * k

        for i in range(k):
            test, train = kfolds.get(i)
            train_set = ratings.get_index_split(train)
            test_set = ratings.get_index_split(test)

            recommender.train(train_set)
            results[i] = recommender.test(test_set)

        return get_average(results)


class KFolds(object):
    def __init__(self, N, k, seed):
        # type: (int, int, int) -> None
        self.k = k
        self.seed = seed
        self.N = N

        np.random.seed(seed)
        idx = np.random.choice(N, size=N, replace=False)

        self._folds = [np.array([j for j, x in enumerate(idx) if x % self.k == i], dtype=np.int32) for i in
                       range(self.k)]

    def get(self, fold):
        # type: (int) -> (np.array[bool], np.array[bool])
        bool_array = np.zeros(self.N, dtype=np.bool)
        bool_array[self._folds[fold]] = True
        return bool_array, ~bool_array

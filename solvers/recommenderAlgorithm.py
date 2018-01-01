from core.ratings import Ratings
from solvers.testResult import TestResult


class RecommenderAlgorithm(object):
    def __init__(self):
        raise RuntimeError("Can't instantiate RecommenderAlgorithm (abstract)!")

    def train(self, ratings):
        # type: (Ratings) -> None
        raise RuntimeError("Not implemented!")

    def test(self, ratings):
        # type: (Ratings) -> TestResult
        raise RuntimeError("Not implemented!")

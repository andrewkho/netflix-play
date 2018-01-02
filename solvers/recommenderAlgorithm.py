import numpy as np

from core.ratings import Ratings


class RecommenderAlgorithm(object):
    def __init__(self):
        raise RuntimeError("Can't instantiate RecommenderAlgorithm (abstract)!")

    def train(self, ratings):
        # type: (Ratings) -> None
        raise RuntimeError("Not implemented!")

    def predict(self, ratings):
        # type: (Ratings) -> np.array
        raise RuntimeError("Not implemented!")


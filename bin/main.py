### andrew's main file calling these functions

import numpy as np

from core.flixdata import FlixData
from solvers.kFoldsCrossValidator import KFolds
from core.ratings import Ratings

resample = False

file_name = "data/.subsample.npy"
pkl_name = "data/pkl/all_ratings"
ratings_name = "data/ratings.pkl"

print("  Loading Data...")
fd = FlixData("data/arrays/")
print("  Create Random subsample of full Ratings...")
np.random.seed(12345)
idx = np.random.choice(fd.userIDsForUsers.size, size=fd.userIDsForUsers.size, replace=False)
N = int(1e6)
print("  Building Ratings matrix")
ratings = Ratings(fd.userIDsForUsers[idx[:N]],
                  fd.movieIDs[idx[:N]],
                  fd.userRatings[idx[:N]])

print("  Building K-folds")
kfolds = KFolds(ratings.size, 10, 12345)

test, train = kfolds.get(0)
test_set = ratings.get_index_split(test)
train_set = ratings.get_index_split(train)

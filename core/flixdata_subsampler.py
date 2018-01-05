import numpy as np

from itertools import izip

from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from .ratings import Ratings
from .flixdata import FlixData

class FlixDataSubsampler(object):
    """
    Create subsamples of the ratings matrix
    """
    def __init__(self):
        pass

    @staticmethod
    def get_all(flixdata):
        # type: (FlixData) -> Ratings
        """
        Return a new Ratings object generated from all the entries in flixdata
        :param flixdata: the FlixData to sample from
        :return: a new ratings object
        """
        return Ratings(flixdata.userIDsForUsers,
                       flixdata.movieIDs,
                       flixdata.userRatings)

    @staticmethod
    def random_sample(seed, N, flixdata):
        # type: (int, int, FlixData) -> Ratings
        """
        Return a new Ratings object generated from a naive random
        sample of N observations from ratings
        :param seed: passed to np.random.seed
        :param N: number of output ratings. Will be coerced to int
        :param flixdata: the FlixData object to sample from
        :return: a new ratings object with N ratings
        """
        N = int(N)
        np.random.seed(seed)
        idx = np.random.choice(flixdata.userIDsForUsers.size, size=flixdata.userIDsForUsers.size, replace=False)
        return Ratings(flixdata.userIDsForUsers[idx[:N]],
                       flixdata.movieIDs[idx[:N]],
                       flixdata.userRatings[idx[:N]])

    @staticmethod
    def popularity_sample(seed, top_n_movies, top_n_users, flixdata):
        # type: (int, int, int, FlixData) -> Ratings

        top_n_movies = int(top_n_movies)
        top_n_users = int(top_n_users)

        mat = coo_matrix((flixdata.userRatings, (flixdata.userIDsForUsers, flixdata.movieIDs)),
                         shape=(flixdata.numusers, flixdata.nummovies), dtype=np.float16)

        print("getting top movies")
        mat_csc = mat.tocsc()  # type: csc_matrix
        col_sizes = mat_csc.indptr[1:] - mat_csc.indptr[:-1]  # Get number of ratings in each movie
        top_movies = set(np.argsort(col_sizes)[-top_n_movies:][::-1])

        print("getting top users")
        mat_csr = mat.tocsr()  # type: csr_matrix
        row_sizes = mat_csr.indptr[1:] - mat_csr.indptr[:-1]  # Get number of ratings in each movie
        top_users = np.argsort(col_sizes)[-top_n_users:][::-1]

        print("finding ratings for %d users, %d movies" % (top_n_users, top_n_movies))
        I = []
        J = []
        R = []
        for uid in top_users:
            st = mat_csr.indptr[uid]
            end = mat_csr.indptr[uid+1]
            for mid, rating in izip(mat_csr.indices[st:end], mat_csr.data[st:end]):
                if mid in top_movies:
                    I.append(uid)
                    J.append(mid)
                    R.append(rating)

        print("found %d ratings" % len(I))

        print("Generating Ratings matrix")
        return Ratings(np.array(I, dtype=np.int32),
                       np.array(J, dtype=np.int32),
                       np.array(R, dtype=np.int32))




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
    def random_sample(flixdata, seed, N):
        # type: (FlixData, int, int) -> Ratings
        """
        Return a new Ratings object generated from a naive random
        sample of N observations from ratings

        :param flixdata: the FlixData object to sample from
        :param seed: passed to np.random.seed
        :param N: number of output ratings. Will be coerced to int
        :return: a new ratings object with N ratings
        """
        N = int(N)
        np.random.seed(seed)
        idx = np.random.choice(flixdata.userIDsForUsers.size, size=flixdata.userIDsForUsers.size, replace=False)
        return Ratings(flixdata.userIDsForUsers[idx[:N]],
                       flixdata.movieIDs[idx[:N]],
                       flixdata.userRatings[idx[:N]])

    @staticmethod
    def random_sample_movies(flixdata, seed, N, M, minratings=9):
        # type: (FlixData, int, int, int) -> Ratings
        """
        Return a new Ratings object with randomly chosen movies.
        Returns random sample of N users who rated at least *minratings* of these movies.

        :param flixdata: the FlixData object to sample from
        :param seed: passed to np.random.seed
        :param N: Number of unique users
        :param M: Number of unique movies
        :param minratings: filter out users who don't have at least this many ratings
        :return: a new ratings object with M movies and users who rated them
        """
        M = int(M)
        N = int(N)

        print "  random sample of %d users, %d movies, minratings: %d" % (N, M, minratings)
        print "  building csc_matrix"
        all_csc = csc_matrix((flixdata.userRatings, (flixdata.userIDsForUsers, flixdata.movieIDs)),
                             shape=(flixdata.numusers, flixdata.nummovies))
        all_csc.sort_indices()

        #user_counts = np.bincount(flixdata.userIDsForUsers)
        #np.random.seed(seed)
        #users = np.random.choice(np.where(user_counts >= minratings)[0], size=nusers, replace=False)
        movies = np.random.choice(flixdata.nummovies, size=M, replace=False)
        movies.sort()
        users = []

        print "  building user set"
        for movie in movies:
            users.extend(all_csc.indices[all_csc.indptr[movie]:all_csc.indptr[movie+1]])

        print "  counting user set"
        usercounts = np.bincount(users)
        users = np.where(usercounts > minratings)[0]
        users = np.random.choice(users, size=N, replace=False)

        print "  Building index set"
        user_idxs = np.isin(flixdata.userIDsForUsers, users)
        movie_idxs = np.isin(flixdata.movieIDs, movies)
        idxs = np.logical_and(user_idxs, movie_idxs)

        print "  building new Ratings object"
        return Ratings(flixdata.userIDsForUsers[idxs],
                       flixdata.movieIDs[idxs],
                       flixdata.userRatings[idxs])

    @staticmethod
    def random_sample_users(flixdata, seed, N, nusers, minratings=9):
        # type: (FlixData, int, int, int, int) -> Ratings
        """
        Return a new Ratings object with N ratings and nusers,
        guaranteeing that each user will have at least minratings ratings

        :param flixdata: the FlixData object to sample from
        :param seed: passed to np.random.seed
        :param N: Number of output ratings
        :param nusers: Number of output users
        :param minratings: filter out users who don't have at least this many ratings
        :return: a new ratings object with N users and all of their ratings
        """
        N = int(N)
        nusers = int(nusers)

        all_csr = csr_matrix((flixdata.userRatings, (flixdata.userIDsForUsers, flixdata.movieIDs)),
                             shape=(flixdata.numusers, flixdata.nummovies))
        all_csr.sort_indices()

        user_counts = np.bincount(flixdata.userIDsForUsers)
        np.random.seed(seed)
        users = np.random.choice(np.where(user_counts >= minratings)[0], size=nusers, replace=False)
        users.sort()
        idxs = []
        ## Get first minratings for each user
        for user in users:
            try:
                uindexes = np.random.choice(np.arange(all_csr.indptr[user],all_csr.indptr[user+1]), size=minratings, replace=False)
            except ValueError as e:
                print "bad user: %d, %d ratings" % (user, all_csr.indptr[user+1]-all_csr.indptr[user])
                raise e
            idxs.extend(uindexes)

        remain = N - len(idxs)
        if remain < 0:
            print "Warning: we already exceeded N (it should be greater than nusers*minratings)"

        all_idxs = np.isin(flixdata.userIDsForUsers, users)
        chosen_idxs = np.isin(np.arange(flixdata.numratings), np.array(idxs))
        remain_idxs = np.where(np.logical_xor(all_idxs, chosen_idxs))[0]

        sample_remain = np.random.choice(remain_idxs, size=remain, replace=False)
        idxs.extend(sample_remain)
        dupe_check = set(idxs)
        if len(dupe_check) != len(idxs):
            raise RuntimeError("generated indexes has non-unique values! orig: %d, unique %d" % (len(idxs), len(dupe_check)))
        idxs = np.array(idxs)
        idxs.sort()

        return Ratings(flixdata.userIDsForUsers[idxs],
                       flixdata.movieIDs[idxs],
                       flixdata.userRatings[idxs])

    @staticmethod
    def popularity_sample(flixdata, seed, top_n_movies, top_n_users):
        # type: (FlixData, int, int, int) -> Ratings
        """
        a biased subsample of flixdata.
        selects the top_n_movies and top_n_users (their intersection).
        for small n's, the dataset will be quite dense

        :param flixdata:
        :param seed:
        :param top_n_movies:
        :param top_n_users:
        :return:
        """

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
        row_sizes = mat_csr.indptr[1:] - mat_csr.indptr[:-1]  # Get number of ratings by each user
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




import numpy as np
import scipy.sparse

from itertools import izip
from enum import Enum

from core.ratings import Ratings
from solvers.recommenderAlgorithm import RecommenderAlgorithm

import solvers.kmeans.kdtree as kdtree


Distances = Enum("Distances", "manhatten euclidean")


class KMeansSolver(RecommenderAlgorithm):
    """
    A recommender engine which uses k-means clustering to generate nearest neighbours
    """

    def __init__(self, k, dist=Distances.manhatten):
        # type: (int, Distances) -> None
        self.k = k  # type: int
        self.dist = dist  # type: Distances
        self._ratings = None  # type: Ratings
        self._cov = None  # type: scipy.sparse.csr_matrix
        self._means = None
        self._cluster = None
        self._kdtree = None

    def train(self, ratings, seed = None):
        # type: (Ratings) -> None
        self._ratings = ratings

        self._kmeans_cluster(seed)

    def _kmeans_cluster(self, seed = None):
        if seed is not None:
            np.random.seed(int(seed))

        print("Generating random centroids")
        ## Generate k random starting points
        self._means = [5 * np.random.random_sample(self._ratings.num_cols) for _ in range(self.k)]
        self._cluster = np.zeros(self._ratings.num_rows, dtype=np.int32)

        csr_mat = self._ratings.get_csr_matrix()
        curriter = 0
        while True:
            curriter += 1
            delta = 0
            # Update clusters
            print("Running iteration %d-reassign clusters" % curriter)
            for uid in range(self._ratings.num_rows):
                uindices = csr_mat.indices[csr_mat.indptr[uid]:csr_mat.indptr[uid+1]]
                udata = csr_mat.data[csr_mat.indptr[uid]:csr_mat.indptr[uid+1]]
                cur_cluster = self._cluster[uid]

                best_cluster = -1
                best_dist = np.finfo(np.float64).max
                for cluster in range(self.k):
                    mn = self._means[cluster]
                    dist = 0
                    for ind, data in izip(uindices, udata):
                        dist += abs(data - mn[ind])
                    if np.isnan(dist):
                        print "Distance is nan!!"
                    dist /= uindices.size
                    if dist < best_dist:
                        best_dist = dist
                        best_cluster = cluster
                if best_cluster != cur_cluster:
                    delta += 1
                    self._cluster[uid] = best_cluster

            if delta == 0:
                break

            # Update means
            print("Running iteration %d-update means (%d changes)" % (curriter, delta))
            self._means = [np.zeros(self._ratings.num_cols, dtype=np.float64) for _ in range(self.k)]
            counter = np.zeros(self.k, dtype=np.int32)
            for uid in range(self._ratings.num_rows):
                cluster = self._cluster[uid]
                counter[cluster] += 1

                uindices = csr_mat.indices[csr_mat.indptr[uid]:csr_mat.indptr[uid+1]]
                udata = csr_mat.data[csr_mat.indptr[uid]:csr_mat.indptr[uid+1]]
                for ind, data in izip(uindices, udata):
                    self._means[cluster][ind] += data

            for cluster in range(self.k):
                if counter[cluster] > 0:
                    self._means[cluster] /= counter[cluster]

    def predict(self, testratings):
        # type: (Ratings) -> np.array
        n = testratings.get_coo_matrix().nnz
        print("nnz testratigns: " + str(n))
        pred = np.zeros(n)
        ##dok_matrix = self._ratings.get_coo_matrix().todok()  ## DoK matrix is faster but different MSE (!@?!?!?!)
        dok_matrix = self._ratings.get_coo_matrix().copy().todok()

        for i in range(n):
            uidx = testratings.get_coo_matrix().row[i]
            midx = testratings.get_coo_matrix().col[i]
            uid, mid = testratings.reverse_translate(uidx, midx)
            pred[i] = self.predict_single(dok_matrix, uid, mid)

        print("len(pred): " + str(len(pred)))

        return pred

    def predict_single(self, dok_matrix, uid, mid):
        # type: (int, int) -> float

        try:
            user_idx, movie_idx = self._ratings.translate(uid, mid)
        except KeyError:
            return None

        uindices = self._cov.indices[self._cov.indptr[user_idx]:self._cov.indptr[user_idx+1]]
        ucovar = self._cov.data[self._cov.indptr[user_idx]:self._cov.indptr[user_idx+1]]

        neighbours = {}

        counter = 0
        sorted_neighbours = ucovar.argsort()[::-1]
        for nei_idx in sorted_neighbours:
            neiidx = uindices[nei_idx]
            rating = dok_matrix[neiidx, movie_idx]
            if rating == 0 or rating is None: # Neighbour hasn't rated movie
                continue
            neighbours[neiidx] = rating
            counter += 1
            if counter >= self.k:
                break

        #if counter < self.k:
        #    print('Warning: not enough neighbours for k, counter: ' + str(counter))

        # Just try an average
        tot = 0.
        for neiId, rating in neighbours.items():
            tot += rating

        if counter == 0:
            return None
        else:
            return tot / counter




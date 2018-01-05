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

    TODO: Use some datastructure to speed up the nearest centroid search times!
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
        self._means = 5 * np.random.random_sample((self.k, self._ratings.num_cols))
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
                    valid_indices = np.isin(uindices, np.where(~np.isnan(self._means[cluster, :])))
                    if valid_indices.sum() == 0:
                        continue
                    dist = np.linalg.norm(udata[valid_indices] - self._means[cluster, uindices[valid_indices]], ord=1) / valid_indices.sum()
                    if dist < best_dist:
                        best_dist = dist
                        best_cluster = cluster
                if best_cluster == -1:
                    print "no best cluster!"
                if best_cluster != cur_cluster:
                    delta += 1
                    self._cluster[uid] = best_cluster

            if delta == 0:
                print("Done! after %d iters, delta=%d" % (curriter, delta))
                break

            # Update means
            print("Running iteration %d-update means (%d changes)" % (curriter, delta))
            self._means = np.zeros((self.k, self._ratings.num_cols), dtype=np.float64)
            counter = np.zeros(self._means.shape, dtype=np.int32)
            for uid in range(self._ratings.num_rows):
                cluster = self._cluster[uid]

                uindices = csr_mat.indices[csr_mat.indptr[uid]:csr_mat.indptr[uid+1]]
                udata = csr_mat.data[csr_mat.indptr[uid]:csr_mat.indptr[uid+1]]

                self._means[cluster, uindices] += udata
                counter[cluster, uindices] += 1

            self._means /= counter

    def predict(self, testratings):
        # type: (Ratings) -> np.array
        n = testratings.get_coo_matrix().nnz
        print("predicting %d testratigns: " % n)
        pred = np.zeros(n)
        csr_mat = self._ratings.get_coo_matrix().tocsr()

        for i in range(n):
            uidx = testratings.get_coo_matrix().row[i]
            midx = testratings.get_coo_matrix().col[i]

            uid, mid = testratings.reverse_translate(uidx, midx)
            pred[i] = self.predict_single(uid, mid, csr_mat)

        print("len(pred): " + str(len(pred)))

        return pred

    def predict_single(self, uid, mid, csr_mat):
        # type: (int, int, scipy.sparse.csr_matrix) -> float

        try:
            user_idx, movie_idx = self._ratings.translate(uid, mid)
        except KeyError:
            return None

        uindices = csr_mat.indices[csr_mat.indptr[user_idx]:csr_mat.indptr[user_idx+1]]
        udata = csr_mat.data[csr_mat.indptr[user_idx]:csr_mat.indptr[user_idx+1]]

        best_cluster = -1
        best_dist = np.finfo(np.float64).max
        for cluster in range(self.k):
            mn = self._means[cluster, :]
            if valid_indices.sum() == 0:
                continue
            valid_indices = np.isin(uindices, np.where(~np.isnan(mn)))
            dist = np.linalg.norm(udata[valid_indices] - self._means[cluster, uindices[valid_indices]], ord=1) / valid_indices.sum()

            if dist < best_dist:
                best_dist = dist
                best_cluster = cluster
        if best_cluster == -1:
            print "No best cluster!"
            return None
        ## For now just use the cluster mean
        return self._means[best_cluster, movie_idx]

        neighbours = {}

        counter = 0
        sorted_neighbours = ucovar.argsort()[::-1]
        for nei_idx in sorted_neighbours:
            neiidx = uindices[nei_idx]
            rating = csr_mat[neiidx, movie_idx]
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




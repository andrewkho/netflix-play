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
                print("Done! after %d iters, delta=%d" % (curriter, delta))
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
        print("predicting %d testratigns: " % n)
        pred = np.zeros(n)
        csr_matrix = self._ratings.get_coo_matrix().tocsr()

        for i in range(n):
            uidx = testratings.get_coo_matrix().row[i]
            midx = testratings.get_coo_matrix().col[i]
            uindices = csr_matrix.indices[csr_matrix.indptr[uidx]:csr_matrix.indptr[uidx+1]]
            udims = [testratings.reverse_translate_movie(_) for _ in uindices]
            udata = csr_matrix.data[csr_matrix.indptr[uidx]:csr_matrix.indptr[uidx+1]]

            uid, mid = testratings.reverse_translate(uidx, midx)
            pred[i] = self.predict_single(uid, mid, udims, udata)

        print("len(pred): " + str(len(pred)))

        return pred

    def predict_single(self, uid, mid, udims, udata):
        # type: (int, int) -> float

        try:
            user_idx, movie_idx = self._ratings.translate(uid, mid)
        except KeyError:
            return None

        local_dims = list()
        local_data = list()
        for i, dim in enumerate(udims):
            try:
                local_dims.append(self._ratings.translate_movie(dim))
                local_data.append(udata[i])
            except KeyError:
                continue

        best_cluster = -1
        best_dist = np.finfo(np.float64).max
        for cluster in range(self.k):
            mn = self._means[cluster]
            dist = 0
            for ind, data in izip(local_dims, local_data):
                dist += abs(data - mn[ind])
            if np.isnan(dist):
                print "Distance is nan!!"
            dist /= len(local_dims)
            if dist < best_dist:
                best_dist = dist
                best_cluster = cluster

        ## For now just use the cluster mean
        return self._means[best_cluster][movie_idx]

        neighbours = {}

        counter = 0
        sorted_neighbours = ucovar.argsort()[::-1]
        for nei_idx in sorted_neighbours:
            neiidx = uindices[nei_idx]
            rating = csr_matrix[neiidx, movie_idx]
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




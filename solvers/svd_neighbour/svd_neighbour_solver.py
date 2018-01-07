import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from sklearn.cluster import KMeans

from itertools import izip

from core.ratings import Ratings
from solvers.recommenderAlgorithm import RecommenderAlgorithm

from util.mle_cov import mle_cov
from util.incomplete_projection import incomplete_projection

from solvers.svd_neighbour.svd_neighbour_predict import cy_svdn_predict, cy_svdn_predict_single


class SvdNeighbourSolver(RecommenderAlgorithm):
    """
    Generates a SVD representation of the ratings matrix
    """

    def __init__(self, svd_k, knn_k, cluster_k):
        # type: (int) -> None
        self.svd_k = svd_k  # type: int
        self.knn_k = knn_k  # type: int
        self.cluster_k = cluster_k  # type: int
        self._ratings = None  # type: Ratings
        self._cov = None  # type: scipy.sparse.csr_matrix

        self._svd_u = None # type: np.ndarray
        self._svd_s = None # type: np.ndarray
        self._svd_vt = None # type: np.ndarray
        self._eigenratings = None # type:np.ndarray

    def train(self, ratings, seed = None):
        # type: (Ratings) -> None
        self._ratings = ratings

        ## compute and store data needed for predictions
        print "  mle covariance"
        self._cov = mle_cov(self._ratings.get_coo_matrix())
        print "  %d top svd" % self.svd_k
        self._svd_u, self._svd_s, self._svd_vt = scipy.sparse.linalg.svds(self._cov, self.svd_k)
        print "  eigenratings"
        self._eigenratings = incomplete_projection(self._ratings.get_csr_matrix(), self._svd_u)

        print "  clustering into %d clusters" % self.cluster_k
        ## Since eigenratings is dense, we can use off the shelf KMeans solver from sklearn
        kmeans_estimator = KMeans(n_clusters=self.cluster_k, n_init=10, random_state=seed)
        kmeans_estimator.fit(self._eigenratings)

        print "  generate cluster mappings"
        mapped_idxs = np.zeros(shape=kmeans_estimator.labels_.shape, dtype=np.int32)
        labels = kmeans_estimator.labels_
        for cluster in range(self.cluster_k):
            cluster_bools = labels == cluster
            mapped_idxs = mapped_idxs + (np.cumsum(cluster_bools) - 1)*cluster_bools
        self._cluster_labels = np.vstack((labels, mapped_idxs)).transpose().astype(dtype=np.int32)

        print "  compute ratings correlation in eigen space, for each cluster"
        self._correlation = list()
        for cluster in range(self.cluster_k):
            self._correlation.append(np.corrcoef(self._eigenratings[labels == cluster]))
        #self._correlation = np.corrcoef(self._eigenratings)

        print "  Nearest neighbour sort"
        #self._neighbours = np.argsort(self._correlation, axis=1)[:, ::-1][:, 1:]
        self._neighbours = list()
        for cluster in range(self.cluster_k):
            if self._correlation[cluster].ndim == 1: ## For empty clusters
                self._neighbours.append(None)
            self._neighbours.append(np.argsort(self._correlation[cluster], axis=1)[:, ::-1][:, 1:])


    def predict(self, testratings):
        # type: (Ratings) -> np.array
        if False:
            return self.predict_cy(testratings)

        n = testratings.get_coo_matrix().nnz
        pred = np.zeros(n)
        rat_mat = self._ratings.get_coo_matrix().tolil()
        csc_mat = self._ratings.get_csc_matrix()

        for i in range(n):
            uidx = testratings.get_coo_matrix().row[i]
            midx = testratings.get_coo_matrix().col[i]

            uid, mid = testratings.reverse_translate(uidx, midx)
            pred[i] = self.predict_single(uid, mid, csc_mat)

        return pred

    def predict_cy(self, testratings):
        # type: (Ratings) -> np.array
        n = testratings.get_coo_matrix().nnz
        uidxs = np.zeros(n, dtype=np.int32)
        midxs = np.zeros(n, dtype=np.int32)
        preds = np.zeros(n, dtype=np.float64)
        for i in range(n):
            uidx = testratings.get_coo_matrix().row[i]
            midx = testratings.get_coo_matrix().col[i]

            uid, mid = testratings.reverse_translate(uidx, midx)
            try:
                this_user_idx, this_movie_idx = self._ratings.translate(uid, mid)
            except KeyError:
                this_user_idx = None
                this_movie_idx = None
            uidxs[i] = this_user_idx
            midxs[i] = this_movie_idx

        csc_mat = self._ratings.get_csc_matrix()

        cy_svdn_predict(preds, uidxs, midxs, self.knn_k, self._correlation, self._neighbours,
                        csc_mat.indptr, csc_mat.indices, csc_mat.data)

        return preds


    def predict_single(self, uid, mid, csc_mat):
        # type: (int, int, scipy.sparse.csc_matrix) -> float

        try:
            user_idx, movie_idx = self._ratings.translate(uid, mid)
        except KeyError:
            return None

        cluster = self._cluster_labels[user_idx, 0]
        mapped_user_idx = self._cluster_labels[user_idx, 1]

        potentials = csc_mat.indices[csc_mat.indptr[movie_idx]:csc_mat.indptr[movie_idx+1]]
        mapped_potentials = self._cluster_labels[:, 1][potentials]
        potential_ratings = csc_mat.data[csc_mat.indptr[movie_idx]:csc_mat.indptr[movie_idx+1]]

        #return cy_svdn_predict_single(user_idx, self._correlation, self.knn_k, self._neighbours[user_idx, :], potentials, potential_ratings)
        return cy_svdn_predict_single(mapped_user_idx, self._correlation[cluster], self.knn_k,
                                      self._neighbours[cluster][mapped_user_idx, :], mapped_potentials, potential_ratings)



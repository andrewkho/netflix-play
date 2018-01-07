import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from itertools import izip

from core.ratings import Ratings
from solvers.recommenderAlgorithm import RecommenderAlgorithm

from util.mle_cov import mle_cov
from util.incomplete_projection import incomplete_projection

class SvdNeighbourSolver(RecommenderAlgorithm):
    """
    Generates a SVD representation of the ratings matrix
    """

    def __init__(self, svd_k, knn_k):
        # type: (int) -> None
        self.svd_k = svd_k  # type: int
        self.knn_k = knn_k  # type: int
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
        #self._cov = self.mle_cov(self._ratings.get_coo_matrix())
        print "  %d top svd" % self.svd_k
        self._svd_u, self._svd_s, self._svd_vt = scipy.sparse.linalg.svds(self._cov, self.svd_k)
        print "  eigenratings"
        self._eigenratings = incomplete_projection(self._ratings.get_csr_matrix(), self._svd_u)
        print "  ratings correlation in eigen space"
        self._correlation = np.corrcoef(self._eigenratings)
        print "  Nearest neighbour sort"
        self._neighbours = np.argsort(self._correlation, axis=1)[:, ::-1][:, 1:]

    # def mle_cov(self, ratings_mat):
    #     """
    #     An attempt at a numpy solution to calculating MLE covariance (column-wise)
    #     :param ratings_mat: with size NxM
    #     :return: sparse matrix with size MxM (coo_matrix)
    #     """
    #     csc_mat = ratings_mat.tocsc()
    #     csc_mat.sort_indices()
    #     csc_indices = csc_mat.indices
    #     csc_indptr = csc_mat.indptr
    #     csc_data = csc_mat.data
    #
    #     for item_col in range(ratings_mat.shape[1]):
    #         item_rows = csc_indices[csc_indptr[item_col]:csc_indptr[item_col + 1]]
    #         item_rats = csc_data[csc_indptr[item_col]:csc_indptr[item_col + 1]]
    #         #item_rats_dict = rating_dicts[item_col]
    #         #neighbours = cppset[int]()
    #
    #         #for i in range(item_rows.shape[0]):
    #         #    item_row = item_rows[i]
    #         #    tmp = csr_indices[csr_indptr[item_row]:csr_indptr[item_row+1]]
    #         #    for j in range(tmp.shape[0]):
    #         #        neighbours.insert(tmp[j])
    #         for other_col in range(ratings_mat.shape[1]):
    #             other_rows = csc_indices[csc_indptr[other_col]:csc_indptr[other_col + 1]]
    #             other_rats = csc_data[csc_indptr[other_col]:csc_indptr[other_col + 1]]
    #             other_rats_dict = rating_dicts[other_col]
    #
    #             n = min(item_rows.shape[0], other_rows.shape[0])
    #             #ixn = <int*> malloc(n * sizeof(int))
    #             #ixn_size = cyintersect1d(ixn, item_rows, other_rows)
    #
    #             item_mean = item_means[item_col]
    #             other_mean = item_means[other_col]
    #             cov = 0
    #             denom_u = 0
    #             denom_o = 0
    #             for i in range(ixn_size):
    #                 user = ixn[i]
    #                 udiff = item_rats_dict[user] - item_mean
    #                 odiff = other_rats_dict[user] - other_mean
    #                 cov += udiff * odiff
    #
    #             if ixn_size == 0:
    #                 cov = 0
    #             else:
    #                 cov /= ixn_size
    #
    #             out_row[out_counter] = item_col
    #             out_col[out_counter] = other_col
    #             out_data[out_counter] = cov
    #             out_counter += 1


    def predict(self, testratings):
        # type: (Ratings) -> np.array
        n = testratings.get_coo_matrix().nnz
        print("predicting %d testratigns: " % n)
        pred = np.zeros(n)
        lil_mat = self._ratings.get_lil_matrix()

        for i in range(n):
            uidx = testratings.get_coo_matrix().row[i]
            midx = testratings.get_coo_matrix().col[i]

            uid, mid = testratings.reverse_translate(uidx, midx)
            pred[i] = self.predict_single(uid, mid, lil_mat)

        print("len(pred): " + str(len(pred)))

        return pred

    def predict_single(self, uid, mid, lil_mat):
        # type: (int, int, scipy.sparse.lil_matrix) -> float

        try:
            user_idx, movie_idx = self._ratings.translate(uid, mid)
        except KeyError:
            return None

        neighbours = self._neighbours[user_idx, :]
        counter = 0
        tot = 0.
        denom = 0.
        for nei_idx in neighbours:
            rating = lil_mat[nei_idx, movie_idx]
            if rating == 0:
                continue
            counter += 1
            tot += rating * self._correlation[user_idx, nei_idx]
            denom += self._correlation[user_idx, nei_idx]
            if counter >= self.knn_k:
                break

        #if counter == self.knn_k:
        #    print "found k neighbours!"
        #    #print "Warning, counter less than knn_k (%d, %d)" % (counter, self.knn_k)

        tot /= denom
        return tot






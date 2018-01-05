import numpy as np

from scipy.sparse import coo_matrix, csr_matrix, lil_matrix

class Ratings(object):
    """
    <Immutable> Sparse representation of user/movie ratings
    underlying representation is a coo_matrix
    """

    def __init__(self, uids, mids, ratings):
        # type: (np.array, np.array, np.array) -> None
        """
        Constructs a Ratings object from a list of uids, mids, ratings (ala coo_matrix).
        Any duplicate entries are *summed*. length of uids, mids, ratings must be identical

        :param uids: user ids (e.g. row indices)
        :param mids: mids (e.g. column indices)
        :param ratings: (movie ratings)
        """

        # Build lookup indexes for ratings
        self._uid_to_idx = dict()
        self._mid_to_idx = dict()

        self._idx_to_uid = []
        self._idx_to_mid = []

        I = -np.ones(shape=uids.shape)
        J = -np.ones(shape=mids.shape)
        R = -np.ones(shape=ratings.shape)
        i = 0
        j = 0
        for counter in range(uids.size):
            # if counter % 10000 == 0:
            #    print "progress: " + str(counter)
            uid = uids[counter]
            mid = mids[counter]
            rating = ratings[counter]
            if uid not in self._uid_to_idx:
                self._uid_to_idx[uid] = i
                self._idx_to_uid.append(uid)
                i += 1
            if mid not in self._mid_to_idx:
                self._mid_to_idx[mid] = j
                self._idx_to_mid.append(mid)
                j += 1

            I[counter] = self._uid_to_idx[uid]
            J[counter] = self._mid_to_idx[mid]
            R[counter] = rating

        self._idx_to_uid = np.array(self._idx_to_uid)
        self._idx_to_mid = np.array(self._idx_to_mid)

        self._data = coo_matrix((R, (I, J)), shape=(i, j))
        self._data.sum_duplicates()

    def get_simple_split(self, k, index):
        # type: (int, int) -> (Ratings, Ratings)
        n = self._data.size
        size = n / k

        if index < k - 1:
            test_I = self._data.row[size * index:size * (index + 1)]
            test_J = self._data.col[size * index:size * (index + 1)]
            test_R = self._data.data[size * index:size * (index + 1)]

            train_I = np.append(self._data.row[0:size * index], self._data.row[size * (index + 1):])
            train_J = np.append(self._data.col[0:size * index], self._data.col[size * (index + 1):])
            train_R = np.append(self._data.data[0:size * index], self._data.data[size * (index + 1):])
        else:
            test_I = self._data.row[size * index:]
            test_J = self._data.col[size * index:]
            test_R = self._data.data[size * index:]

            train_I = self._data.row[0:size * index]
            train_J = self._data.col[0:size * index]
            train_R = self._data.data[0:size * index]

        return Ratings(test_I, test_J, test_R), Ratings(train_I, train_J, train_R)

    def get_index_split(self, indexes):
        # type: (np.array[int]) -> Ratings
        I = self._idx_to_uid[self._data.row[indexes]]
        J = self._idx_to_mid[self._data.col[indexes]]
        R = self._data.data[indexes]

        return Ratings(I, J, R)

    def translate(self, uid, mid):
        # type: (int, int) -> (int, int)
        """Translate a global uid, mid combo to indexes for *this* Ratings matrix
        """
        return self._uid_to_idx[uid], self._mid_to_idx[mid]

    def reverse_translate(self, uidx, midx):
        # type: (int, int) -> (int, int)
        """Translate a uidx, midx for this matrix to the global user Id, movie Id
        """
        return self._idx_to_uid[uidx], self._idx_to_mid[midx]

    def translate_user(self, uid):
        # type: (int) -> int
        """Translate a global uid to index for *this* Ratings matrix
        """
        return self._uid_to_idx[uid]

    def reverse_translate_user(self, uidx):
        # type: (int) -> int
        """Translate a uidx for this matrix to the global user Id
        """
        return self._idx_to_uid[uidx]

    def translate_movie(self, mid):
        # type: (int) -> int
        """Translate a global mid to index for *this* Ratings matrix
        """
        return self._mid_to_idx[mid]

    def reverse_translate_movie(self, midx):
        # type: (int) -> int
        """Translate a midx for this matrix to the global movie Id
        """
        return self._idx_to_mid[midx]

    def get_coo_matrix(self):
        # type: () -> coo_matrix
        return self._data

    def get_lil_matrix(self):
        # type: () -> lil_matrix
        return self._data.tolil()

    def get_csr_matrix(self):
        # type: () -> csr_matrix
        return self._data.tocsr()

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def num_rows(self):
        return self._idx_to_uid.size

    @property
    def num_cols(self):
        return self._idx_to_mid.size

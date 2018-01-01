# User-wise covariance matrix

def rating_cov(ratings):
    for user_idx in range(ratings.shape()[0]):
        for other_idx in range(ratings.shape()[0]):
            userId = ratings.get_idx_user(user_idx)
            otherId = ratings.get_idx_user(other_idx)
            cov = get_cov(ratings, userId, otherId)
            self._cov[user_idx, other_idx] = cov
            self._cov[other_idx, user_idx] = cov


def get_cov(ratings, userId, otherId):
    user_row = ratings.get_user_row(userId)
    other_row = ratings.get_user_row(otherId)

    for mid in user_row:
        user_mean += ratings.get(userId, mid)
    for mid in other_row:
        user_mean += ratings.get(otherId, mid)

    cov = 0
    n = 0
    for mid in set(user_row).intersection(other_row):
        cov += (ratings.get(userId, mid) - user_mean) * (ratings.get(otherId, mid) - other_mean)
        n += 1

    if n == 0:
        return 0
    elif n == 1:
        return cov
    else:
        return cov / n - 1

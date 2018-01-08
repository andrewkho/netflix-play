#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

from libc.math cimport fabs
from libc.stdio cimport printf

from cython.parallel import prange

import numpy as np
cimport numpy as np

def svd_train_sgd(double[:,:] left, double[:,:] right, float[:] resid, int[:] rand_order,
                      int[:] uids, int[:] mids, double[:] ratings, double rate, double eps,
                      int maxiters, double gamma, int include_bias):
    return _svd_train_sgd(left, right, resid, rand_order, uids, mids, ratings, rate, eps, maxiters, gamma, include_bias)

cdef float _svd_train_sgd(double[:,:] left, double[:,:] right, float[:] resid, int[:] rand_order,
                             int[:] uids, int[:] mids, double[:] ratings, double rate, double eps,
                             int maxiters, double gamma, int include_bias) nogil:

    cdef double max_change, dleft, dright, dresid
    cdef double rat, yhat, err, total_err, old_resid
    cdef int uid, mid, ob, iter, k_, k, _

    old_resid = 0
    max_change = eps + 1
    for iter in range(maxiters):
        max_change = 0
        total_err = 0
        for _ in range(uids.shape[0]):
            ob = rand_order[_]
            uid = uids[ob]
            mid = mids[ob]
            rat = ratings[ob]

            yhat = 0
            for k_ in prange(left.shape[1]):
                yhat += left[uid, k_] * right[mid, k_]
            err = rat - yhat
            total_err += err*err

            if include_bias:
                k = 0
                dright = err * left[uid, k] - gamma * right[mid, k]
                right[mid, k] += rate * dright

                k = 1
                dleft = err * right[mid, k] - gamma * left[uid, k]  ## gamma controls regularization
                left[uid, k] += rate * dleft

                for k in prange(2, left.shape[1]):
                    dleft = err * right[mid, k] - gamma * left[uid, k]  ## gamma controls regularization
                    dright = err * left[uid, k] - gamma * right[mid, k]
                    left[uid, k] += rate * dleft
                    right[mid, k] += rate * dright

            else:
                for k in prange(left.shape[1]):
                    dleft = err * right[mid, k] - gamma * left[uid, k]  ## gamma controls regularization
                    dright = err * left[uid, k] - gamma * right[mid, k]
                    left[uid, k] += rate * dleft
                    right[mid, k] += rate * dright

                    # if rate*rate*dleft*dleft > max_change:
                    #     max_change = rate*rate*dleft*dleft
                    # if rate*rate*dright*dright > max_change:
                    #     max_change = rate*rate*dright*dright

        if iter % 20 == 0:
            printf("  resid: %e\n", total_err)
            if fabs(old_resid - total_err)/total_err < eps:
                break
            old_resid = total_err

        if iter == maxiters:
            printf("  Reached max iters!!!\n")
    return iter




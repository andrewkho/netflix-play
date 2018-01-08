#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

from libc.math cimport abs

import numpy as np
cimport numpy as np

def svd_train_sgd(double[:,:] left, double[:,:] right, float[:] resid, int[:] rand_order,
                      int[:] uids, int[:] mids, double[:] ratings, double rate, double eps,
                      int maxiters, double gamma, int include_bias):
    return _svd_train_sgd(left, right, resid, rand_order, uids, mids, ratings, rate, eps, maxiters, gamma, include_bias)

cdef float _svd_train_sgd(double[:,:] left, double[:,:] right, float[:] resid, int[:] rand_order,
                             int[:] uids, int[:] mids, double[:] ratings, double rate, double eps,
                             int maxiters, double gamma, int include_bias):

    cdef double max_change, dleft, dright, dresid
    cdef double rat, yhat, err, total_err, old_resid
    cdef int uid, mid, ob, iter, k

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
            for k_ in range(left.shape[1]):
                yhat += left[uid, k_] * right[mid, k_]
            err = rat - yhat
            total_err += err*err

            for k in range(left.shape[1]):
                dleft = err * right[mid, k] - gamma * left[uid, k]  ## gamma controls regularization
                dright = err * left[uid, k] - gamma * right[mid, k]
                if include_bias and k == 0:
                    dleft = 0
                if include_bias and k == 1:
                    dright = 0
                left[uid, k] += rate * dleft
                right[mid, k] += rate * dright

                if rate*rate*dleft*dleft > max_change:
                    max_change = rate*rate*dleft*dleft
                if rate*rate*dright*dright > max_change:
                    max_change = rate*rate*dright*dright

        if iter % 20 == 0:
            print "  max_change %e, eps: %e, resid: %e" % (max_change, eps, total_err)
        if abs(old_resid - total_err)/total_err < eps:
            break
        old_resid = total_err
        if iter == maxiters:
            print "  Reached max iters!!!"

    return iter




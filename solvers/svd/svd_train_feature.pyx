#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

from libc.math cimport abs

import numpy as np
cimport numpy as np

def svd_train_feature(double[:,:] left, double[:,:] right, float[:] resid, int k,
                      int[:] uids, int[:] mids, double[:] ratings, double rate, double eps,
                      int maxiters, double gamma, int ignore_left, int ignore_right):
    _svd_train_feature(left, right, resid, k, uids, mids, ratings, rate, eps, maxiters, gamma, ignore_left, ignore_right)

cdef void _svd_train_feature(double[:,:] left, double[:,:] right, float[:] resid, int k,
                             int[:] uids, int[:] mids, double[:] ratings, double rate, double eps,
                             int maxiters, double gamma, int ignore_left, int ignore_right):

    cdef double max_change, dleft, dright, dresid
    cdef double rat, yhat, err, total_err, old_resid
    cdef int uid, mid, ob, iter

    old_resid = 0
    max_change = eps + 1
    for iter in range(maxiters):
        max_change = 0
        total_err = 0
        for ob in range(uids.shape[0]):
            uid = uids[ob]
            mid = mids[ob]
            rat = ratings[ob]

            #yhat = 0
            #for k_ in range(left.shape[1]):
            #    yhat += left[uid, k_] * right[mid, k_]
            #err = rat - yhat

            err = resid[ob] - left[uid,k] * right[mid,k]
            total_err += err*err

            dleft = err * right[mid, k] - gamma * left[uid, k]  ## gamma controls regularization
            dright = err * left[uid, k] - gamma * right[mid, k]
            if ignore_left:
                dleft = 0
            if ignore_right:
                dright = 0
            left[uid, k] += rate * dleft
            right[mid, k] += rate * dright

            if rate*rate*dleft*dleft > max_change:
                max_change = rate*rate*dleft*dleft
            if rate*rate*dright*dright > max_change:
                max_change = rate*rate*dright*dright

        if iter % 100 == 0:
            print "  max_change %e, eps: %e, resid: %e" % (max_change, eps, total_err)
        if abs(old_resid - total_err)/total_err < eps or iter > maxiters:
            break
        old_resid = total_err




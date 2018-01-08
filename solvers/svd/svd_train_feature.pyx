#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

import numpy as np
cimport numpy as np

def svd_train_feature(float[:,:] left, float[:,:] right, float[:] resid, int k,
                      int[:] uids, int[:] mids, double[:] ratings, double rate, double eps,
                      int maxiters):
    _svd_train_feature(left, right, resid, k, uids, mids, ratings, rate, eps, maxiters)

cdef void _svd_train_feature(float[:,:] left, float[:,:] right, float[:] resid, int k,
                             int[:] uids, int[:] mids, double[:] ratings, double rate, double eps,
                             int maxiters):

    cdef double max_change, dleft, dright, dresid
    cdef double rat, yhat, err
    cdef int uid, mid, ob, iter

    max_change = eps + 1
    iter = 0
    while max_change > eps*rate*rate:
        max_change = 0
        for ob in range(uids.shape[0]):
            uid = uids[ob]
            mid = mids[ob]
            rat = ratings[ob]

            yhat = 0
            for k_ in range(k+1):
                yhat += left[uid, k_] * right[mid, k_]
            err = rat - yhat
            #err = resid[ob] - left[uid,k] * right[mid,k]


            dleft = err * right[mid, k]
            dright = err * left[uid, k]
            left[uid, k] += rate * dleft
            right[mid, k] += rate * dright

            if rate*rate*dleft*dleft > max_change:
                max_change = rate*rate*dleft*dleft
            if rate*rate*dright*dright > max_change:
                max_change = rate*rate*dright*dright

        iter += 1
        if iter > maxiters:
            break
        if iter % 100 == 0:
            print "  max_change %e, eps: %e" % (max_change, eps)




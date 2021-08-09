# cython: language_level=3, boundscheck=False
from libc.math cimport log, exp, INFINITY, NAN
from autograd.extend import primitive, defvjp
import numpy as np
from libc.math cimport exp
import cython

@primitive
def streaming_logsumexp(double [::1] a):
    # From: https://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
    cdef long i
    cdef double alpha = -INFINITY
    cdef double r = 0.0
    cdef double x
    for i in range(a.shape[0]):
        x = a[i]
        if x <= alpha:
            r += exp(x - alpha)
        else:
            r *= exp(alpha - x)
            r += 1.0
            alpha = x
    return log(r) + alpha

def streaming_logsumexp_vjp(ans, x):
    x_shape = x.shape
    return lambda g: np.full(x_shape, g) * np.exp(x - np.full(x_shape, ans))

defvjp(streaming_logsumexp, streaming_logsumexp_vjp)

cdef void lse_init(lse *l) nogil:
    l.alpha = -INFINITY
    l.r = 0.0
    l.ans = NAN

cdef void lse_update(lse* l, double x) nogil:
    if x <= l.alpha:
        l.r += exp(x - l.alpha)
    else:
        l.r *= exp(l.alpha - x)
        l.r += 1.0
        l.alpha = x

cdef double lse_result(lse* l) nogil:
    l.ans = log(l.r) + l.alpha

    return l.ans

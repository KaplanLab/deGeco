# cython: language_level=3, boundscheck=False
from libc.math cimport log, exp, INFINITY, NAN
from autograd.extend import primitive, defvjp
import numpy as np
from libc.math cimport exp

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

cdef class StreamingLogsumexp:
    def __cinit__(self):
        self.alpha = -INFINITY
        self.r = 0.0
        self.ans = NAN

    cpdef void update(self, double x):
        if x <= self.alpha:
            self.r += exp(x - self.alpha)
        else:
            self.r *= exp(self.alpha - x)
            self.r += 1.0
            self.alpha = x

    cpdef double result(self):
        self.ans = log(self.r) + self.alpha

        return self.ans

    cpdef double jac_elem(self, double x):
        return exp(x - self.ans)

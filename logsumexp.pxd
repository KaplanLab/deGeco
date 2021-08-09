# cython: language_level=3
cdef struct lse:
    double alpha
    double r
    double ans

cdef void lse_init(lse*) nogil
cdef void lse_update(lse*, double) nogil
cdef double lse_result(lse*) nogil

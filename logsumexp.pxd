# cython: language_level=3
cdef class StreamingLogsumexp:
    cdef double alpha
    cdef double r
    cdef double ans

    cpdef void update(self, double x)
    cpdef double result(self)
    cpdef double jac_elem(self, double x)

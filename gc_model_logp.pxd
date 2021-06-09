#cython: language_level=3

cdef double calc_dd(long i, long j, double alpha, double beta, long[::1] chr_assoc)
cdef double calc_gc(long i, long j, double[:, ::1] lambdas, double[:, ::1] weights)
cdef (double, double) lambdas_jac_element(long i, long j, long s1, double[:, ::1] lambdas, double[:, ::1] weights)
cdef double weights_jac_element(long i, long j, long s1, long s2, double[:, ::1] lambdas)
cdef (double, double) dd_jac_element(long i, long j, long[::1] chr_assoc)

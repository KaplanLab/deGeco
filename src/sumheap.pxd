# cython: language_level=3, boundscheck=False, infer_types=True, nonecheck=False
# cython: overflowcheck=False, initializedcheck=False, wraparound=False, cdivision=True
cdef class SumHeap:
    cdef readonly:
        long[:] S
        long n, d
        long remaining

    cpdef void seed(self, int s)
    cpdef void heapify(self, long[::1] w)
    cpdef void decrease(self, long k)
    cpdef long sample(self, u=*)
    cpdef long[::1] swor(self, long k)


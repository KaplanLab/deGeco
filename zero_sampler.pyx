# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
from libc.math cimport ceil, log2
from sumheap cimport SumHeap

cdef class ZeroSampler:
    cdef long nbins
    cdef int[::1] _bin1_id
    cdef int[::1] _bin2_id
    cdef int[::1] holes
    
    def __init__(self, long nbins, int[::1] bin1_id, int[::1] bin2_id):
        self.nbins = nbins
        self._bin1_id = bin1_id
        self._bin2_id = bin2_id
        
        self.holes = np.empty(bin1_id.shape[0]+1, dtype=np.int32)
        
    def _gap_size(self, r1, c1, r2, c2):
        cdef int partial_row1, partial_row2
        
        assert r1 <= r2 or c1 <= c2
        if r1 == r2:
            if c1 == c2:
                return 0
            return c2 - c1 - 1

        # Full rows aren't really full - we only need the upper triangle
        full_rows = np.sum([ self.nbins - i for i in range(r1+1, r2)])
        partial_row1 = self.nbins - c1 - 1
        partial_row2 = c2 - r2

        return full_rows + partial_row1 + partial_row2

    def sample_zeros(self, n):
        cdef:
            long i
            int cur_x, cur_y
            int prev_x=0, prev_y=0

        for i in range(self._bin1_id.shape[0]):
            cur_x = self._bin1_id[i]
            cur_y = self._bin2_id[i]
            self.holes[i] = self._gap_size(prev_x, prev_y, cur_x, cur_y)
            prev_x = cur_x
            prev_y = cur_y
        i = self._bin1_id.shape[0]
        self.holes[i] = self._gap_size(prev_x, prev_y, self.nbins-1, self.nbins)

        return np.asarray(self.holes)

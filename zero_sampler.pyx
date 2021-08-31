# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
from libc.math cimport ceil, log2
from sumheap cimport SumHeap

SAMPLE_BOTH = 0
SAMPLE_CIS = 1
SAMPLE_TRANS = 2

cdef class ZeroSampler:
    cdef long nbins
    cdef int[::1] _bin1_id
    cdef int[::1] _bin2_id
    cdef char[::1] _non_nan_mask
    cdef int[:, ::1] _zero_indices
    cdef long[::1] rows_index
    cdef readonly long[::1] zero_dist
    cdef readonly long zero_count
    cdef long[:] cis_lengths
    cdef int sample_area
    cdef SumHeap heap
    
    def __init__(self, long nbins, int[::1] bin1_id, int[::1] bin2_id, char[::1] non_nan_mask, cis_lengths=None, sample_area='both'):
        self.nbins = nbins
        self._bin1_id = bin1_id
        self._bin2_id = bin2_id
        self._non_nan_mask = non_nan_mask
        if sample_area == 'both':
            self.sample_area = SAMPLE_BOTH
            self.cis_lengths = None
        elif sample_area == 'trans':
            if cis_lengths is None:
                raise ValueError("cis_lengths is requried when sample_area=trans")
            self.sample_area = SAMPLE_TRANS
            self.cis_lengths = np.array(cis_lengths)
        else:
            raise ValueError(f"Bad sample_area value: {sample_area}")
        
        # nbins + 1 ensures we have an element following the last one, easier to slice in a loop later
        self.rows_index = np.searchsorted(bin1_id, np.arange(nbins+1))
        self.zero_dist = self._count_zeros()
        self.heap = SumHeap(self.zero_dist)
        self.zero_count = self.heap.remaining
        self._zero_indices = None
        
    def _count_zeros(self):
        cdef int i, s, e
        cdef long[::1] rows_index = self.rows_index
        cdef char[::1] non_nan_mask = self._non_nan_mask
        cdef int[::1] bin2_id = self._bin2_id
        cdef long[::1] zeros_counts = np.empty(self.nbins, dtype=int)
        cdef int sample_area = self.sample_area
        cdef long[::1] cis_offsets
        cdef long[::1] trans_lengths
        cdef int[::1] chr_assoc
        cdef int[::1] cols
        cdef long nn_cols, available_cols = np.sum(non_nan_mask)
        if sample_area == SAMPLE_TRANS:
            cis_offsets = np.cumsum(self.cis_lengths)
            trans_lengths = np.sum(self.cis_lengths) - cis_offsets
            chr_assoc = np.repeat(np.arange(np.size(self.cis_lengths)) , self.cis_lengths).astype(np.int32)
        for i in range(self.nbins):
            if not non_nan_mask[i]:
                zeros_counts[i] = 0
                continue
            s, e = rows_index[i], rows_index[i+1]
            if sample_area == SAMPLE_TRANS:
                available_cols = np.sum(non_nan_mask[cis_offsets[chr_assoc[i]]:])
                while s < cis_offsets[chr_assoc[i]]:
                    s += 1
            cols = bin2_id[s:e]
            nn_cols = self._non_nan_mask.base[cols].sum()
            zeros_counts[i] = available_cols - nn_cols
            available_cols -= 1
        
        return zeros_counts

    def sample_zeros(self, n):
        if n > self.zero_count:
            print(f"Trying to sample {n} which is more than total zeors count {self.zero_count}, using total instead")
            n = self.zero_count
        if self._zero_indices is None or self._zero_indices.shape != (2, n):
            self._zero_indices = np.empty((2, n), dtype=np.int32)
        cdef int[:, ::1] zero_indices = self._zero_indices
        cdef long row = 0
        cdef int r, col_idx
        cdef int[::1] sampled_cols
        
        cdef long[::1] rows_index = self.rows_index
        cdef int[::1] bin2_id = self._bin2_id
        cdef long[::1] zeros_rows = self.heap.swor(n)
        cdef int sample_area = self.sample_area
        cdef long[::1] cis_offsets
        cdef int[::1] chr_assoc
        if sample_area != SAMPLE_BOTH:
            cis_offsets = np.cumsum(self.cis_lengths)
            chr_assoc = np.repeat(np.arange(np.size(self.cis_lengths)) , self.cis_lengths).astype(np.int32)

        nan_cols = set(np.nonzero(self._non_nan_mask.base == 0)[0])
        for r in range(self.nbins):
            if zeros_rows[r] == 0:
                continue
            s, e = rows_index[r], rows_index[r+1]
            if sample_area == SAMPLE_TRANS:
                all_zero_set = set(range(cis_offsets[chr_assoc[r]], self.nbins))
                while s < cis_offsets[chr_assoc[r]]:
                    s += 1
            else:
                all_zero_set = set(range(r, self.nbins))
            nonzero_cols = bin2_id[s:e]
            available_cols = np.array(list(all_zero_set - set(nonzero_cols) - nan_cols), dtype=np.int32)
            sampled_cols = np.random.choice(available_cols, size=zeros_rows[r], replace=False)
            for col_idx in range(sampled_cols.shape[0]):
                zero_indices[0, row] = r
                zero_indices[1, row] = sampled_cols[col_idx]
                row += 1

        return np.asarray(zero_indices)

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np

def balance_counts(int[::1] bin1_id not None, int[::1] bin2_id not None, counts not None,
                   double[::1] balance_weights not None):
    balanced_counts = np.ndarray.astype(counts, float, copy=False)
    cdef double[::1] balanced_counts_view = balanced_counts
    cdef int i, j

    for row in range(balanced_counts_view.shape[0]):
        i = bin1_id[row]
        j = bin2_id[row]
        balanced_counts_view[row] *= balance_weights[i] * balance_weights[j]

    return balanced_counts

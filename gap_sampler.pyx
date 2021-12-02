# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
cimport cython
import numpy as np

cdef long get_position(long p, long[::1] gaps_cumsum, long gaps_pos) nogil:
    while p >= gaps_cumsum[gaps_pos]:
        gaps_pos += 1
    return gaps_pos

cdef int get_gap(long[::1] gaps_cumsum, long p, long i) nogil:
    if p == 0:
        return i + 1
    return i - gaps_cumsum[p-1] + 1

cdef (int, int) pos2rowcol(long p, int[::1] bin1_id, int[::1] bin2_id) nogil:
    cdef long nonzero_p

    if p == 0:
        return 0, -1
    nonzero_p = p - 1

    return bin1_id[nonzero_p], bin2_id[nonzero_p]

cdef (int, int) move_right(int nbins, int row, int col, int n) nogil:
    #flattened = (nbins + nbins - (row-1)) * row // 2 + col
    #new_flattened = flattened + n
    #new_row = int(((2*nbins+1) - np.sqrt((2*nbins+1)**2 - 8*new_flattened))/2)
    #new_col = new_flattened - (nbins + nbins - (new_row-1)) * new_row // 2 
    cdef int new_col, new_row

    new_col = col + n
    new_row = row
    while new_col >= nbins:
        new_row += 1
        new_col = new_col - nbins + new_row

    return new_row, new_col

@cython.wraparound(True)
cpdef sample(long nbins, int[::1] bin1_id, int[::1] bin2_id, long[::1] gaps_cumsum, int step=1, int start=0):
    count = gaps_cumsum[-1]
    pos = 0
    zeros = []
    for i in range(start, count, step):
        pos = get_position(i, gaps_cumsum, pos)
        i_gap = get_gap(gaps_cumsum, pos, i)
        row, col = pos2rowcol(pos, bin1_id, bin2_id)
        zero_row, zero_col = move_right(nbins, row, col, i_gap)
        zeros.append((zero_row, zero_col))

    return zeros

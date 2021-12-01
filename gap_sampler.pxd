# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

cdef int get_position(int p, long[::1] gaps_cumsum, long gaps_pos) nogil
cdef int get_gap(long[::1] gaps_cumsum, long p, int i) nogil
cdef (int, int) pos2rowcol(long p, int[::1] bin1_id, int[::1] bin2_id) nogil
cdef (int, int) move_right(int nbins, int row, int col, int n) nogil
cpdef sample(long nbins, int[::1] bin1_id, int[::1] bin2_id, long[::1] gaps_cumsum, int step=*, int start=*)

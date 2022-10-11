# cython: language_level=3, boundscheck=False, infer_types=True, nonecheck=False
# cython: overflowcheck=False, initializedcheck=False, wraparound=False, cdivision=True
# Taken from: https://raw.githubusercontent.com/timvieira/arsenal/master/arsenal/datastructures/heap/sumheap.pyx
# Described in http://timvieira.github.io/blog/post/2016/11/21/heaps-for-incremental-computation/
import numpy as np

cdef extern from "stdlib.h":
    double drand48()
    void srand48(int)

from libc.math cimport log2, ceil


cdef class SumHeap:
    def __init__(self, long[::1] w):
        self.n = w.shape[0]
        self.remaining = np.sum(w)
        self.d = long(2**ceil(log2(self.n)))   # number of intermediates
        self.S = np.zeros(2*self.d, dtype=int)           # intermediates + leaves
        self.heapify(w)

    def __getitem__(self, long k):
        return self.S[self.d + k]

    cpdef void seed(self, int s):
        srand48(s)

    cpdef void heapify(self, long[::1] w):
        "Create sumheap from weights `w` in O(n) time."
        d = self.d; n = self.n
        self.S[d:d+n] = w                         # store `w` at leaves.
        for i in reversed(range(1, d)):
            self.S[i] = self.S[2*i] + self.S[2*i + 1]

    cpdef void decrease(self, long k):
        "Update w[k] -= 1` in time O(log n)."
        i = self.d + k
        self.S[i] -= 1
        while i > 0:   # fix parents in the tree.
            i //= 2
            self.S[i] -= 1

    cpdef long sample(self, u=None):
        "Sample from sumheap, O(log n) per sample."
        cdef long left
        cdef double p, r
        if u is None:
            r = drand48()
        else:
            r = u
        d = self.S.shape[0]//2     # number of internal nodes.
        p = r * self.S[1]  # random probe, p ~ Uniform(0, z)
        # Use binary search to find the index of the largest CDF (represented as a
        # heap) value that is less than a random probe.
        i = 1
        while i < d:
            # Determine if the value is in the left or right subtree.
            i *= 2            # Point at left child
            left = self.S[i]  # Probability mass under left subtree.
            if p > left:      # Value is in right subtree.
                p -= left     # Subtract mass from left subtree
                i += 1        # Point at right child
        return i - d

    cpdef long[::1] swor(self, long k):
        "Sample without replacement `k` times."
        cdef long s
        cdef long[::1] z = np.zeros(self.n, dtype=int)
        if k > self.remaining:
            raise ValueError(f"Can't sample {k} when only {self.remaining} are available")
        for i in range(k):
            s = self.sample()
            z[s] += 1
            self.decrease(s)
        self.remaining -= k
        return z

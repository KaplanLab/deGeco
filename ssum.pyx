def msum(double[::1] iterable):
    "Full precision summation using multiple floats for intermediate values"
    # Rounded x+y stored in hi with the round-off stored in lo.  Together
    # hi+lo are exactly equal to x+y.  The inner loop applies hi/lo summation
    # to each partial so that the list of partial sums remains exact.
    # Depends on IEEE-754 arithmetic guarantees.  See proof of correctness at:
    # www-2.cs.cmu.edu/afs/cs/project/quake/public/papers/robust-arithmetic.ps

    cdef long k, m, i
    cdef double x, y, hi, lo

    partials = []               # sorted, non-overlapping partial sums
    for k in range(iterable.shape[0]):
        x = iterable[k]
        i = 0
        for m in range(len(partials)):
            y = partials[m] 
            if x < y:
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo:
                partials[i] = lo
                i += 1
            x = hi
        partials[i:] = [x]
    return sum(partials, 0.0)

cdef inline (double, double) fsum_step(double sum, double c, double x):
    cdef double t, y

    y = x - c
    t = sum + y
    c = (t - sum) - y

    return t, c


def fsum(double[::1] iterable):
    cdef double sum = 0.0
    cdef double c = 0.0
    cdef double y, t
    cdef long i

    for i in range(iterable.shape[0]):
        sum, c = fsum_step(sum, c, iterable[i])

    return sum


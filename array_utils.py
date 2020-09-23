import warnings
from autograd.extend import primitive, defvjp
import autograd.numpy as np
from toolz import memoize

memoize_tri = memoize(np.tri)

@primitive
def get_lower_triangle(mat, k=-1):
    """
    Return a vector of values of the lower triangle of the given matrix, starting from the k-th diagonal.

    :param array mat: matrix
    :return: A vector of the lower triangle's values
    :rtype: 1d array
    """
    shape = mat.shape
    tri_mask = memoize_tri(N=shape[0], M=shape[1], k=k, dtype=bool)

    return mat[tri_mask]

def get_lower_triangle_vjp(ans, mat, k=-1):
    def _vjp(g):
        n, m = mat.shape
        tri_mask = memoize_tri(N=n, k=k, M=m, dtype=bool)
        r = np.zeros(mat.shape)
        r[tri_mask] = g
        return r
    return _vjp

defvjp(get_lower_triangle, get_lower_triangle_vjp)

def normalize(array, normalize_axis=None):
    """
    Normalize the given array by dividing by array.sum(axis=normalize_axis). This is used to get the
    'real' values of values that have some normalization constraint (such as probabilities). We do 
    this to bypass a L-BFGS-B limitation that only supports bound contraints.
    """
    normalization_factor = array.sum(axis=normalize_axis)[None].T # To make broadcasting work

    return array/normalization_factor

def nannormalize(array, normalize_axis=None):
    """
    Normalize the given array by dividing by np.nansum(array, axis=normalize_axis).
    """
    normalization_factor = np.nansum(array, axis=normalize_axis)[None].T # To make broadcasting work

    return array/normalization_factor

def normalize_tri_l1(a):
    """
    Divides the input matrix by the L1 norm of the lower triangle of A.

    * not autograd-safe.
    """
    return a / np.nansum(np.tril(a, -1))

def triangle_to_symmetric(N, tri_values, k=0, fast=False):
    """
    Convert the lower triangle values (from k-th diagonal) given by tri_values to a symmetric NxN matrix

    if fast=True runs a faster version that's not autograd-safe
    """
    # This implementation is a bit more complicated than expected, because autograd doesn't support
    # array assignments (so symmat[x,y] = symmat[y,x] = values won't work). Instead we build the matrix one
    # column at a time
    if k > 0:
        raise ValueError("Only lower triangle is supported")
    x, y = np.tril_indices(N, k=k)

    if fast:
        symmat = np.zeros((N, N))
        symmat[x, y] = symmat[y, x] = tri_values
        return symmat

    nans_insert = np.full(abs(k), np.nan)

    def get_values(index):
        return tri_values[(x == index) | (y == index)]

    def pad_values(i, v):
        # insert nans in the diags up to k
        if k == 0:
            return v
        return np.insert(v, i, nans_insert)

    symmat = np.array([pad_values(i, get_values(i)) for i in range(N)])
    return symmat

def ensure_symmetric(a):
    """
    Make A[i, j] = A[j, i] = max(A[i, j], A[j, i])
    """
    return np.maximum(a, a.T)
    
def remove_main_diag(a):
    """
    Put NaNs in the main diagonal of A so it's removed from all calculations.
    """
    nan_in_diag = a.copy()
    np.fill_diagonal(nan_in_diag, np.nan)

    return nan_in_diag

def balance(matrix, epsilon=1e-3):
    """
    Ensure all rows and columns of the given matrix have the same mean, up to epsilon
    """
    column_mean = lambda m: np.nanmean(m, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        while np.any(np.abs(column_mean(matrix) - 1.0) > epsilon):
          matrix = matrix / column_mean(matrix)
          matrix = matrix.T # transpose and do the same for rows

    return matrix


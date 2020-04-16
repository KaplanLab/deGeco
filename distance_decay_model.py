import autograd.numpy as np
from toolz import memoize

from array_utils import get_lower_triangle

def distance_matrix(n):
    indices = np.arange(n)
    return 1.0 * np.abs(indices[:, None] - indices[None, :])

@memoize(key=lambda args, kwargs: tuple(args[1]))
def log_distance_vector(n, non_nan_mask):
    distances_filtered = distance_matrix(n)[non_nan_mask, :][:, non_nan_mask]
    return np.log(get_lower_triangle(distances_filtered))

def log_distance_decay(n, non_nan_mask, alpha):
    dd = log_distance_vector(n, non_nan_mask) * alpha
    return dd

def init_variables():
    return (-1,)

def init_bounds():
    return [(-2, -0.5)]

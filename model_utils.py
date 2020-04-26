import autograd.numpy as np
from autograd.scipy.special import logsumexp

def log_likelihood(x, log_p):
    x_mask = np.isfinite(x)
    usable_x = x[x_mask]
    usable_log_p = log_p[x_mask]

    log_z = logsumexp(usable_log_p)

    return np.sum(np.multiply(usable_x, usable_log_p - log_z))

def expand_by_mask(a, mask, blank_value=np.nan):
    target_shape = list(a.shape)
    target_shape[0] = mask.size
    
    expanded = np.full(target_shape, blank_value)
    expanded[mask] = a

    return expanded

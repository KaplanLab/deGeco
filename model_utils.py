import autograd.numpy as np
from autograd.scipy.special import logsumexp

def log_likelihood(x, log_p):
    log_z = logsumexp(log_p)

    return np.sum(np.multiply(x, log_p - log_z))

def expand_by_mask(a, mask, blank_value=np.nan):
    target_shape = list(a.shape)
    target_shape[0] = mask.size
    
    expanded = np.full(target_shape, blank_value)
    expanded[mask] = a

    return expanded

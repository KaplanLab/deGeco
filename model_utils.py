import autograd.numpy as np
from logsumexp import streaming_logsumexp as logsumexp
import array_utils
from autograd.extend import primitive, defvjp

# TODO: Convert this to a class
def log_likelihood_by(x, A=1):
    x_mask = np.isfinite(x)
    usable_x = A * x[x_mask]
    x_sum = np.sum(usable_x)
    log_z = None

    @primitive
    def log_likelihood(log_p, z_const=-np.inf, zeros=None):
        nonlocal x_mask
        nonlocal usable_x
        nonlocal log_z
        nonlocal x_sum

        if zeros is None:
            usable_log_p = np.append(log_p[x_mask], z_const)
            assert (np.isfinite(usable_log_p[:-1]).all()) == True

            # TODO: Put that in Cython to save memory - scipy's logsumexp copies its data
            log_z = logsumexp(usable_log_p)
            assert (np.isfinite(log_z).all()) == True

            A = usable_x @ usable_log_p[:-1]  - x_sum * log_z
        else:
            usable_log_p = log_p
            assert (np.isfinite(usable_log_p).all()) == True

            # TODO: Put that in Cython to save memory - scipy's logsumexp copies its data
            log_z = logsumexp(np.append(usable_log_p, zeros))
            assert (np.isfinite(log_z).all()) == True

            A = usable_x @ usable_log_p  - x_sum * log_z
        return A

    def log_likelihood_vjp(ans, log_p):
        def _vjp(g):
            nonlocal x_mask
            nonlocal usable_x
            nonlocal log_z
            nonlocal x_sum
            usable_log_p = log_p[x_mask]

            logsumexp_jac = np.exp(usable_log_p - log_z)

            r = np.zeros(log_p.shape)
            r[x_mask] = g * (usable_x - x_sum * logsumexp_jac)
            return r
        return _vjp

    defvjp(log_likelihood, log_likelihood_vjp)

    return log_likelihood

def expand_by_mask(a, mask, blank_value=np.nan):
    target_shape = list(a.shape)
    target_shape[0] = mask.size
    
    expanded = np.full(target_shape, blank_value)
    expanded[mask] = a

    return expanded

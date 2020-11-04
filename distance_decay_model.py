import autograd.numpy as np
from autograd import grad
import scipy as sp
from toolz import memoize

import array_utils
import hic_analysis
import model_utils

def distance_matrix(n):
    if np.ndim(n) == 0:
        bin_distances = np.arange(n)
    elif np.ndim(n) == 1:
        bin_distances = np.append([0], np.cumsum(n))
    else:
        raise ValueError("n must be a scalar or 1D vector")

    return 1.0 * np.abs(bin_distances[:, None] - bin_distances[None, :])

@memoize(key= lambda args, kwargs: (tuple(args[0]), tuple(args[1])))
def cis_trans_mask(cis_lengths, non_nan_mask):
    groups = np.arange(np.size(cis_lengths))
    groups_per_bin = np.repeat(groups, cis_lengths)[non_nan_mask]
    cis_trans_matrix = groups_per_bin[:, None] == groups_per_bin[None, :]

    return array_utils.get_lower_triangle(cis_trans_matrix)

@memoize(key=lambda args, kwargs: tuple(args[1]))
def log_distance_vector(n, non_nan_mask):
    distances_filtered = distance_matrix(n)[non_nan_mask, :][:, non_nan_mask]
    return np.log(array_utils.get_lower_triangle(distances_filtered))

def log_distance_decay(cis_lengths, non_nan_mask, alpha, beta):
    n = np.sum(cis_lengths)
    mask = cis_trans_mask(cis_lengths, non_nan_mask)
    cis_interactions = log_distance_vector(n, non_nan_mask) * alpha
    trans_interactions = np.full(cis_interactions.shape, beta)

    return np.where(mask, cis_interactions, trans_interactions)

def init_variables(init_values=None):
    alpha = -1
    beta = -2
    if init_values:
        alpha = init_values.get('alpha', alpha)
        beta = init_values.get('beta', beta)
    return (alpha, beta)

def init_bounds(fixed_values=None):
    bounds = [(-2, -0.5), (None, 0)]
    if fixed_values:
        fixed_alpha = fixed_values.get('alpha')
        fixed_beta = fixed_values.get('beta')
        bounds[0] = bounds[0] if fixed_alpha is None else (fixed_alpha, fixed_alpha)
        bounds[1] = bounds[1] if fixed_beta is None else (fixed_beta, fixed_beta)

    return bounds

def extract_params(variables, n, non_nan_mask):
    alpha, beta = variables[:2]

    return n, non_nan_mask, alpha, beta

def fit(interactions_mat):
    """
    """
    number_of_bins = interactions_mat.shape[0]
    non_nan_mask = ~np.isnan(interactions_mat).all(1)
    new_number_of_bins = non_nan_mask.sum()
    unique_interactions = array_utils.get_lower_triangle(hic_analysis.remove_unusable_bins(hic_analysis.preprocess(interactions_mat)))

    x0 = np.concatenate([
            init_variables(), # alpha
        ])
    bounds = np.concatenate([
            init_bounds(), # alpha
        ])
    optimize_options = dict(disp=True, ftol=1.0e-20, gtol=1e-020, eps=1e-20, maxfun=10000000, maxiter=10000000, maxls=100)

    log_likelihood = model_utils.log_likelihood_by(unique_interactions)
    def likelihood_minimizer(variables):
        model_params = extract_params(variables, number_of_bins, non_nan_mask)
        model_interactions = log_distance_decay(*model_params)
        l = -log_likelihood(model_interactions)
        return l

    likelihood_grad = grad(likelihood_minimizer)
    res = sp.optimize.minimize(fun=likelihood_minimizer, x0=x0, method='L-BFGS-B', jac=likelihood_grad, bounds=bounds,
            options=optimize_options)

    number_of_bins, _, alpha, = extract_params(res.x, number_of_bins, non_nan_mask)
    
    return alpha, number_of_bins

def generate_interactions_matrix(alpha, number_of_bins):
    """
    """
    non_nan_mask = np.ones(number_of_bins, dtype=bool)
    log_interactions = log_distance_decay(number_of_bins, non_nan_mask, alpha)
    interactions_vec = array_utils.nannormalize(np.exp(log_interactions))
    interactions_mat = array_utils.remove_main_diag(array_utils.triangle_to_symmetric(number_of_bins, interactions_vec, k=-1))

    return interactions_mat

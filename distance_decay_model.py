import autograd.numpy as np
from autograd import grad
import scipy as sp
from toolz import memoize

import array_utils
import hic_analysis
import model_utils

def distance_matrix(n):
    indices = np.arange(n)
    return 1.0 * np.abs(indices[:, None] - indices[None, :])

@memoize(key=lambda args, kwargs: tuple(args[1]))
def log_distance_vector(n, non_nan_mask):
    distances_filtered = distance_matrix(n)[non_nan_mask, :][:, non_nan_mask]
    return np.log(array_utils.get_lower_triangle(distances_filtered))

def log_distance_decay(n, non_nan_mask, alpha):
    dd = log_distance_vector(n, non_nan_mask) * alpha
    return dd

def init_variables():
    return (-1,)

def init_bounds():
    return [(-2, -0.5)]

def extract_params(variables, n, non_nan_mask):
    alpha = variables[0]

    return n, non_nan_mask, alpha

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

    def likelihood_minimizer(variables):
        model_params = extract_params(variables, number_of_bins, non_nan_mask)
        model_interactions = log_distance_decay(*model_params)
        l = -model_utils.log_likelihood(unique_interactions, model_interactions)
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

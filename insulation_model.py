import autograd.numpy as np
from autograd.scipy.special import logsumexp
from autograd import grad
from scipy.sparse import dia_matrix
import scipy as sp
from scipy import optimize
import functools
import itertools
from toolz import memoize, curry
from toolz.curried import do, compose, thread_first

from hic_analysis import preprocess, remove_unusable_bins 
from array_utils import get_lower_triangle, triangle_to_symmetric, nannormalize, remove_main_diag
from model_utils import expand_by_mask, log_likelihood_by
import distance_decay_model

@memoize
def get_start_mat(N):
    return np.flipud(np.triu(np.ones((N,N)), k=2))

def log_insulation_probability(log_ck):
    """
    Calculates the log probability of insulation between any two loci.

    :param array log_ck: (N-2) sized vector with the log probability of _no_ insulation at each locus

    :return: a NxN matrix where cell i,j is the probability of insulation between loci i,j
    """
    N = len(log_ck) + 2
    start_mat = get_start_mat(N)
    padded_log_ck = np.concatenate(([0, 0], log_ck))
    flipped_probability_mat = np.cumsum(start_mat * padded_log_ck[None, :], axis=1)

    return get_lower_triangle(np.flipud(flipped_probability_mat).T)

def log_interaction_probability(log_ck, alpha, n, non_nan_mask):
    dd = distance_decay_model.log_distance_decay([n], non_nan_mask, alpha, -1)
    ip = log_insulation_probability(log_ck)
    
    return dd + ip

def extract_params(variables, n, non_nan_mask):
    log_ck = variables[:-1]
    alpha = variables[-1]

    return log_ck, alpha, n, non_nan_mask

def init_variables(ck_param_count):
    ck_init = np.log(np.random.rand(ck_param_count))

    return ck_init

def init_bounds(ck_param_count):
    ck_bounds = [(None, 0)] * ck_param_count

    return ck_bounds

@curry
def remove_diags_from(start_diag, a):
    if start_diag is None:
        return a
    _a = a.copy()
    max_diag = a.shape[0]
    for d in range(start_diag, max_diag):
        np.fill_diagonal(_a[:, d:], np.nan)
        np.fill_diagonal(_a[d:], np.nan)
    return _a

def fit(interactions_mat, valid_distance=None):
    """
    """
    number_of_bins = interactions_mat.shape[0]
    non_nan_mask = ~np.isnan(interactions_mat).all(1)
    new_number_of_bins = non_nan_mask.sum()
    unique_interactions = thread_first(interactions_mat, preprocess, remove_diags_from(valid_distance), remove_unusable_bins,
                                       get_lower_triangle)

    ck_param_count = new_number_of_bins - 2
    assert ck_param_count > 0
    x0 = np.concatenate([
        init_variables(ck_param_count),
        distance_decay_model.init_variables()
    ])
    bounds = np.concatenate([
        init_bounds(ck_param_count),
        distance_decay_model.init_bounds()
    ])
    optimize_options = dict(disp=True, ftol=1.0e-20, gtol=1e-020, eps=1e-20, maxfun=10000000, maxiter=10000000, maxls=100)

    log_likelihood = model_utils.log_likelihood_by(unique_interactions)
    def likelihood_minimizer(variables):
        model_params = extract_params(variables, number_of_bins, non_nan_mask)
        model_interactions = log_interaction_probability(*model_params)
        return -log_likelihood(model_interactions)
    def tap(p):
        def _tap(x):
            print(f"{p}: {x}")
            return x
        return _tap
    likelihood_grad = grad(likelihood_minimizer)
    res = sp.optimize.minimize(fun=likelihood_minimizer, x0=x0, method='L-BFGS-B', jac=likelihood_grad, bounds=bounds,
            options=optimize_options)

    log_ck, alpha, *_ = extract_params(res.x, number_of_bins, non_nan_mask)
    log_ck_with_edges = np.concatenate([[0], log_ck, [0]])
    expanded_log_ck = expand_by_mask(log_ck_with_edges, non_nan_mask)[1:-1]

    return expanded_log_ck, alpha

def generate_interactions_matrix(insulation_probabilities, alpha):
    number_of_bins = insulation_probabilities.shape[0] + 2
    non_nan_mask = np.ones(number_of_bins, dtype=bool)

    log_p = log_interaction_probability(np.nan_to_num(insulation_probabilities), alpha, number_of_bins, non_nan_mask)
    P_vec = nannormalize(np.exp(log_p))
    P_mat = remove_main_diag(triangle_to_symmetric(number_of_bins, P_vec, k=-1))

    return P_mat

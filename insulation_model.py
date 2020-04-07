import autograd.numpy as np
from autograd.scipy.special import logsumexp
from autograd import grad
from scipy.sparse import dia_matrix
import scipy as sp
from scipy import optimize
import functools
import itertools
from toolz import memoize
from toolz.curried import do, compose

from hic_analysis import preprocess, remove_unusable_bins 
from array_utils import get_lower_triangle, triangle_to_symmetric, nannormalize, remove_main_diag
from model_utils import expand_by_mask, log_likelihood
from distance_decay_model import log_distance_decay

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
    dd = log_distance_decay(n, non_nan_mask, alpha)
    ip = log_insulation_probability(log_ck)
    
    return dd + ip

def extract_params(variables, n, non_nan_mask):
    log_ck = variables[:-1]
    alpha = variables[-1]

    return log_ck, alpha, n, non_nan_mask

def init_variables(number_of_bins):
    ck_init = np.log(np.random.rand(number_of_bins))
    alpha_init = (-1,)

    return np.concatenate((ck_init, alpha_init))

def init_bounds(number_of_bins):
    ck_bounds = [(None, 0)] * number_of_bins
    alpha_bounds = [(-2, -0.5)]
    bounds = np.concatenate((ck_bounds, alpha_bounds), axis=0)  

    return bounds

def fit(interactions_mat):
    """
    """
    number_of_bins = interactions_mat.shape[0]
    non_nan_mask = ~np.isnan(interactions_mat).all(1)
    new_number_of_bins = non_nan_mask.sum()
    unique_interactions = get_lower_triangle(remove_unusable_bins(preprocess(interactions_mat)))

    ck_param_count = new_number_of_bins - 2
    assert ck_param_count > 0
    x0 = init_variables(ck_param_count)
    bounds = init_bounds(ck_param_count)
    optimize_options = dict(disp=True, ftol=1.0e-20, gtol=1e-020, eps=1e-20, maxfun=10000000, maxiter=10000000, maxls=100)

    def likelihood_minimizer(variables):
        model_params = extract_params(variables, number_of_bins, non_nan_mask)
        model_interactions = log_interaction_probability(*model_params)
        return -log_likelihood(unique_interactions, model_interactions)
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

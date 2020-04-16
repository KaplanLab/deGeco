from __future__ import division
import autograd.numpy as np
from autograd import grad
import functools
import scipy as sp
from scipy import optimize

from hic_analysis import preprocess, remove_unusable_bins, zeros_to_nan
from array_utils import get_lower_triangle, normalize, nannormalize, triangle_to_symmetric, remove_main_diag
from model_utils import log_likelihood, expand_by_mask
import distance_decay_model

def compartments_interactions(state_probabilities, state_weights):
    return get_lower_triangle(state_probabilities @ state_weights @ state_probabilities.T)

def log_interaction_probability(state_probabilities, state_weights, distance_decay_power, number_of_bins, non_nan_mask):
    compartments = np.log(compartments_interactions(state_probabilities, state_weights))
    distance_decay = distance_decay_model.log_distance_decay(number_of_bins, non_nan_mask, distance_decay_power)
    return compartments + distance_decay

def extract_params(variables, probabilities_params_count, weights_param_count, number_of_states, weights_function,
        lambdas_function, number_of_bins, non_nan_mask):
    """
    Slice and transform the given 1-D variable vector into the model parameters
    """
    weights_start_idx = probabilities_params_count
    weights_end_idx = weights_start_idx + weights_param_count

    prob_vars = variables[:probabilities_params_count]
    weights_vars = variables[weights_start_idx:weights_end_idx]
    alpha = variables[-1]

    lambdas = normalize(lambdas_function(prob_vars).reshape(-1, number_of_states), normalize_axis=1)
    weights = normalize(weights_function(number_of_states, weights_vars))

    return lambdas, weights, alpha, number_of_bins, non_nan_mask

def init_variables(probabilities_params_count, weights_param_count, lambdas=None, weights=None, alpha=None):
    """
    Give an intial value to all the variables we optimize.
    """
    if lambdas is None:
        prob_init = (np.random.rand(probabilities_params_count) / 2) + 0.25
    else:
        assert lambdas.size == probabilities_params_count
        prob_init = np.copy(lambdas)

    if weights is None:
        weights_init = normalize(np.random.rand(weights_param_count))
    else:
        assert weights.size == weights_param_count
        weights_init = np.copy(weights)

    x0 = np.concatenate((prob_init, weights_init))

    return x0

def init_bounds(probabilities_params_count, weights_param_count):
    """
    Set bound for the probability and state weights
    """
    prob_bounds = [(0.01, 0.99)] * probabilities_params_count 
    weights_bounds = [(0, 1)] * weights_param_count
    bounds = np.concatenate((prob_bounds, weights_bounds), axis=0)  

    return bounds

def lambdas_hyper_default(non_nan_mask, number_of_states):
    non_nan_indices = non_nan_mask.sum()
    param_count = non_nan_indices * number_of_states
    func = lambda p: p

    return func, param_count

def weight_hyperparams(shape, number_of_states):
    if shape == 'symmetric':
        function = triangle_to_symmetric
        param_count = np.tril_indices(number_of_states)[0].size
    elif shape == 'diag':
        function = lambda n, v: np.diag(v)
        param_count = number_of_states
    elif shape == 'binary' or shape == 'eye':
        function = lambda n, v: np.eye(n)
        param_count = 1 # should be 0, but later code doesn't handle that well
    elif shape == 'eye_with_empty':
        function = lambda n, v:  np.diag(np.concatenate([[0], np.ones(n-1)]))
        param_count = 1 # should be 0, but later code doesn't handle that well
    else:
        raise ValueError("Invalid weight shape")

    return function, param_count

def fit(interactions_mat, number_of_states=2, weights_shape='symmetric', lambdas_hyper=None, init_values={}):
    """
    Return the model parameters that best explain the given Hi-C interaction matrix using L-BFGS-B.

    :param array interaction_mat: the Hi-C interaction matrix as a numpy array
    :param int number_of_states: the number of possible states (compartments) for each bin
    :param str weights_shape: how the weights matrix should look. can be: 'symmetric', 'diag', 'eye', 'eye_with_empty'
    :return: A tuple of each bin's state probability (shape BINSxSTATES), state-state interaction
             probabiity (shape STATES-STATES) and the distance-decay power value (scalar)
    :rtype: tuple
    """
    number_of_bins = interactions_mat.shape[0]
    non_nan_mask = ~np.isnan(interactions_mat).all(1)
    unique_interactions = get_lower_triangle(remove_unusable_bins(preprocess(interactions_mat)))

    _lambdas_hyper = lambdas_hyper if lambdas_hyper is not None else lambdas_hyper_default
    lambdas_function, probabilities_params_count = _lambdas_hyper(non_nan_mask, number_of_states)
    weights_function, weights_param_count = weight_hyperparams(weights_shape, number_of_states)

    x0 = np.concatenate([
        init_variables(probabilities_params_count, weights_param_count, **init_values),
        distance_decay_model.init_variables()
    ])
    bounds = np.concatenate([
        init_bounds(probabilities_params_count, weights_param_count),
        distance_decay_model.init_bounds()
    ])
    optimize_options = dict(disp=True, ftol=1.0e-20, gtol=1e-020, eps=1e-20, maxfun=10000000, maxiter=10000000, maxls=100)

    def likelihood_minimizer(variables):
        model_params = extract_params(variables, probabilities_params_count, weights_param_count, number_of_states,
                weights_function, lambdas_function, number_of_bins, non_nan_mask)
        model_interactions = log_interaction_probability(*model_params)
        return -log_likelihood(unique_interactions, model_interactions)

    likelihood_grad = grad(likelihood_minimizer)
    res = sp.optimize.minimize(fun=likelihood_minimizer, x0=x0, method='L-BFGS-B', jac=likelihood_grad, bounds=bounds,
            options=optimize_options)

    model_probabilities, model_weights, model_dd_power, *_ = extract_params(res.x, probabilities_params_count,
            weights_param_count, number_of_states, weights_function, lambdas_function, number_of_bins, non_nan_mask)
    expanded_model_probabilities = expand_by_mask(model_probabilities, non_nan_mask)
    
    return expanded_model_probabilities, model_weights, model_dd_power

def generate_interactions_matrix(state_probabilities, state_weights, distance_decay_power):
    """
    Generate an NxN interactions matrix by using the given model parameters. Diagonal values will be
    NaN as they're aren't modeled, along with any bins that have NaN in their state probabilities.

    :param array state_probabilities: NxM matrix, N being the bin count and M the number of states
    :param array state_weights: MxM symmetric matrix for state-state interaction probability
    :param number distance_decay_power: The rate by which interaction rate decreases with distance
    :return: interaction matrix generated by the model
    :rtype: NxN array
    """
    number_of_bins = state_probabilities.shape[0]
    non_nan_mask = np.ones(number_of_bins, dtype=bool)
    log_interactions = log_interaction_probability(state_probabilities, state_weights, distance_decay_power,
            number_of_bins, non_nan_mask)
    interactions_vec = nannormalize(np.exp(log_interactions))
    interactions_mat = remove_main_diag(triangle_to_symmetric(number_of_bins, interactions_vec, k=-1))

    return interactions_mat

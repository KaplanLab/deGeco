from __future__ import division
import autograd.numpy as np
from autograd import value_and_grad
import scipy as sp
from scipy import optimize
import itertools

from hic_analysis import preprocess, remove_unusable_bins, zeros_to_nan
from array_utils import get_lower_triangle, normalize, nannormalize, triangle_to_symmetric, remove_main_diag
from model_utils import log_likelihood_by, expand_by_mask
import distance_decay_model

def compartments_interactions(state_probabilities, state_weights):
    return get_lower_triangle(state_probabilities @ state_weights @ state_probabilities.T)

def log_interaction_probability(state_probabilities, state_weights, cis_dd_power, trans_dd, cis_lengths,
        non_nan_mask):
    compartments = np.log(compartments_interactions(state_probabilities, state_weights))
    distance_decay = distance_decay_model.log_distance_decay(cis_lengths, non_nan_mask, cis_dd_power, trans_dd)
    return compartments + distance_decay

def extract_params(variables, probabilities_params_count, weights_param_count, number_of_states, weights_function,
        lambdas_function, cis_lengths, non_nan_mask):
    """
    Slice and transform the given 1-D variable vector into the model parameters
    """
    weights_start_idx = probabilities_params_count
    weights_end_idx = weights_start_idx + weights_param_count

    prob_vars = variables[:probabilities_params_count]
    weights_vars = variables[weights_start_idx:weights_end_idx]
    if not weights_vars.any():
        # make all-zero weights behave simply as all-equal weights
        weights_vars = np.ones(weights_end_idx - weights_start_idx)
    alpha, beta = variables[-2:]

    lambdas = normalize(lambdas_function(prob_vars).reshape(-1, number_of_states), normalize_axis=1)
    weights = normalize(weights_function(number_of_states, weights_vars))

    return lambdas, weights, alpha, beta, cis_lengths, non_nan_mask

def init_variables(probabilities_params_count, weights_init, init_values_override=None):
    """
    Give an intial value to all the variables we optimize.
    """
    if init_values_override:
        lambdas = init_values.get('lambdas')
        weights = init_values.get('weights')
    else:
        lambdas = weights = None
    if lambdas is None:
        prob_init = (np.random.rand(probabilities_params_count) / 2) + 0.25
    else:
        if np.ndim(lambdas) > 1:
            lambdas = lambdas[~np.isnan(lambdas).all(axis=1)]
        assert lambdas.size == probabilities_params_count
        prob_init = np.copy(lambdas.flatten())

    if weights is not None:
        if np.ndim(weights) > 1:
            weights = get_lower_triangle(weights, k=0)
        assert weights.size == weights_param_count
        weights_init = np.copy(weights.flatten())

    x0 = np.concatenate((prob_init, weights_init))

    return x0

def init_bounds(probabilities_params_count, weights_param_count, fixed_values=None):
    """
    Set bound for the probability and state weights
    """
    prob_bounds = [(0.01, 0.99)] * probabilities_params_count 
    weights_bounds = [(0, 1)] * weights_param_count
    
    if fixed_values:
        fixed_lambdas = fixed_values.get('lambdas')
        if fixed_lambdas is not None:
            prob_bounds = [ o if f is None else (f, f) for o, f in itertools.zip_longest(prob_bounds, fixed_lambdas.flatten())]
        fixed_weights = fixed_values.get('weights')
        if fixed_weights is not None:
            weights_bounds = [ o if f is None else (f, f) for o, f in itertools.zip_longest(weights_bounds, fixed_weights.flatten())]

    bounds = np.concatenate((prob_bounds, weights_bounds), axis=0)  

    return bounds

def lambdas_hyper_default(non_nan_mask, number_of_states):
    non_nan_indices = non_nan_mask.sum()
    param_count = non_nan_indices * number_of_states
    func = lambda p: p

    return func, param_count

def weight_hyperparams(shape, number_of_states):
    if shape == 'symmetric':
        param_function = triangle_to_symmetric
        param_count = np.tril_indices(number_of_states)[0].size
        init_values = normalize(get_lower_triangle(np.eye(number_of_states), k=0))
    elif shape == 'diag':
        param_function = lambda n, v: np.diag(v)
        param_count = number_of_states
        init_values = normalize(np.ones(number_of_states))
    elif shape == 'binary' or shape == 'eye':
        param_function = lambda n, v: np.eye(n)
        param_count = 1 # should be 0, but later code doesn't handle that well
        init_values = np.array([0])
    elif shape == 'eye_with_empty':
        param_function = lambda n, v:  np.diag(np.concatenate([[0], np.ones(n-1)]))
        param_count = 1 # should be 0, but later code doesn't handle that well
        init_values = np.array([0])
    else:
        raise ValueError("Invalid weight shape")

    return param_function, param_count, init_values

def sort_weights(weights):
    self_weights = np.diag(weights)
    M = self_weights.size
    weights_order = np.argsort(self_weights)
    sorted_weights = np.empty_like(weights)
    for i in range(M):
        w_i = weights_order[i]
        for j in range(M):
            w_j = weights_order[j]
            sorted_weights[i, j] = weights[w_i, w_j]
    return sorted_weights, weights_order

def regularization_l1diff(state_probabilities, state_weights, cis_dd_power, trans_dd, cis_lengths, non_nan_mask):
    # The likelihood has O(N^2) elements, and this sum has only O(N). To make them increase at the same rate, we multiply by N
    return state_probabilities.shape[0] * np.sum(np.abs(state_probabilities[1:] - state_probabilities[:-1]))

def regularization_l2diff(state_probabilities, state_weights, cis_dd_power, trans_dd, cis_lengths, non_nan_mask):
    # The likelihood has O(N^2) elements, and this sum has only O(N). To make them increase at the same rate, we multiply by N
    return state_probabilities.shape[0] * np.linalg.norm(state_probabilities[1:] - state_probabilities[:-1], ord=2)

def regularization_nonuniform(state_probabilities, state_weights, cis_dd_power, trans_dd, cis_lengths, non_nan_mask):
    # The likelihood has O(N^2) elements, and this sum has only O(N). To make them increase at the same rate, we multiply by N
    # The minus sign is because we want larger values of this number (i.e. further than 0.5 as possible)
    return -state_probabilities.shape[0] * np.sum(np.abs(state_probabilities - 0.5))

def regularization_empty(*args):
    return 0

REGULARIZATIONS = dict(l1diff=regularization_l1diff, l2diff=regularization_l2diff, nonuniform=regularization_nonuniform)

def fit(interactions_mat, cis_lengths=None, number_of_states=2, weights_shape='diag', lambdas_hyper=None,
        init_values={}, fixed_values={}, optimize_options={}, R=0, regularization=None):
    """
    Return the model parameters that best explain the given Hi-C interaction matrix using L-BFGS-B.

    :param array interaction_mat: the Hi-C interaction matrix as a numpy array
    :param array cis_lengths:   Optional list of lengths of cis-interacting blocks (chromosomes). If not passed,
                                all bins are considered cis-interacting.
    :param int number_of_states: the number of possible states (compartments) for each bin
    :param str weights_shape: how the weights matrix should look. can be: 'symmetric', 'diag', 'eye', 'eye_with_empty'
    :return: A tuple of each bin's state probability (shape BINSxSTATES), state-state interaction
             probabiity (shape STATES-STATES), the cis distance-decay power value and the trans decay value (scalrs)
    :rtype: tuple
    """
    _cis_lengths = cis_lengths if cis_lengths is not None else [interactions_mat.shape[0]]
    unique_interactions = get_lower_triangle(remove_unusable_bins(preprocess(interactions_mat)))
    non_nan_mask = ~np.isnan(interactions_mat).all(1)
    del interactions_mat

    _lambdas_hyper = lambdas_hyper if lambdas_hyper is not None else lambdas_hyper_default
    lambdas_function, probabilities_params_count = _lambdas_hyper(non_nan_mask, number_of_states)
    weights_function, weights_param_count, weight_init = weight_hyperparams(weights_shape, number_of_states)

    x0 = np.concatenate([
        init_variables(probabilities_params_count, weight_init, init_values),
        distance_decay_model.init_variables(init_values)
    ])
    bounds = np.concatenate([
        init_bounds(probabilities_params_count, weights_param_count, fixed_values),
        distance_decay_model.init_bounds(fixed_values)
    ])
    optimize_options_defaults = dict(disp=True, ftol=1.0e-9, gtol=1e-9, eps=1e-9, maxfun=10000000, maxiter=10000000, maxls=100)
    _optimize_options = { **optimize_options_defaults, **optimize_options }

    log_likelihood = log_likelihood_by(unique_interactions)
    del unique_interactions
    if callable(regularization):
        _regularization_func = regularization
    elif regularization is None:
        _regularization_func = regularization_empty
    else:
        _regularization_func = REGULARIZATIONS[regularization]
    def likelihood_minimizer(variables):
        model_params = extract_params(variables, probabilities_params_count, weights_param_count, number_of_states,
                weights_function, lambdas_function, _cis_lengths, non_nan_mask)
        model_interactions = log_interaction_probability(*model_params)
        return -log_likelihood(model_interactions) + R * _regularization_func(*model_params)

    result = sp.optimize.minimize(fun=value_and_grad(likelihood_minimizer), x0=x0, method='L-BFGS-B', jac=True, 
            bounds=bounds, options=_optimize_options) 

    model_probabilities, model_weights, cis_dd_power, trans_dd, *_ = extract_params(result.x, probabilities_params_count, weights_param_count,
            number_of_states, weights_function, lambdas_function, _cis_lengths, non_nan_mask)
    expanded_model_probabilities = expand_by_mask(model_probabilities, non_nan_mask)

    sorted_weights, weights_order = sort_weights(model_weights)
    sorted_probabilities = expanded_model_probabilities[:, weights_order]
    
    return sorted_probabilities, sorted_weights, cis_dd_power, trans_dd, result

def generate_interactions_matrix(state_probabilities, state_weights, cis_dd_power, trans_dd, cis_lengths=None):
    """
    Generate an NxN interactions matrix by using the given model parameters. Diagonal values will be
    NaN as they're aren't modeled, along with any bins that have NaN in their state probabilities.

    :param array state_probabilities: NxM matrix, N being the bin count and M the number of states
    :param array state_weights: MxM symmetric matrix for state-state interaction probability
    :param number cis_dd_power: The rate by which cis interaction rate decreases with distance
    :param number trans_dd:     The constant that determines the relative strength of trans interactions
    :param array cis_lengths:   Optional list of lengths of cis-interacting blocks (chromosomes). If not passed,
                                all bins are considered cis-interacting.
    :return: interaction matrix generated by the model
    :rtype: NxN array
    """
    _cis_lengths = cis_lengths if cis_lengths is not None else [state_probabilities.shape[0]]
    number_of_bins = np.sum(_cis_lengths)
    non_nan_mask = np.ones(number_of_bins, dtype=bool)
    log_interactions = log_interaction_probability(state_probabilities, state_weights, cis_dd_power, trans_dd,
            _cis_lengths, non_nan_mask)
    interactions_vec = nannormalize(np.exp(log_interactions))
    interactions_mat = remove_main_diag(triangle_to_symmetric(number_of_bins, interactions_vec, k=-1, fast=True))

    return interactions_mat

from __future__ import division
import autograd.numpy as np
from autograd import grad
import functools
import scipy as sp
from scipy import optimize

from hic_analysis import preprocess, remove_unusable_bins, zeros_to_nan

def calculate_likelihood(interactions, non_nan_indices, number_of_states, probabilities_params_count, weights_param_count,
        number_of_bins, weights_function, variables):
    """
    Caulculate the log-likelihood of a given interaction (in interactions) using a model with a given number of states and
    with the current values for all model parameters (state probabilities, distance decay power value, etc)

    :return: The minus of the log-likelihood, to be used by minimization optimization algorithms.
    """
    prob_vars, weights_vars, dd_power = extract_variables(probabilities_params_count, weights_param_count, variables)
    lower_triangle_indices = np.tril_indices(number_of_bins, -1)

    distance_mat = np.absolute(non_nan_indices[None].T - non_nan_indices) + np.diag(np.full(number_of_bins, np.nan))
    distance_vec = distance_mat[lower_triangle_indices]

    state_probabilities_mat = normalize(prob_vars.reshape(number_of_bins, number_of_states), normalize_axis=1)
    state_weights_mat = normalize(weights_function(number_of_states, weights_vars))
    state_interactions_model = state_probabilities_mat @ state_weights_mat @ state_probabilities_mat.T
    state_interactions_vec = state_interactions_model[lower_triangle_indices]

    model_interactions_vec = (distance_vec ** dd_power) * state_interactions_vec
    probability_normalizing_constant = np.sum(model_interactions_vec)

    log_likelihood = np.dot(interactions, np.log(model_interactions_vec/probability_normalizing_constant))

    return -log_likelihood

def get_unique_interactions(interaction_mat):
    """
    Return the lower triangle of the given matrix as a vector, resulting in a vector of only unique
    interactions (as the upper triangle is symmetric to the lower)

    :param array interaction_mat: the interaction matrix, preferably after preprocessing
    :return: A vector of the lower triangle's values
    :rtype: 1d array
    """
    number_of_bins = np.size(interaction_mat, 0)
    lower_triangle_indices = np.tril_indices(number_of_bins, -1)
    value_vec = interaction_mat[lower_triangle_indices]

    return value_vec

def init_variables(probabilities_params_count, weights_param_count):
    """
    Give an intial value to all the variables we optimize.
    """
    prob_init = (np.random.rand(probabilities_params_count) / 2) + 0.25
    weights_init = normalize(np.random.rand(weights_param_count))
    dd_init = (-1,)
    x0 = np.concatenate((prob_init, weights_init, dd_init))

    return x0

def init_bounds(probabilities_params_count, weights_param_count):
    """
    Set bound for the probability, state weights and distance decay power variables
    """
    prob_bounds = [(0.01, 0.99)] * probabilities_params_count 
    weights_bounds = [(0.01, 0.99)] * weights_param_count
    dd_bounds = [(-2, -0.5)]
    bounds = np.concatenate((prob_bounds, weights_bounds, dd_bounds), axis=0)  

    return bounds

def extract_variables(probabilities_params_count, weights_param_count, variables):
    """
    Extract the three groups of variables from a single 1D vector
    """
    weights_start_idx = probabilities_params_count
    weights_end_idx = weights_start_idx + weights_param_count
    dd_idx = weights_end_idx

    prob_vars = variables[:probabilities_params_count]
    weights_vars = variables[weights_start_idx:weights_end_idx]
    dd_var = variables[dd_idx]

    return prob_vars, weights_vars, dd_var

def normalize(array, normalize_axis=None):
    """
    Normalize the given array by dividing by array.sum(axis=normalize_axis). This is used to get the
    'real' values of values that have some normalization constraint (such as probabilities). We do 
    this to bypass a L-BFGS-B limitation that only supports bound contraints.
    """
    normalization_factor = array.sum(axis=normalize_axis)[None].T # To make broadcasting work

    return array/normalization_factor

def triangle_to_symmetric(matrix_size, tri_values):
    """
    Convert the lower triangle values given by tri_values to a symmetric matrix with the given size
    """
    # This implementation is a bit more complicated than expected, because autograd doesn't support
    # array assignments (so symmat[x,y] = symmat[y,x] = values won't work). Instead we build the matrix one
    # column at a time
    x, y = np.tril_indices(matrix_size)
    def get_values(index):
        return tri_values[(x == index) | (y == index)]

    symmat = np.array([get_values(i) for i in range(matrix_size)])
    return symmat

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

def fit(interactions_mat, number_of_states=2, weights_shape='symmetric'):
    """
    Return the model parameters that best explain the given Hi-C interaction matrix using L-BFGS-B.

    :param array interaction_mat: the Hi-C interaction matrix as a numpy array
    :param int number_of_states: the number of possible states (compartments) for each bin
    :param str weights_shape: how the weights matrix should look. can be: 'symmetric', 'diag', 'eye', 'eye_with_empty'
    :return: A tuple of each bin's state probability (shape BINSxSTATES), state-state interaction
             probabiity (shape STATES-STATES) and the distance-decay power value (scalar)
    :rtype: tuple
    """
    original_number_of_bins = np.size(interactions_mat, 0) 
    clean_interactions_mat = remove_unusable_bins(preprocess(interactions_mat))
    new_number_of_bins = np.size(clean_interactions_mat, 0)
    unique_interactions = get_unique_interactions(clean_interactions_mat)
    non_nan_mask = ~np.isnan(interactions_mat).all(1)
    non_nan_indices = np.where(non_nan_mask)[0]

    probabilities_params_count = new_number_of_bins * number_of_states
    weights_function, weights_param_count = weight_hyperparams(weights_shape, number_of_states)
    distance_decay_param_count = 1

    x0 = init_variables(probabilities_params_count, weights_param_count)
    bounds = init_bounds(probabilities_params_count, weights_param_count)
    optimize_options = dict(disp=True, ftol=1.0e-10, gtol=1e-010, eps=1e-10, maxfun=100000, maxiter=100000, maxls=100)

    likelihood_with_model = functools.partial(calculate_likelihood, unique_interactions, non_nan_indices,
            number_of_states, probabilities_params_count, weights_param_count, new_number_of_bins, weights_function)
    likelihood_grad = grad(likelihood_with_model)
    res = sp.optimize.minimize(fun=likelihood_with_model, x0=x0, method='L-BFGS-B', jac=likelihood_grad, bounds=bounds,
            options=optimize_options)

    prob_vars, weights_vars, model_dd_power = extract_variables(probabilities_params_count, weights_param_count, res.x)
    model_probabilities = normalize(prob_vars.reshape(new_number_of_bins, number_of_states), normalize_axis=1)
    model_state_weights = normalize(weights_function(number_of_states, weights_vars))
    expanded_model_probabilities = np.full((original_number_of_bins, number_of_states), np.nan)
    expanded_model_probabilities[non_nan_indices, :] = model_probabilities
    
    return expanded_model_probabilities, model_state_weights, model_dd_power

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
    compartments_interactions = state_probabilities.dot(state_weights).dot(state_probabilities.T)

    distance_matrix = 1.0*np.absolute(np.arange(number_of_bins)[None].T - np.arange(number_of_bins))
    np.fill_diagonal(distance_matrix, np.nan)
    distance_decay_interactions = distance_matrix ** distance_decay_power
    unnormalized_all_interactions = distance_decay_interactions * compartments_interactions
    normalization_factor = np.nansum(np.tril(unnormalized_all_interactions, k=-1))

    return unnormalized_all_interactions / normalization_factor

from __future__ import division
import autograd.numpy as np
from autograd import value_and_grad
import scipy as sp
from scipy import optimize
import itertools
import os

from hic_analysis import preprocess, remove_unusable_bins, zeros_to_nan
from array_utils import get_lower_triangle, normalize, nannormalize, triangle_to_symmetric, remove_main_diag
from model_utils import log_likelihood_by, expand_by_mask, logsumexp
import loglikelihood
import distance_decay_model

def tap(f, debug):
    def _tap(x):
        r = f(x)
        if not callable(debug):
            breakpoint()
        else:
            debug(r, x)
        return r
    return _tap

def compartments_interactions(state_probabilities, cis_weights, trans_weights, cis_trans_mask):
    cis_interactions = get_lower_triangle(state_probabilities @ cis_weights @ state_probabilities.T)
    trans_interactions = get_lower_triangle(state_probabilities @ trans_weights @ state_probabilities.T)
    combined_interactions = np.where(cis_trans_mask, cis_interactions, trans_interactions)
    return combined_interactions


def log_interaction_probability(state_probabilities, cis_weights, trans_weights, cis_dd_power, trans_dd, cis_lengths,
        non_nan_mask, resolution):
    cis_trans_mask = distance_decay_model.cis_trans_mask(cis_lengths, non_nan_mask)
    compartments = np.log(compartments_interactions(state_probabilities, cis_weights, trans_weights, cis_trans_mask))
    distance_decay = distance_decay_model.log_distance_decay(cis_lengths, non_nan_mask, cis_dd_power, trans_dd, resolution)
    return compartments + distance_decay

def calc_logp(lambdas, cis_weights, trans_weights, alpha, beta, bin1_id, bin2_id, chr_assoc, non_nan_map):
    bincount = bin1_id.shape[0]
    nstates = cis_weights.shape[0]

    def a(row):
        i = bin1_id[row]
        j = bin2_id[row]
        i_nn = non_nan_map[i]
        j_nn = non_nan_map[j]
        if i == j or i_nn < 0 or j_nn < 0:
            return np.nan
        # Check cis/trans
        gc = 0
        if chr_assoc[i] == chr_assoc[j]:
            dd = alpha * np.log(j - i) # We iterate over the upper triangle, so i < j.
            for s1 in range(nstates):
                for s2 in range(nstates):
                    gc += lambdas[i_nn, s1] * lambdas[j_nn, s2] * cis_weights[s1, s2]
        else:
            dd = beta
            for s1 in range(nstates):
                for s2 in range(nstates):
                    gc += lambdas[i_nn, s1] * lambdas[j_nn, s2] * trans_weights[s1, s2]
        gc = np.log(gc)
        return dd + gc

    ret = np.array([ a(row) for row in range(bincount) ])
    return ret[np.isfinite(ret)]

def extract_params(variables, probabilities_params_count, cis_weights_param_count, trans_weights_param_count, number_of_states,
        cis_weights_function, trans_weights_function, lambdas_function, cis_lengths, non_nan_mask, resolution):
    """
    Slice and transform the given 1-D variable vector into the model parameters
    """
    cis_weights_start_idx, cis_weights_end_idx,\
            trans_weights_end_idx = np.cumsum([probabilities_params_count, cis_weights_param_count, trans_weights_param_count])
    # If trans_weights has no params, it means it should be the same as cis_weights
    if trans_weights_param_count == 0:
        trans_weights_start_idx = cis_weights_start_idx
        trans_weights_function = cis_weights_function
    else:
        trans_weights_start_idx = cis_weights_end_idx

    prob_vars = variables[:probabilities_params_count]
    cis_weights_vars = variables[cis_weights_start_idx:cis_weights_end_idx]
    trans_weights_vars = variables[trans_weights_start_idx:trans_weights_end_idx]
    if not cis_weights_vars.any():
        # make all-zero weights behave simply as all-equal weights
        cis_weights_vars = np.ones(cis_weights_end_idx - cis_weights_start_idx)
    if not trans_weights_vars.any():
        # make all-zero weights behave simply as all-equal weights
        trans_weights_vars = np.ones(trans_weights_end_idx - trans_weights_start_idx)
    alpha, beta = variables[-2:]

    lambdas = normalize(lambdas_function(prob_vars).reshape(-1, number_of_states), normalize_axis=1)
    cis_weights = normalize(cis_weights_function(number_of_states, cis_weights_vars))
    trans_weights = normalize(trans_weights_function(number_of_states, trans_weights_vars))

    return lambdas, cis_weights, trans_weights, alpha, beta, cis_lengths, non_nan_mask, resolution

def override_value(value, override, flatten_function=np.ndarray.flatten):
    """
    Override the initial value of an array parameter. If override has more than one dimension, it is
    first flattened using flatten_function.
    """
    if override is None:
        return value
    if np.ndim(override) > 1:
        override = flatten_function(override)
    assert np.size(override) == np.size(value), f"Override should have {np.size(value)} elements"

    return override

def fix_values(bounds, fixed_values, flatten_function=np.ndarray.flatten):
    """
    Fix values of an array parameter by changing its bounds in every index where fixed_values is not None.
    If fixed_values has more than one dimension it is first flattened using flatten_function.

    Returns the new bounds unchanged if fixed_values is None.
    """
    if fixed_values is None:
        return bounds
    if np.ndim(fixed_values) > 1:
        fixed_values = flatten_function(fixed_values)
    return np.array([ b if f is None else (f, f) for b, f in itertools.zip_longest(bounds, fixed_values)])

def lambdas_get_hyperparams(non_nan_mask, number_of_states, lambdas_hyper=None):
    if lambdas_hyper is None:
        hyperparams = dict()
    else:
        hyperparams = lambdas_hyper(non_nan_mask, number_of_states)
    non_nan_indices = non_nan_mask.sum()
    hyperparams.setdefault('param_count', non_nan_indices * number_of_states)
    # param_function: convert a flattened list of params to a flattened lambdas array, used by extract_params()
    hyperparams.setdefault('param_function', lambda p: p)
    hyperparams.setdefault('init_values', (np.random.rand(hyperparams['param_count']) / 2) + 0.25)
    hyperparams.setdefault('bounds', [(0.01, 0.99)] * hyperparams['param_count'] )
    # flatten_function: converts a multi-dimensional array to a list of params, used to more easily init or fix values
    hyperparams.setdefault('flatten_function', lambda a: a[non_nan_mask].flatten() )

    return hyperparams

def weights_get_hyperparams(shape, number_of_states):
    if shape == 'symmetric':
        param_function = triangle_to_symmetric
        param_count = np.tril_indices(number_of_states)[0].size
        init_values = normalize(get_lower_triangle(np.eye(number_of_states), k=0))
        flatten_function = lambda a: get_lower_triangle(a, k=0)
    elif shape == 'diag':
        param_function = lambda n, v: np.diag(v)
        param_count = number_of_states
        init_values = normalize(np.ones(number_of_states))
        flatten_function = np.diag
    elif shape == 'binary' or shape == 'eye':
        param_function = lambda n, v: np.eye(n)
        param_count = 1 # should be 0, but later code doesn't handle that well
        init_values = np.array([0])
        flatten_function = lambda x: x
    elif shape == 'eye_with_empty':
        param_function = lambda n, v:  np.diag(np.concatenate([[0], np.ones(n-1)]))
        param_count = 1 # should be 0, but later code doesn't handle that well
        init_values = np.array([0])
        flatten_function = lambda x: x
    else:
        raise ValueError("Invalid weight shape")

    bounds = [(0, 1)] * param_count
    return dict(param_function=param_function, param_count=param_count, init_values=init_values, bounds=bounds,
            flatten_function=flatten_function)

def weights_get_trans_hyperparams(shape, number_of_states):
    if shape == 'none':
        param_function = lambda n, v:  None
        param_count = 0
        init_values = np.array([])
        bounds = []
        flatten_function = lambda x: x

        return dict(param_function=param_function, param_count=param_count, init_values=init_values, bounds=bounds,
                flatten_function=flatten_function)

    return weights_get_hyperparams(shape, number_of_states)

def checkpoint_get_latest(checkpoint_dir):
    max_iter = -np.inf
    for f in os.listdir(checkpoint_dir):
        try:
            noext = int(os.path.splitext(f)[0])
            iter_num = int(noext)
        except ValueError:
            continue
        if iter_num > max_iter:
            max_iter = iter_num
    return max_iter

def checkpoint_load(checkpoint_dir, iter_num):
    checkpoint_path = os.path.join(checkpoint_dir, f"{iter_num}.npz")
    params = np.load(checkpoint_path)

    return params['x']

def checkpoint_restore_from_dir(checkpoint_dir, x0_beginning):
    iter_num = checkpoint_get_latest(checkpoint_dir)
    if not np.isfinite(iter_num):
        return 0, x0_beginning

    return iter_num+1, checkpoint_load(checkpoint_dir, iter_num)

def sort_weights(weights, order=None):
    self_weights = np.diag(weights)
    M = self_weights.size
    if order is None:
        weights_order = np.argsort(self_weights)
    else:
        weights_order = order
    sorted_weights = np.empty_like(weights)
    for i in range(M):
        w_i = weights_order[i]
        for j in range(M):
            w_j = weights_order[j]
            sorted_weights[i, j] = weights[w_i, w_j]
    return sorted_weights, weights_order

def regularization_l1diff(state_probabilities, *args):
    # The likelihood has O(N^2) elements, and this sum has only O(N). To make them increase at the same rate, we multiply by N
    return state_probabilities.shape[0] * np.sum(np.abs(state_probabilities[1:] - state_probabilities[:-1]))

def regularization_l2diff(state_probabilities, *args):
    # The likelihood has O(N^2) elements, and this sum has only O(N). To make them increase at the same rate, we multiply by N
    return state_probabilities.shape[0] * np.linalg.norm(state_probabilities[1:] - state_probabilities[:-1], ord=2)

def regularization_nonuniform(state_probabilities, *args):
    # The likelihood has O(N^2) elements, and this sum has only O(N). To make them increase at the same rate, we multiply by N
    # The minus sign is because we want larger values of this number (i.e. further than 0.5 as possible)
    return -state_probabilities.shape[0] * np.sum(np.abs(state_probabilities - 0.5))

def regularization_empty(*args):
    return 0

REGULARIZATIONS = dict(l1diff=regularization_l1diff, l2diff=regularization_l2diff, nonuniform=regularization_nonuniform)

def fit(interactions_mat, cis_lengths=None, number_of_states=2, cis_weights_shape='symmetric', trans_weights_shape='symmetric', lambdas_hyper=None,
        init_values={}, fixed_values={}, optimize_options={}, R=0, regularization=None, resolution=1, debug=False):
    """
    Return the model parameters that best explain the given Hi-C interaction matrix using L-BFGS-B.

    :param array interaction_mat: the Hi-C interaction matrix as a numpy array
    :param array cis_lengths:   Optional list of lengths of cis-interacting blocks (chromosomes). If not passed,
                                all bins are considered cis-interacting.
    :param int number_of_states: the number of possible states (compartments) for each bin
    :param str cis_weights_shape: how the cis weights matrix should look. can be: 'symmetric', 'diag', 'eye', 'eye_with_empty'
    :param str trans_weights_shape: how the trans weights matrix should look. Same values as cis_weights_shape + 'none' to use only the cis matrix
    :return: A tuple of each bin's state probability (shape BINSxSTATES), state-state interaction
             probabiity (shape STATES-STATES), the cis distance-decay power value and the trans decay value (scalrs)
    :rtype: tuple
    """
    _cis_lengths = cis_lengths if cis_lengths is not None else [interactions_mat.shape[0]]
    unique_interactions = get_lower_triangle(remove_unusable_bins(preprocess(interactions_mat)))
    non_nan_mask = ~np.isnan(interactions_mat).all(1)
    del interactions_mat

    lambdas_hyperparams = lambdas_get_hyperparams(non_nan_mask, number_of_states, lambdas_hyper)
    cis_weights_hyperparams = weights_get_hyperparams(cis_weights_shape, number_of_states)
    trans_weights_hyperparams = weights_get_trans_hyperparams(trans_weights_shape, number_of_states)
    model_state = (lambdas_hyperparams['param_count'], cis_weights_hyperparams['param_count'], trans_weights_hyperparams['param_count'],
            number_of_states, cis_weights_hyperparams['param_function'], trans_weights_hyperparams['param_function'],
            lambdas_hyperparams['param_function'], _cis_lengths, non_nan_mask, resolution)

    x0 = np.concatenate(filter(np.size, [
        override_value(lambdas_hyperparams['init_values'], init_values.get('state_probabilities'), lambdas_hyperparams['flatten_function']),
        override_value(cis_weights_hyperparams['init_values'], init_values.get('cis_weights'), cis_weights_hyperparams['flatten_function']),
        override_value(trans_weights_hyperparams['init_values'], init_values.get('trans_weights'), trans_weights_hyperparams['flatten_function']),
        distance_decay_model.init_variables(init_values)
    ]))
    bounds = np.concatenate(filter(np.size, [
        fix_values(lambdas_hyperparams['bounds'], fixed_values.get('state_probabilities'), lambdas_hyperparams['flatten_function']),
        fix_values(cis_weights_hyperparams['bounds'], fixed_values.get('cis_weights'), cis_weights_hyperparams['flatten_function']),
        fix_values(trans_weights_hyperparams['bounds'], fixed_values.get('trans_weights'), trans_weights_hyperparams['flatten_function']),
        distance_decay_model.init_bounds(fixed_values)
    ]))
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
        model_params = extract_params(variables, *model_state)
        model_interactions = log_interaction_probability(*model_params)
        return -log_likelihood(model_interactions) + R * _regularization_func(*model_params)

    if debug:
        vg = tap(value_and_grad(likelihood_minimizer), debug)
    else:
        vg = value_and_grad(likelihood_minimizer)
    result = sp.optimize.minimize(fun=vg, x0=x0, method='L-BFGS-B', jac=True,
            bounds=bounds, options=_optimize_options)

    model_probabilities, cis_weights, trans_weights, cis_dd_power, trans_dd, *_ = extract_params(result.x, *model_state)
    expanded_model_probabilities = expand_by_mask(model_probabilities, non_nan_mask)

    sorted_cis_weights, weights_order = sort_weights(cis_weights)
    sorted_trans_weights, _ = sort_weights(trans_weights, weights_order)
    sorted_probabilities = expanded_model_probabilities[:, weights_order]

    return sorted_probabilities, sorted_cis_weights, sorted_trans_weights, cis_dd_power, trans_dd, result

def fit_sparse(mat_dict, cis_lengths, number_of_states=2, cis_weights_shape='symmetric', trans_weights_shape='symmetric', lambdas_hyper=None,
        init_values={}, fixed_values={}, optimize_options={}, resolution=None, z_const_idx=None, z_count=0, dups='keep', cython=True, debug=False, checkpoint_dir=None, checkpoint_restore=True, nthreads=1):
    """""" # TODO: Use resolution param
    bins_i, bins_j, counts, non_nan_mask = mat_dict['bin1_id'], mat_dict['bin2_id'], mat_dict['count'], mat_dict.get('non_nan_mask')
    if non_nan_mask is None:
        non_nan_mask = np.ones(np.sum(cis_lengths), dtype=bool)
    non_nan_map = np.cumsum(non_nan_mask, dtype=int) - 1
    non_nan_map[~non_nan_mask] = -1
    chr_assoc = np.repeat(np.arange(np.size(cis_lengths)) , cis_lengths)

    lambdas_hyperparams = lambdas_get_hyperparams(non_nan_mask, number_of_states, lambdas_hyper)
    cis_weights_hyperparams = weights_get_hyperparams(cis_weights_shape, number_of_states)
    trans_weights_hyperparams = weights_get_trans_hyperparams(trans_weights_shape, number_of_states)
    model_state = (lambdas_hyperparams['param_count'], cis_weights_hyperparams['param_count'], trans_weights_hyperparams['param_count'],
            number_of_states, cis_weights_hyperparams['param_function'], trans_weights_hyperparams['param_function'],
            lambdas_hyperparams['param_function'], cis_lengths, non_nan_mask, None)

    x0 = np.concatenate(filter(np.size, [
        override_value(lambdas_hyperparams['init_values'], init_values.get('state_probabilities'), lambdas_hyperparams['flatten_function']),
        override_value(cis_weights_hyperparams['init_values'], init_values.get('cis_weights'), cis_weights_hyperparams['flatten_function']),
        override_value(trans_weights_hyperparams['init_values'], init_values.get('trans_weights'), trans_weights_hyperparams['flatten_function']),
        distance_decay_model.init_variables(init_values)
    ]))
    bounds = np.concatenate(filter(np.size, [
        fix_values(lambdas_hyperparams['bounds'], fixed_values.get('state_probabilities'), lambdas_hyperparams['flatten_function']),
        fix_values(cis_weights_hyperparams['bounds'], fixed_values.get('cis_weights'), cis_weights_hyperparams['flatten_function']),
        fix_values(trans_weights_hyperparams['bounds'], fixed_values.get('trans_weights'), trans_weights_hyperparams['flatten_function']),
        distance_decay_model.init_bounds(fixed_values)
    ]))
    optimize_options_defaults = dict(disp=True, ftol=1.0e-9, gtol=1e-9, eps=1e-9, maxfun=10000000, maxiter=10000000, maxls=100)
    _optimize_options = { **optimize_options_defaults, **optimize_options }

    if cython:
        nbins = non_nan_mask.sum()
        loglikelihood.preallocate(nbins, number_of_states, nthreads)
    else:
        counts_mask = np.isfinite(counts)
        log_likelihood = log_likelihood_by(counts[counts_mask])
        del counts_mask
        if z_const_idx is not None:
            z_const_idx_len = z_const_idx.shape[1]

    iter_count, x0 = checkpoint_restore_from_dir(checkpoint_dir, x0)

    def likelihood_minimizer(variables):
        nonlocal iter_count
        lambdas, cis_weights, trans_weights, alpha, beta, *_ = extract_params(variables, *model_state)

        if cython:
            ll = -loglikelihood.calc_likelihood(lambdas, cis_weights, trans_weights, alpha, beta, bins_i, bins_j, counts,
                    z_const_idx, z_count, chr_assoc, non_nan_map)
        else:
            model_interactions = calc_logp(lambdas, cis_weights, trans_weights, alpha, beta, bins_i, bins_j, chr_assoc, non_nan_map)
            if z_const_idx is not None:
                zeros_interactions = calc_logp(lambdas, cis_weights, trans_weights, alpha, beta, z_const_idx[0], z_const_idx[1], chr_assoc, non_nan_map)
                z_const = np.log(z_count / z_const_idx_len) + logsumexp(zeros_interactions)
            else:
                z_const = None
            ll = -log_likelihood(model_interactions, z_const)

        if checkpoint_dir:
            try:
                np.savez(f"{checkpoint_dir}/{iter_count}.npz",
                         x=variables._value,
                         lambdas=lambdas._value,
                         cis_weights=cis_weights._value,
                         trans_weights=trans_weights._value,
                         alpha=alpha._value,
                         beta=beta._value,
                         ll=ll._value)
            except Exception as e:
                print("Can't save checkpoint:", e)
        iter_count += 1

        return ll

    if debug:
        vg = tap(value_and_grad(likelihood_minimizer), debug)
    else:
        vg = value_and_grad(likelihood_minimizer)
    result = sp.optimize.minimize(fun=vg, x0=x0, method='L-BFGS-B', jac=True, 
            bounds=bounds, options=_optimize_options) 

    model_probabilities, cis_weights, trans_weights, cis_dd_power, trans_dd, *_ = extract_params(result.x, *model_state)
    expanded_model_probabilities = expand_by_mask(model_probabilities, non_nan_mask)

    sorted_cis_weights, weights_order = sort_weights(cis_weights)
    sorted_trans_weights, _ = sort_weights(trans_weights, weights_order)
    sorted_probabilities = expanded_model_probabilities[:, weights_order]
    
    return sorted_probabilities, sorted_cis_weights, sorted_trans_weights, cis_dd_power, trans_dd, result

def generate_interactions_matrix(state_probabilities, cis_weights, trans_weights, cis_dd_power, trans_dd,
        cis_lengths=None, resolution=1):
    """
    Generate an NxN interactions matrix by using the given model parameters. Diagonal values will be
    NaN as they're aren't modeled, along with any bins that have NaN in their state probabilities.

    :param array state_probabilities: NxM matrix, N being the bin count and M the number of states
    :param array cis_weights: MxM symmetric matrix for state-state interaction probability in cis
    :param array trans_weights: MxM symmetric matrix for state-state interaction probability in trans
    :param number cis_dd_power: The rate by which cis interaction rate decreases with distance
    :param number trans_dd:     The constant that determines the relative strength of trans interactions
    :param array cis_lengths:   Optional list of lengths of cis-interacting blocks (chromosomes). If not passed,
                                all bins are considered cis-interacting.
    :param number resolution:   Optional bin size, in base pairs. Used for resolution-independent cis_dd_power.
    :return: interaction matrix generated by the model
    :rtype: NxN array
    """
    _cis_lengths = cis_lengths if cis_lengths is not None else [state_probabilities.shape[0]]
    number_of_bins = np.sum(_cis_lengths)
    non_nan_mask = np.ones(number_of_bins, dtype=bool)
    log_interactions = log_interaction_probability(state_probabilities, cis_weights, trans_weights, cis_dd_power,
            trans_dd, _cis_lengths, non_nan_mask, resolution)
    interactions_vec = nannormalize(np.exp(log_interactions))
    interactions_mat = remove_main_diag(triangle_to_symmetric(number_of_bins, interactions_vec, k=-1, fast=True))

    return interactions_mat

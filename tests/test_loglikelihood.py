"""
Tests for the loglikelihood function as a whole, testing correct handling of zero bins, NaN bins, and different states.
Comparing against dense model + autograd.

I don't test for partial sampling of zeros here, since I haven't thought how to verify the results (they're expected to
be different than the dense model after all)

Other assumptions:
    - model parameters are not None
    - counts and indices are from the upper triangle, so i < j
    - counts in non negative
    - zero indices are from the upper triangle
    - if there are no zeros, zero indices is None
    - chr_assoc and non_nan_map are computed correctly
"""
import pytest
import numpy as np
from autograd import grad

import gc_model
import loglikelihood
from array_utils import triangle_to_symmetric, get_lower_triangle
from model_utils import log_likelihood_by
from utils import almost_equal

@pytest.fixture(params=[0, 0.3, 0.5], ids=lambda a: f"{a:.0%}nans")
def non_nan_mask(request, nbins):
    nan_number = int(nbins * request.param)
    np.random.seed(0)
    nan_indices = np.random.choice(nbins, size=nan_number, replace=False)
    mask = np.ones(nbins, dtype=bool)
    mask[nan_indices] = False

    return mask

@pytest.fixture()
def nn_bins(non_nan_mask):
    return non_nan_mask.sum()

@pytest.fixture()
def nn_lambdas(lambdas, non_nan_mask):
    return lambdas[non_nan_mask]

@pytest.fixture()
def non_nan_map(non_nan_mask):
    non_nan_map = np.cumsum(non_nan_mask, dtype=int) - 1
    non_nan_map[~non_nan_mask] = -1

    return non_nan_map

@pytest.fixture(params=[0, 0.1, 0.7, 1], ids=lambda a: f"{a:.0%}zeros")
def counts_mat(request, nbins, non_nan_mask):
    np.random.seed(0)
    flat_length = nbins * (nbins - 1) // 2
    nzeros = int(request.param * flat_length)
    counts = 0.1 + np.random.random(flat_length)
    zeros_indices = np.random.choice(flat_length, size=nzeros, replace=False)
    counts[zeros_indices] = 0
    mat = triangle_to_symmetric(nbins, counts, k=-1, fast=True)

    mat[~non_nan_mask] = np.nan
    mat[:, ~non_nan_mask] = np.nan
    np.fill_diagonal(mat, np.nan)
    return mat

@pytest.fixture()
def sparse_data(nbins, non_nan_mask, counts_mat):
    upper_tri_mask = ~np.tri(nbins, dtype=bool)
    bin1_id, bin2_id = np.ascontiguousarray(np.nonzero((counts_mat != 0) & upper_tri_mask), dtype=np.int32)
    count = np.ascontiguousarray(counts_mat[bin1_id,bin2_id])
    return dict(count=count, bin1_id=bin1_id, bin2_id=bin2_id)

@pytest.fixture()
def zeros_data(nbins, counts_mat):
    upper_tri_mask = ~np.tri(nbins, dtype=bool)
    indices = np.ascontiguousarray(np.nonzero((counts_mat == 0) & upper_tri_mask), dtype=np.int32)
    if indices.size == 0:
        return dict(count=0, indices=None)

    return dict(count=indices.shape[1], indices=indices)

@pytest.fixture()
def ref_ll_func(counts_mat, cis_lengths, non_nan_mask):
    ll = log_likelihood_by(get_lower_triangle(counts_mat[non_nan_mask, :][:, non_nan_mask]))
    ll_from_params = lambda l, w, a, b: ll(gc_model.log_interaction_probability(l, w, a, b, cis_lengths, non_nan_mask, 1))

    return ll_from_params

@pytest.fixture()
def reference_likelihood(ref_ll_func, nn_lambdas, weights, alpha, beta):
    return ref_ll_func(nn_lambdas, weights, alpha, beta)

@pytest.fixture(params=[1,2,3,4], ids=lambda x: f'{x}threads')
def nthreads(request):
    return request.param

@pytest.fixture(autouse=True)
def sparse_preallocate(nn_bins, nstates, nthreads):
    loglikelihood.preallocate(nn_bins, nstates, nthreads)

@pytest.fixture()
def sparse_ll_func(chr_assoc, sparse_data, zeros_data, non_nan_map):
    return lambda l, w, a, b: loglikelihood.calc_likelihood(l, w, a, b,
                                                            sparse_data['bin1_id'], sparse_data['bin2_id'],
                                                            sparse_data['count'], zeros_data['indices'],
                                                            zeros_data['count'], chr_assoc, non_nan_map)

@pytest.fixture()
def sparse_likelihood(sparse_ll_func, nn_lambdas, weights, alpha, beta):
    return sparse_ll_func(nn_lambdas, weights, alpha, beta)

@pytest.fixture(params=[0,1,2,3], ids=['lambdas', 'weights', 'alpha', 'beta'])
def grad_var(request):
    return request.param

@pytest.fixture()
def sparse_grad(sparse_ll_func, grad_var, nn_lambdas, weights, alpha, beta):
        return grad(sparse_ll_func, grad_var)(nn_lambdas, weights, alpha, beta)

@pytest.fixture()
def reference_grad(ref_ll_func, grad_var, nn_lambdas, weights, alpha, beta):
    if grad_var == 1:
        # Workaround: reference uses lower triangle, Cython uses upper triangle. Makes weight grad transpose
        return grad(ref_ll_func, grad_var)(nn_lambdas, weights, alpha, beta).T
    return grad(ref_ll_func, grad_var)(nn_lambdas, weights, alpha, beta)

def test_none_lambdas(weights, alpha, beta, chr_assoc, sparse_data):
    bin1_id = sparse_data['bin1_id']
    bin2_id = sparse_data['bin2_id']
    count = sparse_data['count']
    with pytest.raises(TypeError):
        loglikelihood.calc_likelihood(None, weights, alpha, beta, bin1_id, bin2_id, count,
                                      None, 0, chr_assoc, np.arange(5))

def test_none_weights(nn_lambdas, alpha, beta, chr_assoc, sparse_data):
    bin1_id = sparse_data['bin1_id']
    bin2_id = sparse_data['bin2_id']
    count = sparse_data['count']
    with pytest.raises(TypeError):
        loglikelihood.calc_likelihood(nn_lambdas, None, alpha, beta, bin1_id, bin2_id, count,
                                      None, 0, chr_assoc, np.arange(5))

def test_value(sparse_likelihood, reference_likelihood):
    almost_equal(sparse_likelihood, reference_likelihood)

def test_grad(reference_grad, sparse_grad):
    almost_equal(reference_grad, sparse_grad)

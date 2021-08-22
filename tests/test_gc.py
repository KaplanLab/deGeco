"""
Basic testing of the unnormalized P(i, j) calculation and Jacobian. Comparing against dense model and autograd.
"""
import pytest
import numpy as np
from numpy import int32

import loglikelihood
import gc_model
from distance_decay_model import cis_trans_mask as _cis_trans_mask
from array_utils import triangle_to_symmetric
from autograd import jacobian
from utils import almost_equal


@pytest.fixture()
def cis_trans_mask(cis_lengths):
    # Basic cis_trans_mask, we don't deal with nan masks here
    all_ones_mask = np.ones(np.sum(cis_lengths), dtype=bool)
    return _cis_trans_mask(cis_lengths, all_ones_mask)


def test_log_dd_cis(alpha, beta, chr_assoc):
    i = int32(0)
    j = int32(2)
    res = loglikelihood.calc_dd(i, j, alpha, beta, chr_assoc)
    assert res == alpha * np.log(np.abs(i - j))

def test_log_dd_trans(alpha, beta, chr_assoc):
    i = int32(0)
    j = int32(chr_assoc.size-1)
    res = loglikelihood.calc_dd(i, j, alpha, beta, chr_assoc)
    assert res == beta

def test_gc_cis(nbins, lambdas, weights, trans_weights):
    cis_trans_mask = np.zeros((nbins**2 - nbins) // 2) # all-cis cis_trans_mask
    gc_reference = triangle_to_symmetric(nbins, gc_model.compartments_interactions(lambdas, weights, weights, cis_trans_mask), fast=True, k=-1)
    gc_res = np.array([ [ loglikelihood.calc_gc(int32(i), int32(j), lambdas, weights) for j in range(nbins) ] for i in range(nbins) ])
    np.fill_diagonal(gc_res, 0)
    almost_equal(gc_res, gc_reference)

def test_gc_trans(nbins, chr_assoc, cis_trans_mask, lambdas, weights, trans_weights):
    gc_reference = triangle_to_symmetric(nbins, gc_model.compartments_interactions(lambdas, weights, trans_weights, cis_trans_mask), fast=True, k=-1)
    gc_res = np.array([ [ loglikelihood.calc_gc(int32(i), int32(j), lambdas, weights if chr_assoc[i] == chr_assoc[j] else trans_weights) for j in range(nbins) ] for i in range(nbins) ])
    np.fill_diagonal(gc_res, 0)
    almost_equal(gc_res, gc_reference)

def test_jac_log_dd_cis(chr_assoc):
    i = int32(0)
    j = int32(2)
    res_alpha, res_beta = loglikelihood.dd_jac_element(i, j, chr_assoc)
    assert res_alpha == np.log(np.abs(i - j))
    assert res_beta == 0

def test_jac_log_dd_trans(chr_assoc):
    i = int32(0)
    j = int32(chr_assoc.size-1)
    res_alpha, res_beta = loglikelihood.dd_jac_element(i, j, chr_assoc)
    assert res_alpha == 0
    assert res_beta == 1

def to_flatten_index(i, j, size):
    if j > i:
        i, j = j, i
    offset = (1 + i-1)*(i-1)//2
    return offset + j

@pytest.fixture()
def ref_jac_lambdas(nbins, lambdas, weights):
    return jacobian(gc_model.compartments_interactions)(lambdas, weights, weights, (nbins,))

def test_jac_lambdas(ref_jac_lambdas, nbins, nstates, lambdas, weights):
    flattened_length = (nbins**2 - nbins) // 2
    res = np.zeros((flattened_length, nbins, nstates))
    for i in range(nbins):
        for j in range(i):
            for s in range(nstates):
                a = to_flatten_index(i, j, nbins)
                res[a, i, s], res[a, j, s] = loglikelihood.lambdas_jac_element(int32(i), int32(j), s, lambdas, weights)

    almost_equal(res, ref_jac_lambdas)

@pytest.fixture()
def ref_jac_weights(nbins, lambdas, weights):
    return jacobian(gc_model.compartments_interactions, 1)(lambdas, weights, weights, (nbins,))

def test_jac_weights(ref_jac_weights, nbins, nstates, lambdas, weights):
    flattened_length = (nbins**2 - nbins) // 2
    res = np.zeros((flattened_length, nstates, nstates))
    for i in range(nbins):
        for j in range(i):
            for s1 in range(nstates):
                for s2 in range(nstates):
                    a = to_flatten_index(i, j, nbins)
                    res[a, s1, s2] = loglikelihood.weights_jac_element(int32(i), int32(j), s1, s2, lambdas)

    almost_equal(res, ref_jac_weights)

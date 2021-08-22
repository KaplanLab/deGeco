"""
Basic testing of the dense model's GC output
"""
import pytest
import numpy as np
from numpy import int32

import gc_model
from distance_decay_model import cis_trans_mask as _cis_trans_mask
from array_utils import get_lower_triangle

@pytest.fixture()
def cis_trans_mask(cis_lengths, non_nan_mask):
    return _cis_trans_mask(cis_lengths, non_nan_mask)

def test_no_trans(nn_lambdas, weights, nbins):
    res = gc_model.compartments_interactions(nn_lambdas, weights, weights, (nbins,))
    reference = get_lower_triangle(nn_lambdas @ weights @ nn_lambdas.T)

    assert np.all(res == reference)

def test_trans_weights_eq_cis_weights(nn_lambdas, weights, cis_trans_mask):
    res = gc_model.compartments_interactions(nn_lambdas, weights, weights, cis_trans_mask)
    reference = get_lower_triangle(nn_lambdas @ weights @ nn_lambdas.T)

    assert np.all(res == reference)

def test_trans_weights_not_eq_cis_weights(nn_lambdas, weights, trans_weights, cis_trans_mask):
    res = gc_model.compartments_interactions(nn_lambdas, weights, trans_weights, cis_trans_mask)
    cis_reference = get_lower_triangle(nn_lambdas @ weights @ nn_lambdas.T)
    trans_reference = get_lower_triangle(nn_lambdas @ trans_weights @ nn_lambdas.T)

    # Should be equal only in cis
    assert np.all(res[cis_trans_mask] == cis_reference[cis_trans_mask])
    # Should be equal only in trans
    assert np.all(res[~cis_trans_mask] == trans_reference[~cis_trans_mask])


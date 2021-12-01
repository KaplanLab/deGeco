import pytest
import numpy as np

import zero_sampler


def test_correct_gaps1():
    nbins = 4
    bin1_id = np.array([0, 0, 2, 2], dtype=np.int32)
    bin2_id = np.array([0, 2, 2, 3], dtype=np.int32)
    #non_nan_mask = np.ones(bin1_id.size, dtype=np.int8)
    z = zero_sampler.ZeroSampler(nbins, bin1_id, bin2_id)

    reference_holes = np.cumsum([0, 1, 4, 0, 1])
    holes = z.sample_zeros(4)
    
    assert np.all(holes == reference_holes)

def test_correct_gaps_before_first_pixel():
    nbins = 4
    bin1_id = np.array([0, 0, 2, 2], dtype=np.int32)
    bin2_id = np.array([1, 2, 2, 3], dtype=np.int32)
    #non_nan_mask = np.ones(bin1_id.size, dtype=np.int8)
    z = zero_sampler.ZeroSampler(nbins, bin1_id, bin2_id)

    reference_holes = np.cumsum([1, 0, 4, 0, 1])
    holes = z.sample_zeros(4)
    
    assert np.all(holes == reference_holes)

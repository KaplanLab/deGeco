import pytest
import numpy as np

import zero_sampler


def test_correct_gaps():
    nbins = 4
    bin1_id = np.array([0, 0, 2, 2], dtype=np.int32)
    bin2_id = np.array([0, 2, 2, 3], dtype=np.int32)
    #non_nan_mask = np.ones(bin1_id.size, dtype=np.int8)
    z = zero_sampler.ZeroSampler(nbins, bin1_id, bin2_id)

    reference_holes = [0, 1, 3, 0, 0]
    holes = z.sample_zeros(4)
    
    assert np.all(holes == reference_holes)


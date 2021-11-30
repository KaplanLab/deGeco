import pytest
import numpy as np

import zero_sampler


def test_correct_gaps():
    z = zero_sampler.ZeroSampler(3, np.array([0, 0, 2, 2], dtype=np.int32), np.array([0, 2, 1, 2], dtype=np.int32), np.ones(4, dtype=np.int8))
    holes = z.sample_zeros(4)
    
    assert np.all(holes == [0, 1, 4, 0, 0)])


import pytest
import numpy as np

import zero_sampler


class TestZeroSampler:
    def test_upper_triangle(self):
        nbins = 30
        bins_i = np.array([], dtype=np.int32)
        bins_j = np.array([], dtype=np.int32)
        nn_mask = np.array([1] * nbins, dtype='int8')
        sampler = zero_sampler.ZeroSampler(nbins, bins_i, bins_j, nn_mask)
        zeros = sampler.sample_zeros(40)

        assert (zeros[0] <= zeros[1]).all()

    def test_ignore_nan(self):
        nbins = 3
        bins_i = np.array([], dtype=np.int32)
        bins_j = np.array([], dtype=np.int32)
        nn_mask = np.array([1, 1, 0], dtype='int8')
        sampler = zero_sampler.ZeroSampler(nbins, bins_i, bins_j, nn_mask)
        zeros = sampler.sample_zeros(3)

        assert (zeros[0] <= zeros[1]).all()
        assert (zeros[0] != 2).all()
        assert (zeros[1] != 2).all()

    def test_ignore_nonzero(self):
        nbins = 3
        bins_i = np.array([0, 0, 0], dtype=np.int32)
        bins_j = np.array([0, 1, 2], dtype=np.int32)
        nn_mask = np.array([1, 1, 1], dtype='int8')
        sampler = zero_sampler.ZeroSampler(nbins, bins_i, bins_j, nn_mask)
        zeros = sampler.sample_zeros(3)

        assert (zeros[0] <= zeros[1]).all()
        assert (zeros[0] != 0).all()
        assert (zeros[1] != 0).all()

    def test_too_large_sample(self):
        nbins = 3
        bins_i = np.array([0, 0, 0], dtype=np.int32)
        bins_j = np.array([0, 1, 2], dtype=np.int32)
        nn_mask = np.array([1, 1, 1], dtype='int8')
        sampler = zero_sampler.ZeroSampler(nbins, bins_i, bins_j, nn_mask)
        zeros = sampler.sample_zeros(30)

        assert zeros.shape[1] == 3

    def test_very_large_request(self):
        # Test for obvious integer overflows
        nbins = 3
        bins_i = np.array([0, 0, 0], dtype=np.int32)
        bins_j = np.array([0, 1, 2], dtype=np.int32)
        nn_mask = np.array([1, 1, 1], dtype='int8')
        sampler = zero_sampler.ZeroSampler(nbins, bins_i, bins_j, nn_mask)
        zeros = sampler.sample_zeros(20000000000)

        assert zeros.shape[1] == 3

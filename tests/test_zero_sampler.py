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

    def test_trans_count(self):
        nbins = 30
        bins_i = np.array([], dtype=np.int32)
        bins_j = np.array([], dtype=np.int32)
        nn_mask = np.array([1] * nbins, dtype='int8')
        cis_lengths = (10, 10, 10)
        sampler = zero_sampler.ZeroSampler(nbins, bins_i, bins_j, nn_mask, cis_lengths, 'trans')

        assert sampler.zero_count == 10*20 + 10*10

    def test_trans_sample(self):
        nbins = 5
        bins_i = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        bins_j = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        nn_mask = np.array([1] * nbins, dtype='int8')
        cis_lengths = (2, 2, 1)
        sampler = zero_sampler.ZeroSampler(nbins, bins_i, bins_j, nn_mask, cis_lengths, 'trans')
        zeros = sampler.sample_zeros(5)

        assert zeros.shape[1] == 5
        assert np.all(zeros == [[1, 1, 1, 2, 3], [2, 3, 4, 4, 4]])

    def test_trans_count_nans(self):
        nbins = 3
        bins_i = np.array([], dtype=np.int32)
        bins_j = np.array([], dtype=np.int32)
        nn_mask = np.array([0, 1, 1], dtype='int8')
        cis_lengths = (2, 1)
        sampler = zero_sampler.ZeroSampler(nbins, bins_i, bins_j, nn_mask, cis_lengths, 'trans')

        assert sampler.zero_count == 1

    def test_trans_sample(self):
        nbins = 3
        bins_i = np.array([], dtype=np.int32)
        bins_j = np.array([], dtype=np.int32)
        nn_mask = np.array([0, 1, 1], dtype='int8')
        cis_lengths = (2, 1)
        sampler = zero_sampler.ZeroSampler(nbins, bins_i, bins_j, nn_mask, cis_lengths, 'trans')
        zeros = sampler.sample_zeros(5)

        assert zeros.shape[1] == 1
        assert np.all(zeros == [[1], [2]])

import pytest
import numpy as np

@pytest.fixture(params=[5], ids=lambda a: f"{a}bins")
def nbins(request):
    return request.param

@pytest.fixture()
def cis_lengths(nbins):
    assert nbins >= 5
    len_first = int(nbins * 0.8)
    len_second = nbins - len_first

    return np.array([len_first, len_second])

@pytest.fixture()
def chr_assoc(cis_lengths):
    labels = np.arange(len(cis_lengths))
    return np.repeat(labels, cis_lengths)

@pytest.fixture()
def alpha():
    return -1.0

@pytest.fixture()
def beta():
    return -1.5

@pytest.fixture(params=[2, 3], ids=lambda a: f"{a}st")
def nstates(request):
    return request.param

@pytest.fixture()
def lambdas(nbins, nstates):
    assert nbins == 5
    if nstates == 2:
        return np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5], [0.1, 0.9], [0.3, 0.7]])
    if nstates == 3:
        return np.array([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2], [0.25, 0.25, 0.5], [0.9, 0, 0.1], [0, 0, 1]])
    assert False

@pytest.fixture()
def weights(nstates):
    if nstates == 2:
        return np.array([[0.5, 0.1], [0.1, 0.3]])
    if nstates == 3:
        return np.array([[0.5, 0, 0.05], [0, 0.2, 0.03], [0.05, 0.03, 0.22]])
    assert False

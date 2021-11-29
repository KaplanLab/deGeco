import numpy as np
import pytest
import ssum
from utils import almost_equal

def test_fsum():
    array = np.array([1, 2e-9, 3e-9] * 1000000)
    result = 1000000 + 5e-3
    # This is a control, should fail
    with pytest.raises(AssertionError):
        naive_sum = sum(array)
        almost_equal(result, naive_sum)
    almost_equal(result, ssum.fsum(array))

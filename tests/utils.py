import numpy as np

def almost_equal(a, b, threshold=1e-10):
    assert (np.abs(a - b) < threshold).all()


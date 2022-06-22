import pytest
import numpy as np

import hic_analysis as hic

@pytest.fixture
def cooler_obj():
    return hic.cooler.Cooler("tests/fixtures/GM_chr1_chr2_500kb.cool")

@pytest.fixture(params=[0.1,0.3,0.5,0.7,1.0])
def end_bin(request, cooler_obj):
    bincount = cooler_obj.shape[0]
    return int(request.param * bincount)

def test_read_up_to_bin(end_bin, cooler_obj):
    reference = cooler_obj.matrix(as_pixels=True)[:end_bin, :end_bin]
    read_up_to = hic._read_square_up_to_bin(cooler_obj, end_bin)

    assert np.all(reference['bin1_id'] == read_up_to['bin1_id'])
    assert np.all(reference['bin2_id'] == read_up_to['bin2_id'])
    assert np.all(reference['count'] == read_up_to['count'])

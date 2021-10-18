"""
Basic testing of the Checkpoint class
"""
import pytest
import numpy as np
import time

import checkpoint

@pytest.fixture()
def checkpoint_not_saved():
    return checkpoint.Checkpoint("/some/filename")

@pytest.fixture()
def checkpoint_recently_saved():
    cp = checkpoint.Checkpoint("/some/filename")
    cp.last_saved = time.time() - cp.save_interval/2

    return cp

def test_persist_first_time(checkpoint_not_saved, mocker):
    mocker.patch('checkpoint.np.savez')
    checkpoint_not_saved.persist()

    checkpoint.np.savez.assert_called_once()

def test_persist_too_fast(checkpoint_recently_saved, mocker):
    mocker.patch('checkpoint.np.savez')
    checkpoint_recently_saved.persist()

    checkpoint.np.savez.assert_not_called()

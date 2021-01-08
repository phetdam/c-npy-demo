__doc__ = "Unit tests for :func:`c_npy_demo.cscale.stdscale`."

import numpy as np
import pytest

# pylint:disable=relative-beyond-top-level
from .. import cscale, pyscale


@pytest.fixture
def test_mat():
    "Test array we use for unit tests."
    return np.array([[1, 2], [3, 4], [5, 6]])


@pytest.mark.parametrize("ddof", [0, 1, 3, 5])
def test_cscale_allclose(test_mat, ddof):
    """Test that :func:`c_npy_demo.cscale.stdscale` uses ``ddof`` correctly.

    :param test_mat: ``pytest`` fixture for test matrix.
    :type test_mat: :class:`numpy.ndarray`
    :param ddof: Delta degrees of freedom for standard deviation calculations.
    :type ddof: int
    """
    np.testing.assert_allclose(
        pyscale.stdscale(test_mat), cscale.stdscale(test_mat)
    )
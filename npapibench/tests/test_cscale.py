"""Unit tests for cscale.stdscale.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest

# pylint: disable=relative-beyond-top-level
from .. import cscale, pyscale


def test_cscale_sanity(test_mat):
    """Test cscale.stdscale input checking sanity.

    Parameters
    ----------
    test_mat : numpy.ndarray
        pytest fixture. See local conftest.py.
    """
    # TypeError raised if missing required ar or if ar can't be converted
    with pytest.raises(TypeError):
        cscale.stdscale()
    with pytest.raises(TypeError):
        cscale.stdscale(ddof=1)
    with pytest.raises(TypeError):
        cscale.stdscale(test_mat.astype(str))
    # ValueError raised if ddof not nonnegative
    with pytest.raises(ValueError, match="ddof must be a nonnegative int"):
        cscale.stdscale(test_mat, ddof=-1)


def test_cscale_empty():
    """Test cscale.stdscale warn when ar is empty."""
    with pytest.warns(RuntimeWarning, match="mean of empty array"):
        cscale.stdscale(np.array([]))


@pytest.mark.parametrize("ddof", [0, 1, 3, 5])
def test_cscale_allclose(test_mat, ddof):
    """Test cscale.stdscale use of ddof and that it matches Python version.

    Parameters
    ----------
    test_mat : numpy.ndarray
        pytest fixture. See local conftest.py.
    ddof : int
        Delta degrees of freedom for standard deviation calculations.
    """
    np.testing.assert_allclose(
        pyscale.stdscale(test_mat, ddof=ddof),
        cscale.stdscale(test_mat, ddof=ddof)
    )
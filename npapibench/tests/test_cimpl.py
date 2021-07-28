"""Unit tests for cimpl.stdscale.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest

# pylint: disable=relative-beyond-top-level
from .. import cimpl, pyimpl


def test_cimpl_sanity(pytype_raise, pyvalue_raise, test_mat):
    """Test cimpl.stdscale input checking sanity.

    Parameters
    ----------
    pytype_raise : function
        pytest fixture. See top-level package conftest.py.
    pyvalue_raise : function
        pytest fixture. See top-level package conftest.py.
    test_mat : numpy.ndarray
        pytest fixture. See local conftest.py.
    """
    # TypeError raised if missing required ar or if ar can't be converted
    with pytype_raise():
        cimpl.stdscale()
    with pytype_raise():
        cimpl.stdscale(ddof=1)
    with pytype_raise():
        cimpl.stdscale(test_mat.astype(str))
    # ValueError raised if ddof not nonnegative
    with pyvalue_raise("ddof must be a nonnegative int"):
        cimpl.stdscale(test_mat, ddof=-1)


def test_cimpl_empty():
    """Test cimpl.stdscale warn when ar is empty."""
    with pytest.warns(RuntimeWarning, match="mean of empty array"):
        cimpl.stdscale(np.array([]))


@pytest.mark.parametrize("ddof", [0, 1, 3, 5])
def test_cimpl_allclose(test_mat, ddof):
    """Test cimpl.stdscale use of ddof and that it matches Python version.

    Parameters
    ----------
    test_mat : numpy.ndarray
        pytest fixture. See local conftest.py.
    ddof : int
        Delta degrees of freedom for standard deviation calculations.
    """
    np.testing.assert_allclose(
        pyimpl.stdscale(test_mat, ddof=ddof),
        cimpl.stdscale(test_mat, ddof=ddof)
    )
__doc__ = """Unit tests for c_npy_demo.cscale.stdscale.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest

# pylint: disable=relative-beyond-top-level
from .. import cscale, pyscale


@pytest.mark.parametrize("ddof_kwarg", [{}, {"ddof": 1}])
def test_cscale_missing_args(ddof_kwarg):
    """Test c_npy_demo.cscale.stdscale raise when missing ar.

    Parameters
    ----------
    ddof_kwarg : dict
        Dict to unpack for ddof parameter
    """
    with pytest.raises(TypeError):
        cscale.stdscale(**ddof_kwarg)


def test_cscale_ar_ndarray_type(test_mat):
    """Test c_npy_demo.cscale.stdscale raise when ar is not numeric.

    Parameters
    ----------
    test_mat : numpy.ndarray
        pytest fixture. See local conftest.py.
    """
    with pytest.raises(TypeError, match="ar must have dtype int or float"):
        cscale.stdscale(test_mat.astype(str))


def test_cscale_ddof(test_mat):
    """Test c_npy_demo.cscale.stdscale raise when ``ddof`` is negative.

    Parameters
    ----------
    test_mat : numpy.ndarray
        pytest fixture. See local conftest.py.
    """
    with pytest.raises(ValueError, match="ddof must be a nonnegative int"):
        cscale.stdscale(test_mat, ddof=-1)


def test_cscale_empty():
    "Test c_npy_demo.cscale.stdscale warn when ar is empty."
    with pytest.warns(RuntimeWarning, match="mean of empty array"):
        cscale.stdscale(np.array([]))


@pytest.mark.parametrize("ddof", [0, 1, 3, 5])
def test_cscale_allclose(test_mat, ddof):
    """Test c_npy_demo.cscale.stdscale use of ddof.

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
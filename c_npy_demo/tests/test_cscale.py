__doc__ = "Unit tests for :func:`c_npy_demo.cscale.stdscale`."

import numpy as np
import pytest

# pylint: disable=relative-beyond-top-level
from .. import cscale, pyscale


@pytest.fixture
def test_mat():
    "Test array we use for unit tests."
    return np.array([[1, 2], [3, 4], [5, 6]])


@pytest.mark.parametrize("ddof_kwarg", [{}, {"ddof": 1}])
def test_cscale_missing_args(ddof_kwarg):
    """Test :func:`c_npy_demo.cscale.stdscale` raise when missing ``ar``.

    :param ddof_kwarg: Dict to unpack for ``ddof`` parameter
    :type ddof_kwarg: dict
    """
    with pytest.raises(TypeError):
        cscale.stdscale(**ddof_kwarg)


def test_cscale_ar_ndarray():
    "Test :func:`c_npy_demo.cscale.stdscale` raise when ``ar`` is not array."
    with pytest.raises(TypeError):
        cscale.stdscale([[2, 4], [1, 2]])


def test_cscale_ar_ndarray_type(test_mat):
    """Test :func:`c_npy_demo.cscale.stdscale` raise when ``ar`` is not numeric.

    :param test_mat: ``pytest`` fixture for test matrix.
    :type test_mat: :class:`numpy.ndarray`
    """
    with pytest.raises(TypeError, match="ar must have dtype int or float"):
        cscale.stdscale(test_mat.astype(str))


def test_cscale_ddof(test_mat):
    """Test :func:`c_npy_demo.cscale.stdscale` raise when ``ddof`` is negative.

    :param test_mat: ``pytest`` fixture for test matrix.
    :type test_mat: :class:`numpy.ndarray`
    """
    with pytest.raises(ValueError, match="ddof must be a nonnegative int"):
        cscale.stdscale(test_mat, ddof=-1)


def test_cscale_empty():
    "Test :func:`c_npy_demo.cscale.stdscale` warn when ``ar`` is empty."
    with pytest.warns(RuntimeWarning, match="mean of empty array"):
        cscale.stdscale(np.array([]))


@pytest.mark.parametrize("ddof", [0, 1, 3, 5])
def test_cscale_allclose(test_mat, ddof):
    """Test :func:`c_npy_demo.cscale.stdscale` use of ``ddof``.

    :param test_mat: ``pytest`` fixture for test matrix.
    :type test_mat: :class:`numpy.ndarray`
    :param ddof: Delta degrees of freedom for standard deviation calculations.
    :type ddof: int
    """
    np.testing.assert_allclose(
        pyscale.stdscale(test_mat), cscale.stdscale(test_mat)
    )
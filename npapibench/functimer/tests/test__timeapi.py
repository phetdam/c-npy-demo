"""Unit tests for timing functions provided by the _timeapi extension.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import pytest
import sys

# pylint: disable=no-name-in-module,relative-beyond-top-level
from .._timeapi import autorange, timeit_enh, timeit_once, timeit_repeat


@pytest.fixture(scope="session")
def timeargs():
    """Default function and positional args to use with timing functions.

    Returns
    -------
    tuple
    """
    return max, (1, 2)


def test_timeit_once_sanity(pytype_raise, pyvalue_raise, timeargs):
    """Sanity checks for _timeapi.timeit_once.

    Parameters
    ----------
    pytype_raise : function
        pytest fixture. See top-level package conftest.py.
    pyvalue_raise : function
        pytest fixture. See top-level package conftest.py.
    timeargs : tuple
        pytest fixture. See timeargs.
    """
    # one positional argument required
    with pytype_raise():
        timeit_once()
    # require callable func and timer (if timer provided)
    with pyvalue_raise("func must be callable"):
        timeit_once("not callable")
    with pyvalue_raise("timer must be callable"):
        timeit_once(*timeargs, timer=22)
    # number of function calls in the trial must be positive
    with pyvalue_raise("number must be positive"):
        timeit_once(*timeargs, number=0)
    # timer must return a float value
    with pyvalue_raise("timer must return a float starting value"):
        timeit_once(*timeargs, timer=lambda: None)


def test_autorange_sanity(pytype_raise, pyvalue_raise, timeargs):
    """Sanity checks for _timeapi.autorange`.

    Parameters
    ----------
    pytype_raise : function
        pytest fixture. See top-level package conftest.py.
    pyvalue_raise : function
        pytest fixture. See top-level package conftest.py.
    timeargs : tuple
        pytest fixture. See timeargs.
    """
    # one positional argument required
    with pytype_raise():
        autorange(args=())
    # timer must be callable (raised by timeit_once)
    with pyvalue_raise("timer must be callable"):
        autorange(*timeargs, timer=None)


@pytest.mark.skip(reason="not yet refactored")
def test_repeat_sanity(timeargs):
    """Sanity checks for :func:`~c_npy_demo.functier.repeat`.

    :param timeargs: ``pytest`` fixture.
    :type timeargs: tuple
    """
    # one positional argument required
    with pytype_raise():
        repeat(args=())
    # args must be a tuple (raised by timeit_once)
    with pytest.raises(TypeError, match="args must be a tuple"):
        repeat(max, args=[1, 2])
    # kwargs must be dict (raised by timeit_once)
    with pytest.raises(TypeError, match="kwargs must be a dict"):
        repeat(max, args=((),), kwargs=["bogus"])
    # timer must be callable (raised by timeit_once)
    with pytype_raise():
        repeat(*timeargs, timer=None)
    # timer must have correct signature (raised by timeit_once)
    with pytype_raise():
        repeat(*timeargs, timer=lambda x: x)
    # number must be int (raised by timeit_once)
    with pytype_raise():
        repeat(*timeargs, number=1.2)
    # number must be positive (raised by timeit_once)
    with pytest.raises(ValueError):
        repeat(*timeargs, number=-1)
    # number must be <= sys.maxsize (PY_SSIZE_T_MAX). raised by timeit_once
    with pytest.raises(OverflowError):
        repeat(*timeargs, number=sys.maxsize + 999)
    # repeat must be positive
    with pytest.raises(ValueError, match="repeat must be positive"):
        repeat(*timeargs, repeat=-1)
    # repeat must be <= sys.maxsize (PY_SSIZE_T_MAX)
    with pytest.raises(OverflowError):
        repeat(*timeargs, repeat=sys.maxsize + 999)


@pytest.mark.skip(reason="not yet refactored")
def test_timeit_enh_sanity(timeargs):
    """Sanity checks for :func:`~c_npy_demo.timeit_enh`.

    Only includes checks for ``number``, ``repeat``, ``unit``, ``precision``
    since ``func``, ``args``, ``kwargs``, ``timer`` are checked by
    :func:`~c_npy_demo.timeit_once` which is called internally.

    :param timeargs: ``pytest`` fixture.
    :type timeargs: tuple
    """
    # number must be positive
    with pytest.raises(ValueError, match="number must be positive"):
        timeit_enh(*timeargs, number=0)
    # repeat must be positive
    with pytest.raises(ValueError, match="repeat must be positive"):
        timeit_enh(*timeargs, repeat=0)
    # unit must be valid
    with pytest.raises(ValueError, match="unit must be one of"):
        timeit_enh(*timeargs, unit="bloops")
    # precision must be positive and less than TimeitResult.MAX_PRECISION
    with pytest.raises(ValueError, match="precision must be positive"):
        timeit_enh(*timeargs, precision=0)
    with pytest.raises(
        ValueError,
        match=f"precision is capped at {TimeitResult.MAX_PRECISION}"
    ):
        timeit_enh(
            *timeargs, precision=TimeitResult.MAX_PRECISION + 1
        )
    # warning will be raised if precision >= TimeitResult.MAX_PRECISION // 2
    with pytest.warns(UserWarning, match="precision is rather high"):
        timeit_enh(
            *timeargs, precision=TimeitResult.MAX_PRECISION // 2
        )
    # this should run normally
    tir = timeit_enh(*timeargs)
    print(tir.brief)
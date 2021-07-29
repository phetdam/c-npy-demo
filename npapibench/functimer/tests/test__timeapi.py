"""Unit tests for timing functions provided by the _timeapi extension.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import pytest

# pylint: disable=no-name-in-module,relative-beyond-top-level
from .._timeapi import autorange, timeit_plus, timeit_once, timeit_repeat


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

    Don't need to check if args is tuple and if kwargs is dict since
    PyArg_ParseTupleAndKeywords handles this for us.

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
    # timer must return a float value and not take arguments
    with pyvalue_raise("timer must return a float starting value"):
        timeit_once(*timeargs, timer=lambda: None)
    with pytype_raise():
        timeit_once(*timeargs, timer=lambda x: x)
    # number of function calls in the trial must be positive
    with pyvalue_raise("number must be positive"):
        timeit_once(*timeargs, number=0)


def test_autorange_sanity(pytype_raise, pyvalue_raise, timeargs):
    """Sanity checks for _timeapi.autorange`.

    Don't need to check if args is tuple and if kwargs is dict since
    PyArg_ParseTupleAndKeywords handles this for us. Don't need to check timer
    since timeit_once is called and will do checks for timer.

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
        autorange()


def test_timeit_repeat_sanity(pytype_raise, pyvalue_raise, timeargs):
    """Sanity checks for _timeapi.timeit_repeat.

    Don't need to check if args is tuple and if kwargs is dict since
    PyArg_ParseTupleAndKeywords handles this for us. Don't need to check timer,
    number since timeit_once is called and will do checks for these.

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
        timeit_repeat()
    # number must be int (raised by timeit_once)
    with pytype_raise():
        timeit_repeat(*timeargs, number=1.2)
    # repeat must be positive
    with pyvalue_raise(match="repeat must be positive"):
        timeit_repeat(*timeargs, repeat=-1)


@pytest.mark.skip(reason="not yet refactored")
def test_timeit_plus_sanity(timeargs):
    """Sanity checks for _timeapi.timeit_plus.

    Only includes checks for ``number``, ``repeat``, ``unit``, ``precision``
    since ``func``, ``args``, ``kwargs``, ``timer`` are checked by
    :func:`~c_npy_demo.timeit_once` which is called internally.

    :param timeargs: ``pytest`` fixture.
    :type timeargs: tuple
    """
    # number must be positive
    with pytest.raises(ValueError, match="number must be positive"):
        timeit_plus(*timeargs, number=0)
    # repeat must be positive
    with pytest.raises(ValueError, match="repeat must be positive"):
        timeit_plus(*timeargs, repeat=0)
    # unit must be valid
    with pytest.raises(ValueError, match="unit must be one of"):
        timeit_plus(*timeargs, unit="bloops")
    # precision must be positive and less than TimeitResult.MAX_PRECISION
    with pytest.raises(ValueError, match="precision must be positive"):
        timeit_plus(*timeargs, precision=0)
    with pytest.raises(
        ValueError,
        match=f"precision is capped at {TimeitResult.MAX_PRECISION}"
    ):
        timeit_plus(
            *timeargs, precision=TimeitResult.MAX_PRECISION + 1
        )
    # warning will be raised if precision >= TimeitResult.MAX_PRECISION // 2
    with pytest.warns(UserWarning, match="precision is rather high"):
        timeit_plus(
            *timeargs, precision=TimeitResult.MAX_PRECISION // 2
        )
    # this should run normally
    tir = timeit_plus(*timeargs)
    print(tir.brief)
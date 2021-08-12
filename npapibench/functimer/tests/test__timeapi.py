"""Unit tests for timing functions provided by the _timeapi extension.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest

# pylint: disable=no-name-in-module,relative-beyond-top-level
from .._timeapi import autorange, timeit_plus, timeit_once, timeit_repeat
from .._timeunit import MAX_PRECISION, VALID_UNITS


@pytest.fixture(scope="session", params=[None, "args", "kwargs"])
def timeargs(request):
    """Fixture for default function and args to use with timing functions.

    Covers no arg, positional-only args, and keyword-only args cases.

    Parameters
    ----------
    request : _pytest.fixtures.FixtureRequest
        Built-in pytest fixture. See the pytest documentation for details.

    Returns
    -------
    tuple
    """
    if request.param == "args":
        return max, (1, 2)
    elif request.param == "kwargs":
        return lambda x=None: x, (), {"x": "identity"}
    else:
        return lambda: 1, ()


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


def test_timeit_once_number(timeargs):
    """Test that number is used correctly in _timeapi.timeit_once.

    Parameters
    ----------
    timeargs : tuple
        pytest fixture. See timeargs.
    """
    # values of number to pass. time should increase as number increases
    numbers = [10, 1000000]
    # compute times for each value of number
    times = np.empty(len(numbers))
    for i, number in enumerate(numbers):
        times[i] = timeit_once(*timeargs, number=number)
    # check that times are in ascending order
    assert np.all(times[:-1] < times[1:])


def test_autorange(timeargs):
    """Test that _timeapi.autorange returns values as expected.

    Don't need to check if args is tuple and if kwargs is dict since
    PyArg_ParseTupleAndKeywords handles this for us. timeit_once is called
    internally and will do checks for func, timer.

    Parameters
    ----------
    timeargs : tuple
        pytest fixture. See timeargs.
    """
    # max is a fast function so we should expect result divisible by 10
    assert autorange(*timeargs) % 10 == 0


def test_timeit_repeat_sanity(pyvalue_raise, timeargs):
    """Sanity checks for _timeapi.timeit_repeat.

    Don't need to check if args is tuple and if kwargs is dict since
    PyArg_ParseTupleAndKeywords handles this for us. timeit_once is called
    internally and will do checks for func, timer, number.

    Parameters
    ----------
    pyvalue_raise : function
        pytest fixture. See top-level package conftest.py.
    timeargs : tuple
        pytest fixture. See timeargs.
    """
    # repeat must be positive
    with pyvalue_raise(match="repeat must be positive"):
        timeit_repeat(*timeargs, repeat=0)


def test_timeit_plus_sanity(pyvalue_raise, timeargs):
    """Sanity checks for _timeapi.timeit_plus.

    Don't need to check if args is tuple and if kwargs is dict since
    PyArg_ParseTupleAndKeywords handles this for us. timeit_repeat is called
    internally and will check func, timer, number, repeat for us.

    Parameters
    ----------
    pyvalue_raise : function
        pytest fixture. See top-level package conftest.py.
    timeargs : tuple
        pytest fixture. See timeargs.
    """
    # get VALID_UNITS as a string of comma-separated double-quoted strings
    units = str(VALID_UNITS)[1:-1].replace("'", "\"")
    # unit must be valid (note escaped brackets)
    with pyvalue_raise(rf"unit must be one of \[{units}\]"):
        timeit_plus(*timeargs, unit="bloop")
    # precision must be positive and less than _timeunit.MAX_PRECISION
    with pyvalue_raise("precision must be positive"):
        timeit_plus(*timeargs, precision=0)
    with pyvalue_raise(f"precision is capped at {MAX_PRECISION}"):
        timeit_plus(*timeargs, precision=MAX_PRECISION + 1)
    # warning will be raised if precision >= _timeunit.MAX_PRECISION // 2
    with pytest.warns(UserWarning, match="precision is rather high"):
        timeit_plus(*timeargs, precision=MAX_PRECISION // 2)


def test_timeit_plus_return(timeargs):
    """Check that _timeapi.timeit_plus works as intended.

    Parameters
    ----------
    timeargs : tuple
        pytest fixture. See timeargs.
    """
    # run timeit_plus with some manually specified arguments
    time_kwargs = dict(number=1000, repeat=3, unit="msec", precision=2)
    res = timeit_plus(*timeargs, **time_kwargs)
    # check that TimeResult attributes match specified arguments
    assert res.unit == time_kwargs["unit"]
    assert res.number == time_kwargs["number"]
    assert res.repeat == time_kwargs["repeat"]
    assert res.precision == time_kwargs["precision"]
    # check that TimeResult times are the right shape
    assert res.times.size == time_kwargs["repeat"]
    # TimeResult unit tests already checked repr, loop_times, brief
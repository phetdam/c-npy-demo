"""Unit tests for the TimeResult class provided by the _timeresult extension.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest

# pylint: disable=no-name-in-module,relative-beyond-top-level
from .._timeresult import TimeResult
from .._timeunit import MAX_PRECISION


@pytest.fixture(scope="module")
def timeargs():
    """Valid args to use when initializing a TimeResult.

    .. note::

       The first positional argument to TimeResult.__new__ is explicitly made
       a float else PyArg_ParseTupleAndKeywords raises a TypeError since ints
       and floats having different object representations.

    Here the best time is 0.88 s, which is 8.8e-5 s/loop and 88 usec/loop.
    Note that if precision is not provided, it defaults to 1.

    Returns
    -------
    tuple
        Best trial time, time unit, number of function calls per trial, number
        of timing trials, results of timing trials.
    """
    return 88., "usec", 10000, 5, np.array([0.88, 1.02, 1.04, 1.024, 1])


@pytest.fixture(scope="session")
def tuple_replace():
    """Return a new tuple from an existing tuple with element modifications.

    Changes to the new tuple are specified with (idx, value) pairs passed
    after the tuple. The new tuple will be same length as the original tuple.

    Parameters
    ----------
    orig : tuple
        The original tuple whose elements are to be replaced
    *args
        Pairs of tuples in format (idx, value) where idx indexes orig and value
        gives the value that orig[i] should be replaced with.

    Returns
    -------
    tuple
        New tuple with all the specified modifications.
    """
    def _tuple_replace(orig, *args):
        orig_ = list(orig)
        for idx, val in args:
            orig_[idx] = val
        return tuple(orig_)

    _tuple_replace.__doc__ = tuple_replace.__doc__

    return _tuple_replace


def test_TimeResult_new_sanity(timeargs, tuple_replace):
    """Sanity checks for TimeResult.__new__.

    Parameters
    ----------
    timeargs : tuple
        pytest fixture. See timeargs.
    tuple_replace : function
        pytest fixture. See tuple_replace.
    """
    # generator for pytest.raises with ValueError and custom match
    raise_gen = lambda x=None: pytest.raises(ValueError, match=x)
    # wrapper for TimeResult with timeargs as default args. varargs accepts the
    # (idx, value) pairs used in tuple_replace varargs.
    TimeResult_Ex = lambda *args: TimeResult(*tuple_replace(timeargs, *args))
    #
    # all arguments except precision are required
    with pytest.raises(TypeError):
        TimeResult()
    # array of timing results must be convertible to double, 1D, with size
    # equal to the number of timing trials passed to TimeResult.__new__
    with raise_gen():
        TimeResult_Ex((4, ["a", "b", "c"]))
    with raise_gen("times must be 1D"):
        TimeResult_Ex((4, np.ones((2, 3))))
    with raise_gen(r"times\.size must equal repeat"):
        TimeResult_Ex((4, np.arange(10)))
    # unit must be valid, loop counts must be positive. trial counts always > 0
    # and if not are caught by the previous with raise_gen block.
    with raise_gen("unit must be one of"):
        TimeResult_Ex((1, "oowee"))
    with raise_gen("number must be positive"):
        TimeResult_Ex((2, 0))
    # check that precision must be valid, i.e. an int in [1, 20]
    with raise_gen("precision must be positive"):
        TimeResult(*timeargs, precision=0)
    with raise_gen(f"precision is capped at {MAX_PRECISION}"):
        TimeResult(*timeargs, precision=9001)


def test_TimeResult_repr(timeargs):
    """Check that TimeResult.__repr__ works as expected.

    Parameters
    ----------
    timeargs : tuple
        pytest fixture. See timeargs.
    """
    # create expected __repr__ string from timeargs
    repr_ex = "TimeResult("
    # each item is separated with ", " and has format "name=item". we append
    # precision timeargs so that we can build the string with for loop only
    for name, item in zip(
        ("best", "unit", "number", "repeat", "times", "precision"),
        timeargs + (1,)
    ):
        repr_ex += name + "=" + repr(item) + ", "
    # remove last ", " from repr_ex and append ")"
    repr_ex = repr_ex[:-2] + ")"
    # instantiate TimeResult and check that __repr__ works correctly
    res = TimeResult(*timeargs)
    assert repr(res) == repr_ex


def test_TimeResult_loop_times(timeargs):
    """Check that TimeResult.loop_times works as expected.

    Parameters
    ----------
    timeargs : tuple
        pytest fixture. See timeargs.
    """
    # compute loop times manually
    loop_times_ex = np.array(timeargs[4]) / timeargs[2]
    # instantiate new TimeResult and check its loop_times against loop_times_ex
    res = TimeResult(*timeargs)
    np.testing.assert_allclose(res.loop_times, loop_times_ex)
    # check that repeated calls produce refs to the same object
    assert id(res.loop_times) == id(res.loop_times)


def test_TimeResult_brief(timeargs):
    """Check that TimeResult.brief works as expected.

    Parameters
    ----------
    timeargs : tuple
        pytest fixture. See timeargs.
    """
    # print expected brief string
    brief_ex = (
        f"{timeargs[2]} loops, best of {timeargs[3]}: "
        f"{timeargs[0]:.1f} {timeargs[1]} per loop"
    )
    # instantiate new TimeResult and check that res.brief matches brief_ex
    res = TimeResult(*timeargs)
    assert res.brief == brief_ex
    # check that repeated calls produce refs to the same object
    assert id(res.brief) == id(res.brief)
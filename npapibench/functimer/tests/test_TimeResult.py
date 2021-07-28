"""Unit tests for the TimeResult class provided by the _timeresult extension.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest

# pylint: disable=no-name-in-module,relative-beyond-top-level
from .._timeresult import TimeResult
from .._timeunit import MAX_PRECISION


@pytest.fixture(scope="session")
def resargs():
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


def test_TimeResult_new_sanity(pyvalue_raise, tuple_replace, resargs):
    """Sanity checks for TimeResult.__new__.

    Parameters
    ----------
    pyvalue_raise : function
        pytest fixture. See top-level package conftest.py.
    tuple_replace : function
        pytest fixture. See top-level package conftest.py.
    resargs : tuple
        pytest fixture. See resargs.
    """
    # wrapper for TimeResult with resargs as default args. varargs accepts the
    # (idx, value) pairs used in tuple_replace varargs.
    TimeResult_Ex = lambda *args: TimeResult(*tuple_replace(resargs, *args))
    # all arguments except precision are required
    with pytest.raises(TypeError):
        TimeResult()
    # array of timing results must be convertible to double, 1D, with size
    # equal to the number of timing trials passed to TimeResult.__new__
    with pyvalue_raise():
        TimeResult_Ex((4, ["a", "b", "c"]))
    with pyvalue_raise("times must be 1D"):
        TimeResult_Ex((4, np.ones((2, 3))))
    with pyvalue_raise(r"times\.size must equal repeat"):
        TimeResult_Ex((4, np.arange(10)))
    # unit must be valid, loop counts must be positive. trial counts always > 0
    # and if not are caught by the previous with pyvalue_raise block.
    with pyvalue_raise("unit must be one of"):
        TimeResult_Ex((1, "oowee"))
    with pyvalue_raise("number must be positive"):
        TimeResult_Ex((2, 0))
    # check that precision must be valid, i.e. an int in [1, 20]
    with pyvalue_raise("precision must be positive"):
        TimeResult(*resargs, precision=0)
    with pyvalue_raise(f"precision is capped at {MAX_PRECISION}"):
        TimeResult(*resargs, precision=9001)


def test_TimeResult_repr(resargs):
    """Check that TimeResult.__repr__ works as expected.

    Parameters
    ----------
    resargs : tuple
        pytest fixture. See resargs.
    """
    # create expected __repr__ string from resargs
    repr_ex = "TimeResult("
    # each item is separated with ", " and has format "name=item". we append
    # precision resargs so that we can build the string with for loop only
    for name, item in zip(
        ("best", "unit", "number", "repeat", "times", "precision"),
        resargs + (1,)
    ):
        repr_ex += name + "=" + repr(item) + ", "
    # remove last ", " from repr_ex and append ")"
    repr_ex = repr_ex[:-2] + ")"
    # instantiate TimeResult and check that __repr__ works correctly
    res = TimeResult(*resargs)
    assert repr(res) == repr_ex


def test_TimeResult_loop_times(resargs):
    """Check that TimeResult.loop_times works as expected.

    Parameters
    ----------
    resargs : tuple
        pytest fixture. See resargs.
    """
    # compute loop times manually
    loop_times_ex = np.array(resargs[4]) / resargs[2]
    # instantiate new TimeResult and check its loop_times against loop_times_ex
    res = TimeResult(*resargs)
    np.testing.assert_allclose(res.loop_times, loop_times_ex)
    # check that repeated calls produce refs to the same object
    assert id(res.loop_times) == id(res.loop_times)


def test_TimeResult_brief(resargs):
    """Check that TimeResult.brief works as expected.

    Parameters
    ----------
    resargs : tuple
        pytest fixture. See resargs.
    """
    # print expected brief string
    brief_ex = (
        f"{resargs[2]} loops, best of {resargs[3]}: "
        f"{resargs[0]:.1f} {resargs[1]} per loop"
    )
    # instantiate new TimeResult and check that res.brief matches brief_ex
    res = TimeResult(*resargs)
    assert res.brief == brief_ex
    # check that repeated calls produce refs to the same object
    assert id(res.brief) == id(res.brief)
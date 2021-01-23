__doc__ = """Unit tests for the :class:`c_npy_demo.functimer.TimeitResult`.

Since this is a C extension type, it is especially tricky to debug.
"""

import numpy as np
import pytest

# pylint: disable=relative-beyond-top-level,no-name-in-module
from ..functimer import TimeitResult


@pytest.fixture(scope = "module")
def __new__args():
    """Valid args to pass to ``partial(TimeitResult.__new__, TimeitResult)``.

    .. note::

       ``best`` is explicitly made to be a float else
       :func:`test_TimeitResult_repr` raises an :class:`AssertionError` due to
       ints and floats having different representations.

    Here the best time is 0.88 s, which is 8e-5 s/loop and 88 usec/loop.
    """
    return 88., "usec", 10000, 5, (0.88, 1.02, 1.04, 1.024, 1)


@pytest.fixture(scope = "module")
def tuple_replace():
    """Return a new tuple from an existing tuple with element modifications.

    Changes to the new tuple are specified with ``(indes, value)`` pairs passed
    after the tuple. The new tuple will be same length as the original tuple.
    """
    def _tuple_replace(orig, *args):
        orig_ = list(orig)
        for idx, val in args:
            orig_[idx] = val
        return tuple(orig_)

    return _tuple_replace


def test_TimeitResult_new(__new__args, tuple_replace):
    """Sanity checks for ``Timeitresult.__new__``.

    :param __new__args: ``pytest`` fixture. See :func:`__new__args`.
    :type __new__args: tuple
    :param tuple_replace: ``pytest`` fixture. See :func:`tuple_replace`.
    :type tuple_replace: function
    """
    # all arguments are required
    with pytest.raises(TypeError):
        TimeitResult()
    # unit must be valid
    with pytest.raises(ValueError, match = "unit must be one of"):
        TimeitResult(*tuple_replace(__new__args, (1, "oowee")))
    # loop count must be valid (positive)
    with pytest.raises(ValueError, match = "number must be positive"):
        TimeitResult(*tuple_replace(__new__args, (2, 0)))
    # number of repeats (trials) must be valid (positive)
    with pytest.raises(ValueError, match = "repeat must be positive"):
        TimeitResult(*tuple_replace(__new__args, (3, 0)))
    # times tuple must be a tuple
    with pytest.raises(TypeError, match = "times must be a tuple"):
        TimeitResult(*tuple_replace(__new__args, (4, [])))
    # len(times) must equal repeat
    with pytest.raises(ValueError, match = r"len\(times\) must equal repeat"):
        TimeitResult(*tuple_replace(__new__args, (4, (0.99, 0.99))))
    # times must only contain floats or ints
    with pytest.raises(TypeError, match = ""):
        TimeitResult(
            *tuple_replace(__new__args, (4, (0.9, 0.99, "invalid", None, "2")))
        )
    # check that precision can be passed as both kwarg and positional arg
    TimeitResult(*__new__args, 1)
    TimeitResult(*__new__args, precision = 1)
    # check that precision must be valid (int in [1, 70])
    with pytest.raises(ValueError, match = "precision must be positive"):
        TimeitResult(*__new__args, precision = 0)
    with pytest.raises(ValueError, match = "precision is capped at 70"):
        TimeitResult(*__new__args, precision = 71)
    # check that warning is raised when brief precision >= 20
    with pytest.warns(UserWarning, match = "precision is rather high"):
        TimeitResult(*__new__args, precision = 20)
    # should initialize correctly
    TimeitResult(*__new__args)


def test_TimeitResult_repr(__new__args):
    """Check that ``TimeitResult.__repr__`` works as expected.

    :param __new__args: ``pytest`` fixture. See :func:`__new__args`.
    :type __new__args: tuple
    """
    # create expected __repr__ string from __new__args
    repr_ex = "TimeitResult("
    # each item is separated with ", " and has format "name=item"
    for name, item in zip(
        ("best", "unit", "number", "repeat", "times"), __new__args
    ):
        repr_ex = repr_ex + name + "=" + repr(item) + ", "
    # remove last ", " from repr_ex and append ")"
    repr_ex = repr_ex[:-2] + ")"
    # instantiate TimeitResult and check that __repr__ works correctly
    tir = TimeitResult(*__new__args)
    assert repr(tir) == repr_ex


def test_TimeitResult_loop_times(__new__args):
    """Check that ``TimeitResult.loop_times`` returns the expected.

    :param __new__args: ``pytest`` fixture. See :func:`__new__args`.
    :type __new__args: tuple
    """
    # compute loop times manually
    loop_times_ex = np.array(__new__args[4]) / __new__args[2]
    # instantiate new TimeitResult
    tir = TimeitResult(*__new__args)
    # check using np.allclose that loop_times_ex matches tir.loop_times
    np.testing.assert_allclose(tir.loop_times, loop_times_ex)


def test_TimeitResult_brief(__new__args):
    """Check that ``TimeitResult.brief`` returns the expected.

    :param __new__args: ``pytest`` fixture. See :func:`__new__args`.
    :type __new__args: tuple
    """
    # print expected brief string
    brief_ex = (
        f"{__new__args[2]} loops, best of {__new__args[3]}: "
        f"{__new__args[0]:.1f} {__new__args[1]} per loop"
    )
    # instantiate new TimeitResult and check that tir.brief matches brief_ex
    tir = TimeitResult(*__new__args)
    assert tir.brief == brief_ex
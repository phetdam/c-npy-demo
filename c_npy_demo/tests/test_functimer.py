__doc__ = "Unit tests for :func:`c_npy_demo.cscale.functimer`."

import pytest

# pylint: disable=relative-beyond-top-level
from .. import functimer


def test_timeit_once_sanity():
    "Sanity checks for ``PyArg_ParseTupleAndKeywords`` in ``timeit_once``."
    # one positional argument required
    with pytest.raises(TypeError):
        functimer.timeit_once(args = (1,))
    # args must be tuple
    with pytest.raises(TypeError, match = "args must be a tuple"):
        functimer.timeit_once(max, args = [1, 2])
    # kwargs must be dict
    with pytest.raises(TypeError, match = "kwargs must be a dict"):
        functimer.timeit_once(max, args = ((),), kwargs = ["bogus"])
    # timer must be callable
    with pytest.raises(TypeError):
        functimer.timeit_once(max, args = (1, 2), timer = None)
    # timer must have correct signature
    with pytest.raises(TypeError):
        functimer.timeit_once(max, args = (1, 2), timer = lambda x: x)
    # number must be int
    with pytest.raises(TypeError):
        functimer.timeit_once(max, args = (1, 2), number = 1.2)
    # number must be positive
    with pytest.raises(ValueError):
        functimer.timeit_once(max, args = (1, 2), number = -1)


def test_timeit_once_timer():
    "Check that the timer for ``timeit_once`` has correct return values"
    # must return a numeric value
    with pytest.raises(TypeError, match = "timer must return a numeric value"):
        functimer.timeit_once(max, args = (1, 2), timer = lambda: "cheese")
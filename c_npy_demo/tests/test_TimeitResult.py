__doc__ = """Unit tests for the :class:`c_npy_demo.functimer.TimeitResult`.

Since this is a C extension type, it is especially tricky to debug.
"""

import pytest

# pylint: disable=relative-beyond-top-level,no-name-in-module
from ..functimer import TimeitResult


@pytest.fixture(scope = "module")
def __new__args():
    "Valid args to pass to ``partial(TimeitResult.__new__, TimeitResult)``."
    return 0.02, "usec", 10000, 5, (0.02, 0.03, 0.04, 0.024, 0.026)


@pytest.fixture(scope = "module")
def tuple_replace():
    """Return a new tuple from an existing tuple with element modifications.

    Changes to the new tuple are specified with ``(value, index)`` pairs passed
    after the tuple. The new tuple will be same length as the original tuple.
    """
    def _tuple_replace(orig, *args):
        orig_ = list(orig)
        for val, idx in args:
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
        TimeitResult(*tuple_replace(__new__args, ("oowee", 1)))
    # loop count must be valid (positive)
    with pytest.raises(ValueError, match = "number must be positive"):
        TimeitResult(*tuple_replace(__new__args, (0, 2)))
    # number of repeats (trials) must be valid (positive)
    with pytest.raises(ValueError, match = "repeat must be positive"):
        TimeitResult(*tuple_replace(__new__args, (0, 3)))
    # times tuple must be a tuple
    with pytest.raises(TypeError, match = "times must be a tuple"):
        TimeitResult(*tuple_replace(__new__args, ([], 4)))
    # len(times) must equal repeat
    with pytest.raises(ValueError, match = r"len\(times\) must equal repeat"):
        TimeitResult(*tuple_replace(__new__args, ((0.03, 0.02), 4)))
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
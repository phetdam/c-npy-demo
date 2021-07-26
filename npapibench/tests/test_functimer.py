"""Unit tests for functimer functions.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import pytest
import sys

# pylint: disable=relative-beyond-top-level,no-name-in-module
from .. import functimer
from ..functimer import TimeitResult


@pytest.fixture(scope="module")
def func_and_args():
    """Default function and positional arguments used during unit testing.

    Uses the builtin :func:`max` function with args ``(1, 2)``.

    :rtype: tuple
    """
    return max, (1, 2)


def test_timeit_once_sanity(func_and_args):
    """Sanity checks for functimer.timeit_once.

    :param func_and_args: ``pytest`` fixture.
    :type func_and_args: tuple
    """
    # one positional argument required
    with pytest.raises(TypeError):
        functimer.timeit_once(args=(1,))
    # args must be tuple
    with pytest.raises(TypeError, match="args must be a tuple"):
        functimer.timeit_once(max, args=[1, 2])
    # kwargs must be dict
    with pytest.raises(TypeError, match="kwargs must be a dict"):
        functimer.timeit_once(max, args=((),), kwargs=["bogus"])
    # timer must be callable
    with pytest.raises(TypeError):
        functimer.timeit_once(*func_and_args, timer=None)
    # timer must have correct signature
    with pytest.raises(TypeError):
        functimer.timeit_once(*func_and_args, timer=lambda x: x)
    # number must be int
    with pytest.raises(TypeError):
        functimer.timeit_once(*func_and_args, number=1.2)
    # number must be positive
    with pytest.raises(ValueError):
        functimer.timeit_once(*func_and_args, number=-1)
    # number must be less than sys.maxsize (PY_SSIZE_T_MAX)
    with pytest.raises(OverflowError):
        functimer.timeit_once(*func_and_args, number=sys.maxsize + 999)


def test_timeit_once_timer(func_and_args):
    """Check :func:`~c_npy_demo.functimer.timeit_once` timer return values.

    :param func_and_args: ``pytest`` fixture.
    :type func_and_args: tuple
    """
    # must return a numeric value
    with pytest.raises(TypeError, match="timer must return a numeric value"):
        functimer.timeit_once(*func_and_args, timer=lambda: "cheese")


def test_autorange_sanity(func_and_args):
    """Sanity checks for :func:`~c_npy_demo.functimer.autorange`.

    :param func_and_args: ``pytest`` fixture.
    :type func_and_args: tuple
    """
    # one positional argument required
    with pytest.raises(TypeError):
        functimer.autorange(args=())
    # args must be a tuple (raised by timeit_once)
    with pytest.raises(TypeError, match="args must be a tuple"):
        functimer.autorange(max, args=[1, 2])
    # kwargs must be a dict (raised by timeit_once)
    with pytest.raises(TypeError, match="kwargs must be a dict"):
        functimer.autorange(max, args=((),), kwargs=["also bogus"])
    # timer must be callable (raised by timeit_once)
    with pytest.raises(TypeError):
        functimer.autorange(*func_and_args, timer=None)


def test_repeat_sanity(func_and_args):
    """Sanity checks for :func:`~c_npy_demo.functier.repeat`.

    :param func_and_args: ``pytest`` fixture.
    :type func_and_args: tuple
    """
    # one positional argument required
    with pytest.raises(TypeError):
        functimer.repeat(args=())
    # args must be a tuple (raised by timeit_once)
    with pytest.raises(TypeError, match="args must be a tuple"):
        functimer.repeat(max, args=[1, 2])
    # kwargs must be dict (raised by timeit_once)
    with pytest.raises(TypeError, match="kwargs must be a dict"):
        functimer.repeat(max, args=((),), kwargs=["bogus"])
    # timer must be callable (raised by timeit_once)
    with pytest.raises(TypeError):
        functimer.repeat(*func_and_args, timer=None)
    # timer must have correct signature (raised by timeit_once)
    with pytest.raises(TypeError):
        functimer.repeat(*func_and_args, timer=lambda x: x)
    # number must be int (raised by timeit_once)
    with pytest.raises(TypeError):
        functimer.repeat(*func_and_args, number=1.2)
    # number must be positive (raised by timeit_once)
    with pytest.raises(ValueError):
        functimer.repeat(*func_and_args, number=-1)
    # number must be <= sys.maxsize (PY_SSIZE_T_MAX). raised by timeit_once
    with pytest.raises(OverflowError):
        functimer.repeat(*func_and_args, number=sys.maxsize + 999)
    # repeat must be positive
    with pytest.raises(ValueError, match="repeat must be positive"):
        functimer.repeat(*func_and_args, repeat=-1)
    # repeat must be <= sys.maxsize (PY_SSIZE_T_MAX)
    with pytest.raises(OverflowError):
        functimer.repeat(*func_and_args, repeat=sys.maxsize + 999)


def test_timeit_enh_sanity(func_and_args):
    """Sanity checks for :func:`~c_npy_demo.functimer.timeit_enh`.

    Only includes checks for ``number``, ``repeat``, ``unit``, ``precision``
    since ``func``, ``args``, ``kwargs``, ``timer`` are checked by
    :func:`~c_npy_demo.functimer.timeit_once` which is called internally.

    :param func_and_args: ``pytest`` fixture.
    :type func_and_args: tuple
    """
    # number must be positive
    with pytest.raises(ValueError, match="number must be positive"):
        functimer.timeit_enh(*func_and_args, number=0)
    # repeat must be positive
    with pytest.raises(ValueError, match="repeat must be positive"):
        functimer.timeit_enh(*func_and_args, repeat=0)
    # unit must be valid
    with pytest.raises(ValueError, match="unit must be one of"):
        functimer.timeit_enh(*func_and_args, unit="bloops")
    # precision must be positive and less than TimeitResult.MAX_PRECISION
    with pytest.raises(ValueError, match="precision must be positive"):
        functimer.timeit_enh(*func_and_args, precision=0)
    with pytest.raises(
        ValueError,
        match=f"precision is capped at {TimeitResult.MAX_PRECISION}"
    ):
        functimer.timeit_enh(
            *func_and_args, precision=TimeitResult.MAX_PRECISION + 1
        )
    # warning will be raised if precision >= TimeitResult.MAX_PRECISION // 2
    with pytest.warns(UserWarning, match="precision is rather high"):
        functimer.timeit_enh(
            *func_and_args, precision=TimeitResult.MAX_PRECISION // 2
        )
    # this should run normally
    tir = functimer.timeit_enh(*func_and_args)
    print(tir.brief)
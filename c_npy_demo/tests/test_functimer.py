__doc__ = "Unit tests for :mod:`c_npy_demo.functimer` functions."

import pytest
import sys
import tracemalloc

# pylint: disable=relative-beyond-top-level,no-name-in-module
from .. import functimer
from ..functimer import TimeitResult

# start tracemalloc so we can take memory snapshots
tracemalloc.start()


@pytest.fixture(scope="module")
def func_and_args():
    """Default function and positional arguments used during unit testing.

    Uses the builtin :func:`max` function with args ``(1, 2)``.

    :rtype: tuple
    """
    return max, (1, 2)


def test_timeit_once_sanity(func_and_args):
    """Sanity checks for :func:`~c_npy_demo.functimer.timeit_once`.

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


def test_timeit_once_memleak(func_and_args):
    """Check if :func:`~c_npy_demo.functimer.timeit_once` is leaking memory.

    .. note::

       Use of :mod:`tracemalloc` does not seem to be able to catch very small
       memory leaks due to improper reference counting. It can, however, catch
       allocation of new objects using the C API that leak their reference;
       for example, a call made to ``PyTuple_New`` that leaks its reference.

    :param func_and_args: ``pytest`` fixture.
    :type func_and_args: tuple
    """
    # filter so that memory allocation tracing is limited to timeit_once call
    trace_filters = [tracemalloc.Filter(True, __file__, lineno=86)]
    # take snapshots before and after running timeit_once
    snap_1 = tracemalloc.take_snapshot().filter_traces(trace_filters)
    functimer.timeit_once(*func_and_args, number=1000)
    snap_2 = tracemalloc.take_snapshot().filter_traces(trace_filters)
    # compare second to first snapshot and print differences (top 10)
    diffs = snap_2.compare_to(snap_1, "lineno")
    print(diffs)


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


def test_autorange_memleak(func_and_args):
    """Check if :func:`~c_npy_demo.functimer.autorange` is leaking memory.

    :param func_and_args: ``pytest`` fixture.
    :type func_and_args: tuple
    """
    # filter so that memory allocation tracing is limited to autorange call
    trace_filters = [tracemalloc.Filter(True, __file__, lineno=123)]
    # take snapshots before and after running timeit_once
    snap_1 = tracemalloc.take_snapshot().filter_traces(trace_filters)
    functimer.autorange(*func_and_args)
    snap_2 = tracemalloc.take_snapshot().filter_traces(trace_filters)
    # compare second to first snapshot and print differences (top 10)
    diffs = snap_2.compare_to(snap_1, "lineno")
    print(diffs[0])


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


def test_repeat_memleak(func_and_args):
    """Check if :func:`~c_npy_demo.functimer.repeat` is leaking memory.

    :param func_and_args: ``pytest`` fixture.
    :type func_and_args: tuple
    """
    # filter so that memory allocation tracing is limited to repeat call
    trace_filters = [tracemalloc.Filter(True, __file__, lineno=178)]
    # take snapshots before and after running timeit_once
    snap_1 = tracemalloc.take_snapshot().filter_traces(trace_filters)
    functimer.repeat(*func_and_args, number=400, repeat=2)
    snap_2 = tracemalloc.take_snapshot().filter_traces(trace_filters)
    # compare second to first snapshot and print differences (top 10)
    diffs = snap_2.compare_to(snap_1, "lineno")
    print(diffs[0])


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


def test_timeit_enh_memleak(func_and_args):
    """Check if :func:`~c_npy_demo.functimer.timeit_enh` is leaking memory.

    :param func_and_args: ``pytest`` fixture.
    :type func_and_args: tuple
    """
    # filter so that memory allocation tracing is limited to repeat call
    trace_filters = [tracemalloc.Filter(True, __file__, lineno=234)]
    # take snapshots before and after running timeit_once
    snap_1 = tracemalloc.take_snapshot().filter_traces(trace_filters)
    functimer.timeit_enh(*func_and_args, precision=2)
    snap_2 = tracemalloc.take_snapshot().filter_traces(trace_filters)
    # compare second to first snapshot and print differences (top 10)
    diffs = snap_2.compare_to(snap_1, "lineno")
    print(diffs[0])
__doc__ = "Unit tests for :func:`c_npy_demo.cscale.functimer` functions."

import pytest
import sys
import tracemalloc

# pylint: disable=relative-beyond-top-level
from .. import functimer

# start tracemalloc so we can take memory snapshots
tracemalloc.start()


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
    # number must be less than sys.maxsize (PY_SSIZE_T_MAX)
    with pytest.raises(OverflowError):
        functimer.timeit_once(max, args = (1, 2), number = sys.maxsize + 999)


def test_timeit_once_timer():
    "Check that the timer for ``timeit_once`` has correct return values"
    # must return a numeric value
    with pytest.raises(TypeError, match = "timer must return a numeric value"):
        functimer.timeit_once(max, args = (1, 2), timer = lambda: "cheese")


def test_timeit_once_memleak():
    """Check if ``timeit_once`` is leaking memory.

    .. note::

       Use of :mod:`tracemalloc` does not seem to be able to catch very small
       memory leaks due to improper reference counting. It can, however, catch
       allocation of new objects using the C API that leak their reference;
       for example, a call made to ``PyTuple_New`` that leaks its reference.
    """
    # filter so that memory allocation tracing is limited to timeit_once call
    trace_filters = [tracemalloc.Filter(True, __file__, lineno = 63)]
    # take snapshots before and after running timeit_once
    snap_1 = tracemalloc.take_snapshot().filter_traces(trace_filters)
    functimer.timeit_once(max, args = (1, 2), number = 1000)
    snap_2 = tracemalloc.take_snapshot().filter_traces(trace_filters)
    # compare second to first snapshot and print differences (top 10)
    diffs = snap_2.compare_to(snap_1, "lineno")
    print(diffs[0])


def test_autorange_sanity():
    "Sanity checks for ``PyArg_ParseTupleAndKeywords`` in ``autorange``."
    # one positional argument required
    with pytest.raises(TypeError):
        functimer.autorange(args = ())
    # args must be a tuple (raised by timeit_once)
    with pytest.raises(TypeError, match = "args must be a tuple"):
        functimer.autorange(max, args = [1, 2])
    # kwargs must be a dict (raised by timeit_once)
    with pytest.raises(TypeError, match = "kwargs must be a dict"):
        functimer.autorange(max, args = ((),), kwargs = ["also bogus"])
    # timer must be callable (raised by timeit_once)
    with pytest.raises(TypeError):
        functimer.autorange(max, args = (1, 2), timer = None)


def test_autorange_memleak():
    "Check if ``autorange`` is leaking memory."
    # filter so that memory allocation tracing is limited to autorange call
    trace_filters = [tracemalloc.Filter(True, __file__, lineno = 92)]
    # take snapshots before and after running timeit_once
    snap_1 = tracemalloc.take_snapshot().filter_traces(trace_filters)
    functimer.autorange(max, args = (1, 2))
    snap_2 = tracemalloc.take_snapshot().filter_traces(trace_filters)
    # compare second to first snapshot and print differences (top 10)
    diffs = snap_2.compare_to(snap_1, "lineno")
    print(diffs[0])


def test_repeat_sanity():
    "Sanity checks for ``PyArg_ParseTupleAndKeywords`` in ``repeat``."
    # one positional argument required
    with pytest.raises(TypeError):
        functimer.repeat(args = ())
    # args must be a tuple (raised by timeit_once)
    with pytest.raises(TypeError, match = "args must be a tuple"):
        functimer.repeat(max, args = [1, 2])
    # kwargs must be dict (raised by timeit_once)
    with pytest.raises(TypeError, match = "kwargs must be a dict"):
        functimer.repeat(max, args = ((),), kwargs = ["bogus"])
    # timer must be callable (raised by timeit_once)
    with pytest.raises(TypeError):
        functimer.repeat(max, args = (1, 2), timer = None)
    # timer must have correct signature (raised by timeit_once)
    with pytest.raises(TypeError):
        functimer.repeat(max, args = (1, 2), timer = lambda x: x)
    # number must be int (raised by timeit_once)
    with pytest.raises(TypeError):
        functimer.repeat(max, args = (1, 2), number = 1.2)
    # number must be positive (raised by timeit_once)
    with pytest.raises(ValueError):
        functimer.repeat(max, args = (1, 2), number = -1)
    # number must be <= sys.maxsize (PY_SSIZE_T_MAX). raised by timeit_once
    with pytest.raises(OverflowError):
        functimer.repeat(max, args = (1, 2), number = sys.maxsize + 999)
    # repeat must be positive
    with pytest.raises(ValueError, match = "repeat must be positive"):
        functimer.repeat(max, args = (1, 2), repeat = -1)
    # repeat must be <= sys.maxsize (PY_SSIZE_T_MAX)
    with pytest.raises(OverflowError):
        functimer.repeat(max, args = (1, 2), repeat = sys.maxsize + 999)


def test_repeat_memleak():
    "Check if ``repeat`` is leaking memory."
    # filter so that memory allocation tracing is limited to repeat call
    trace_filters = [tracemalloc.Filter(True, __file__, lineno = 139)]
    # take snapshots before and after running timeit_once
    snap_1 = tracemalloc.take_snapshot().filter_traces(trace_filters)
    functimer.repeat(max, args = (1, 2), number = 400, repeat = 2)
    snap_2 = tracemalloc.take_snapshot().filter_traces(trace_filters)
    # compare second to first snapshot and print differences (top 10)
    diffs = snap_2.compare_to(snap_1, "lineno")
    print(diffs[0])
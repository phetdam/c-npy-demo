__doc__ = "Benchmarks for the C extension ``_np_bcast`` and ``_ivlib.so``."

import argparse
from functools import wraps
import inspect
import numpy as np
from numpy.random import RandomState
from time import time

from .ivlib import _ivlib
from .utils import options_csv_to_ndarray
from ._np_bcast import np_float64_bcast_1d_ext


def _np_1d_broadcast(f, axis = 0, broadcast = True):
    """Decorate a function to broadcast args into 1D :class:`numpy.ndarray`.
    
    See :func:`np_1d_broadcast` for parameter details.
    
    .. note:: Do not call this directly. Use :func:`np_1d_broadcast` instead.
    """
    # original function wrapper for rows
    @wraps(f)
    def _broadcast_wrap_row(*args, **kwargs):
        # convert args if not iterable/is string 
        args = list(args)
        for i in range(len(args)):
            arg = args[i]
            if hasattr(arg, "__iter__") and (not isinstance(arg, (str, bytes))):
                args[i] = np.array(arg)
            # broadcast scalars, strings, and bytes
            else: args[i] = np.array([arg])
        # get parameters from signature of f
        params = dict(inspect.signature(f).parameters)
        # for any params that have non empty defaults, add to kwargs if they
        # don't already exist in kwargs
        for key, val in params.items():
            if val.default != inspect._empty:
                kwargs.setdefault(key, val.default)
        for key, val in kwargs.items():
            if hasattr(val, "__iter__") and (not isinstance(val, (str, bytes))):
                kwargs[key] = np.array(val)
            else: kwargs[key] = np.array([val])
        # get max length of args and kwargs and extend length-1 arrays
        # note that args and/or kwargs may be empty
        if broadcast == True:
            max_len = 0
            if len(args) > 0:
                max_len = max(tuple(map(lambda x: len(x), args)))
            if len(kwargs) > 0:
                max_len = max(max(tuple(map(lambda x: len(x),
                                            kwargs.values()))), max_len)
            # lengths may be 0
            for i in range(len(args)):
                arg = args[i]
                if arg.shape[0] == 1:
                    args[i] = np.array([arg[0] for _ in range(max_len)])
                elif arg.shape[0] == max_len: pass
                else:
                    raise ValueError("args array shape mismatch: "
                                     f"{arg.shape[0]} != {max_len} (max_len)")
            for key, val in kwargs.items():
                if val.shape[0] == 1:
                    kwargs[key] = np.array([val[0] for _ in range(max_len)])
                elif val.shape[0] == max_len: pass
                else:
                    raise ValueError("kwargs array shape mismatch: "
                                     f"{val.shape[0]} != {max_len} (max_len)")
        # feed args back into original function and get result as ndarray
        res = np.array(f(*args, **kwargs))
        # if length of res is 1, return as scalar, else return
        if res.shape[0] == 1: return res[0]
        return res

    # original function wrapper for columns
    @wraps(f)
    def _broadcast_wrap_col(*args, **kwargs):
        # convert args if not iterable/is string 
        args = list(args)
        for i in range(len(args)):
            arg = args[i]
            if hasattr(arg, "__iter__") and (not isinstance(arg, (str, bytes))):
                args[i] = np.array(arg).reshape((len(arg), 1))
            # broadcast scalars, strings, and bytes
            else: args[i] = np.array([arg]).reshape((1, 1))
        # get parameters from signature of f
        params = dict(inspect.signature(f).parameters)
        # for any params that have non empty defaults, add to kwargs if they
        # don't already exist in kwargs
        for key, val in params.items():
            if val.default != inspect._empty:
                kwargs.setdefault(key, val.default)
        for key, val in kwargs.items():
            if hasattr(val, "__iter__") and (not isinstance(val, (str, bytes))):
                kwargs[key] = np.array(val).reshape((len(val), 1))
            else: kwargs[key] = np.array([val]).reshape((1, 1))
        # get max length of args and kwargs and extend length-1 arrays
        # note that args and/or kwargs may be empty
        if broadcast == True:
            max_len = 0
            if len(args) > 0:
                max_len = max(tuple(map(lambda x: len(x), args)))
            if len(kwargs) > 0:
                max_len = max(max(tuple(map(lambda x: len(x),
                                            kwargs.values()))), max_len)
            # lengths may be 0
            for i in range(len(args)):
                arg = args[i]
                if arg.shape[0] == 1:
                    args[i] = np.array([arg[0] for _ in range(max_len)])
                elif arg.shape[0] == max_len: pass
                else:
                    raise ValueError("args array shape mismatch: "
                                     f"{arg.shape[0]} != {max_len} (max_len)")
            for key, val in kwargs.items():
                if val.shape[0] == 1:
                    kwargs[key] = np.array([val[0] for _ in range(max_len)])
                elif val.shape[0] == max_len: pass
                else:
                    raise ValueError("kwargs array shape mismatch: "
                                     f"{val.shape[0]} != {max_len} (max_len)")
        # feed args back into original function and get result as ndarray
        res = np.array(f(*args, **kwargs))
        # if length of res is 1, return as scalar, else return
        if res.shape[0] == 1: return res[0]
        return res
    
    if axis == 0: return _broadcast_wrap_row
    elif axis == 1: return _broadcast_wrap_col
    raise ValueError("axis must be in (0, 1)")


def np_1d_broadcast(f = None, axis = 0, broadcast = True):
    """Decorate a function to broadcast args into 1D :class:`numpy.ndarray`.
    
    Broadcasting will only work properly on one-dimensional iterables, strings,
    numeric types, and bytes objects. It is assumed that all the arguments of
    the decorated function are expected to be broadcasted to the same length
    and that the returned value will be an iterable.
    
    :param f: Function whose arguments are to be broadcasted.
    :type f: function, optional
    :param axis: Determines orientation of the broadcasted arrays. ``0`` for
        arrays of shape ``(max_len,)``, ``1`` for arrays of shape
        ``(max_len, 1)``. Defaults to ``0``.
    :type axis: int, 
    :param broadcast: ``True`` to broadcast arguments, ``False`` to only convert
        arguments to :class:`numpy.ndarray`. Use ``False`` if the shapes of the
        arguments to ``f`` are known to be the same and do not requrie
        broadcasting; this provides a slight speed gain. Defaults to ``True``.
    :type broadcast: bool, optional
    :returns: Function that broadcasts all its args into :class:`numpy.ndarray`.
    :rtype: function
    """
    # if None, return decorator for f, else return decorated f
    if f is None:
        
        def _wrapper(_f):
            return _np_1d_broadcast(_f, axis = axis, broadcast = broadcast)
        
        return _wrapper
        
    return _np_1d_broadcast(f, axis = axis, broadcast = broadcast)


def _np_float64_bcast_1d(f, axis = 0):
    """Decorate a function to broadcast args into 1D :class:`numpy.ndarray`.
    
    See :func:`np_float64_bcast_1d` for parameter details.
    
    .. note:: Do not call directly. Use :func:`np_float64_bcast_1d` instead.
    
    :rtype: function
    """
    @wraps(f)
    def _broadcast_wrap(*args, **kwargs):
        # get parameters from signature of f
        params = dict(inspect.signature(f).parameters)
        # for any params that have non empty defaults, add to kwargs if they
        # don't already exist in kwargs
        for key, val in params.items():
            if val.default != inspect._empty:
                kwargs.setdefault(key, val.default)
        # convert args and kwargs using np_float64_bcast_1d_ext
        args = np_float64_bcast_1d_ext(args, axis)
        kwargs = np_float64_bcast_1d_ext(kwargs, axis)
        # feed args back into original function and get result as ndarray
        res = np.array(f(*args, **kwargs))
        return res
    
    return _broadcast_wrap


def np_float64_bcast_1d(f = None, axis = 0):
    """Decorate a function to broadcast args into 1D :class:`numpy.ndarray`.
    
    Broadcasting will only work properly on one-dimensional iterables, strings,
    numeric types, and bytes objects. It is assumed that all the arguments of
    the decorated function are expected to be broadcasted to the same length.
    
    This is a wrapper around the C function ``np_float64_bcast_1d_ext`` which
    converts Python objects into 1D float64 NumPy arrays, and is faster than
    :func:`np_1d_broadcast`, although of course it can only convert to float64.
    Lacks a ``broadcast`` parameter like :func:`np_1d_broadcast` since there is
    no extra speed gain in this case.
    
    :param f: Function whose arguments are to be broadcasted.
    :type f: function, optional
    :param axis: Determines orientation of the broadcasted arrays. ``0`` for
        arrays of shape ``(max_len,)``, ``1`` for arrays of shape
        ``(max_len, 1)``. Defaults to ``0``.
    :type axis: int, 
    :returns: Function that broadcasts all its args into :class:`numpy.ndarray`.
    :rtype: function
    """
    # if f is None, return decorator, else return decorated f
    if f is None:
        
        def _wrapper(_f):
            return _np_float64_bcast_1d(_f, axis = axis)
        
        return _wrapper
    
    return _np_float64_bcast_1d(f, axis = axis)


def _timer(f, disp = False, replace_output = True):
    """Simple timer function decorator. Called by :func:`timer`.
    
    See :func:`timer` for parameter details. Results in seconds.

    :rtype: function
    """
    @wraps(f)
    def _wrapper(*args, **kwargs):
        time_a = time()
        res = f(*args, **kwargs)
        exec_time = time() - time_a
        if disp == True:
            print(f"{f.__module__}.{f.__name__}: {exec_time} s")
        if replace_output == True: return exec_time
        return res
    
    return _wrapper


def timer(f = None, disp = False, replace_output = True):
    """Simple timer function decorator.
    
    :param f: Function to decorate for timing
    :type f: function
    :param disp: ``True`` to print time to screen, ``False`` for silence.
    :type disp: bool, optional
    :param replace_output: ``True`` to replace return value of ``f`` with the
        time taken for ``f`` to execute, ``False`` otherwise. If set to
        ``False``, ``disp`` should be ``True`` or no result can be retrieved.
    :type replace_output: bool, optional
    :rtype: function
    """
    # if f is None, return a decorator
    if f is None:
        
        def _decorator(_f):
            return _timer(_f, disp = disp, replace_output = replace_output)
        
        return _decorator
    
    # else just call _timer
    return _timer(f, disp = disp, replace_output = replace_output)

        
def p_int(x):
    """Convert ``x`` to a positive integer, if possible.
    
    Raises :class:`ValueError` if ``x`` can't be converted to a positive int.
    
    :param x: Object to convert
    :type x: object
    :rtype: int
    """
    x = int(x)
    if x <= 0: raise ValueError("cannot convert to positive int")
    return x


def nn_int(x):
    """Convert ``x`` to a non-negative integer, if possible.
    
    Raises :class:`ValueError` if ``x`` can't be converted to nonnegative int.
    
    :param x: Object to convert
    :type x: object
    :rtype: int
    """
    x = int(x)
    if x < 0: raise ValueError("cannot convert to nonnegative int")
    return x


def echo(*args):
    "Returns positional args ``args`` in a tuple."
    return args


_BENCH_EXT_DESC = \
"""Benchmark to compare the decorators in c_npy_demo.utils, np_1d_broadcast and
np_float64_bcast_1d. Both decorators essentially work by converting the args and
kwargs passed to a function into 1D numpy arrays, if possible, although
np_1d_broadcast is a little more general and not limited to just converting to
float64 numpy arrays like np_float64_bcast_1d.

The benchmark will randomly generate nvecs Python lists of length vlen and ncons
random constants, all taking values in [0, 1), use np_1d_broadcast and
np_float64_bcast_1d, both decorated with the timer decorator, to decorate a
function that does nothing but return its positional args as a tuple. The
reported time for each decorator will be the minimum time it takes to convert
the inputs to numpy ndarrays out of ntrs trials. The total benchmark runtime
will also be reported. See the usage for details on the variables mentioned.

One should see that the C implementation is ~2 times faster than the Python
implementation, although of course your mileage may vary.
"""
_BENCH_EXT_SEED_HELP = "The seed for random number generation. Defaults to 7."
_BENCH_EXT_NVECS_HELP = "Vectors to convert to numpy arrays, default 3"
_BENCH_EXT_NCONS_HELP = "Constants to broadcast to numpy arrays, default 2"
_BENCH_EXT_VLEN_HELP = "[Broadcast] [v]ector lengths, default 100000"
_BENCH_EXT_NTRS_HELP = "Number of trials to run, default 5"
_BENCH_VERBOSE_HELP = "Specify for verbose output, disabled by default."


def bench_ext_main(args = None):
    """Main method for ``c_npy_demo.bench.ext`` benchmark script.
    
    To run from the interpreter, pass a list of string arguments to ``args``.
    
    :param args: List of string arguments to pass, default ``None``.
    :type args: list, optional
    """
    # overall routine execution time start
    time_a = time()
    # add arguments and parse
    arp = argparse.ArgumentParser(
        prog = "c_npy_demo.bench.ext", description = _BENCH_EXT_DESC,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )
    arp.add_argument("-s", "--seed", default = 7, type = nn_int,
                     help = _BENCH_EXT_SEED_HELP)
    arp.add_argument("-nv", "--nvecs", default = 3, type = p_int,
                     help = _BENCH_EXT_NVECS_HELP)
    arp.add_argument("-nc", "--ncons", default = 2, type = p_int,
                     help = _BENCH_EXT_NCONS_HELP)
    arp.add_argument("-vl", "--vlen", default = 100000, type = p_int,
                     help = _BENCH_EXT_VLEN_HELP)
    arp.add_argument("-nt", "--ntrs", default = 5, type = p_int,
                     help = _BENCH_EXT_NTRS_HELP)
    arp.add_argument("-v", "--verbose", action = "store_true",
                     help = _BENCH_VERBOSE_HELP)
    args = arp.parse_args(args = args)
    # extract args for convenience
    seed, nvecs, ncons, vlen, ntrs, verbose = (
        args.seed, args.nvecs, args.ncons, args.vlen, args.ntrs, args.verbose
    )
    # seed RandomState (for compatibility)
    rs = RandomState(seed = seed)
    # list of random tuples and constants
    ins = [None for _ in range(nvecs + ncons)]
    # generate nvecs random tuples (forces conversion) and ncons constants
    for i in range(nvecs): ins[i] = tuple(rs.rand(vlen))
    for i in range(nvecs, nvecs + ncons): ins[i] = rs.rand()
    # trial results for np_1d_broadcast and np_float64_bcast_1d
    py_res = np.zeros(ntrs)
    c_res = np.zeros(ntrs)
    # decorate decorators with timer (use echo as eval function)
    py_eval = timer(np_1d_broadcast(echo))
    c_eval = timer(np_float64_bcast_1d(echo))
    # if verbose, print statement on variables used in the benchmark
    if verbose == True:
        # get maximum number of digits contained in one of the variables
        max_digits = max(list(map(lambda x: len(str(x)),
                                  (seed, nvecs, ncons, vlen, ntrs))))
        print(f"seed:                {seed:{max_digits}d}")
        print(f"number of vectors:   {nvecs:{max_digits}d}")
        print(f"number of constants: {ncons:{max_digits}d}")
        print(f"vector length:       {vlen:{max_digits}d}")
        print(f"number of trials:    {ntrs:{max_digits}d}")
    # call each ntrs times and record timing result
    for i in range(ntrs):
        py_res[i] = py_eval(*ins)
        c_res[i] = c_eval(*ins)
    # print results
    py_time = min(py_res)
    c_time = min(c_res)
    print(f"best of {ntrs} for {py_eval}: {py_time:.3e} s")
    print(f"best of {ntrs} for {c_eval}: {c_time:.3e} s")
    print(f"C func is {(py_time / c_time):.5f} times faster than Python func")
    # get total time and report it
    print(f"total runtime: {time() - time_a:5f} s")


_BENCH_VOL_DESC = \
"""Benchmark to compare the computation of Black and Bachelier implied vols
with scipy.optimize.newton and a lightweight C implementation that can compute
the values for several points in parallel using OpenMP for multithreading.

"""
_BENCH_VOL_MAX_PTS_HELP = ("Maximum number of points to compute during "
                           "benchmark. Inputs are duplicated as many times as "
                           "needed to form a vector with <= max_pts options "
                           "data to send to solvers. Defaults to 1000000.")
_BENCH_VOL_X0_HELP = "Initial solver guess, default 0.5"
_BENCH_VOL_ABS_TOL_HELP = "Absolute solver tolerance, default 1.48e-8"
_BENCH_VOL_REL_TOL_HELP = "Relative solver tolerance, default 0"
_BENCH_VOL_MAX_ITER_HELP = "Maximum solver iterations, default 50"
_BENCH_VOL_MAX_THREADS_HELP = ("Max number of threads C implementation can use,"
                               " default -1 for all threads on system")


def bench_vol_main(args = None):
    """Main method for ``c_npy_demo.bench.vol`` benchmark script.
    
    To run from the interpreter, pass a list of string arguments to ``args``.
    
    :param args: List of string arguments to pass, default ``None``.
    :type args: list, optional
    """
    # overall routine execution time start
    time_a = time()
    # add arguments and parse
    arp = argparse.ArgumentParser(
        prog = "c_npy_demo.bench.vol", description = _BENCH_VOL_DESC,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )
    arp.add_argument("-mp", "--max-pts", default = 1000000, type = p_int,
                     help = _BENCH_VOL_MAX_PTS_HELP)
    arp.add_argument("-x0", default = 0.5, type = float,
                     help = _BENCH_VOL_X0_HELP)
    arp.add_argument("-at", "--abs-tol", default = 1.48e-8, type = float,
                     help = _BENCH_VOL_ABS_TOL_HELP)
    arp.add_argument("-rt", "--rel-tol", default = 0, type = float,
                     help = _BENCH_VOL_REL_TOL_HELP)
    arp.add_argument("-mi", "--max-iter", default = 50, type = p_int,
                     help = _BENCH_VOL_MAX_ITER_HELP)
    arp.add_argument("-mt", "--max-threads", default = -1, type = p_int,
                     help = _BENCH_VOL_MAX_THREADS_HELP)
    arp.add_argument("-v", "--verbose", action = "store_true",
                     help = _BENCH_VERBOSE_HELP)
    args = arp.parse_args(args = args)
    # extract args for convenience
    max_pts, x0, abs_tol, rel_tol, max_iter, max_threads, verbose = (
        args.max_pts, args.x0, args.abs_tol, args.rel_tol, args.max_iter,
        args.max_threads, args.verbose
    )
    print("this doesn't do anything yet. oops!")
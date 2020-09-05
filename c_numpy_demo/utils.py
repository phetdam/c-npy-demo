__doc__ = "Utilities for the ``c_numpy_demo`` package."

#import ctypes
import datetime
from functools import wraps
import inspect
import numpy as np

from ._cwrappers import vol_obj_args
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
            if hasattr(val, "__iter__") and (not isinstance(arg, (str, bytes))):
                kwargs[key] = np.array(val)
            else: kwargs[key] = np.array([val])
        # get max length of args and kwargs and extend length-1 arrays
        # args may be empty
        if (broadcast == True) and (len(args) > 0):
            max_len = max(tuple(map(lambda x: len(x), args)))
            for i in range(len(args)):
                arg = args[i]
                if arg.shape[0] == 1:
                    args[i] = np.array([arg[0] for _ in range(max_len)])
                elif arg.shape[0] == max_len: pass
                else:
                    raise ValueError("args array shape mismatch: "
                                     f"{arg.shape[0]} != {max_len} (max_len)")
        # kwargs may be empty
        if (broadcast == True) and (len(kwargs) > 0):
            # need to max with previous max_len
            max_len = max(max(tuple(map(lambda x: len(x), kwargs.values()))),
                          max_len)
            for key, val in kwargs.items():
                if val.shape[0] == 1:
                    kwargs[key] = np.array([val[0] for _ in range(max_len)])
                elif val.shape[0] == max_len: pass
                else:
                    raise ValueError("kwargs array shape mismatch: "
                                     f"{arg.shape[0]} != {max_len} (max_len)")
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
            if hasattr(val, "__iter__") and (not isinstance(arg, (str, bytes))):
                kwargs[key] = np.array(val).reshape((len(val), 1))
            else: kwargs[key] = np.array([val]).reshape((1, 1))
        # get max length of args and kwargs and extend length-1 arrays
        # args may be empty
        if len(args) > 0:
            max_len = max(tuple(map(lambda x: len(x), args)))
            for i in range(len(args)):
                arg = args[i]
                if arg.shape[0] == 1:
                    arg = np.array([arg[0] for _ in range(max_len)])
                    args[i] = arg.reshape((max_len, 1))
                elif arg.shape[0] == max_len: pass
                else:
                    raise ValueError("args array shape mismatch: "
                                     f"{arg.shape[0]} != {max_len} (max_len)")
        # kwargs may be empty
        if len(kwargs) > 0:
            # need to max with previous max_len
            max_len = max(max(tuple(map(lambda x: len(x), kwargs.values()))),
                          max_len)
            for key, val in kwargs.items():
                if val.shape[0] == 1:
                    val = np.array([val[0] for _ in range(max_len)])
                    kwargs[key] = val.reshape((max_len, 1))
                elif val.shape[0] == max_len: pass
                else:
                    raise ValueError("kwargs array shape mismatch: "
                                     f"{arg.shape[0]} != {max_len} (max_len)")
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
    
    .. note:: Do not call this directly. Use :func:`np_1d_broadcast` instead.
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
            return _np_float64_bcast_1d(f, axis = axis)
        
        return _wrapper
    
    return _np_float64_bcast_1d(f, axis = axis)
        
    
def options_csv_to_ndarray(fname):
    """Converts CSV file of European options data into :class:`numpy.ndarray`.
    
    Array type is :class:`numpy.float64`. The CSV file format must have the
    headers ``ccode``,  ``opt_price``, ``fut_price``, ``strike``, ``dfactor``,
    ``call_put``, ``opt_exp``, ``fut_exp``, ``rec_date``, in that order. The
    first column must be a string Bloomberg contract code, ex. EDZ22, the next
    4 columns must contain floating point values, ``call_put`` must contain
    only values of 1 or -1, and the last three columns must contain date
    strings in ``yyyy-mm-dd`` format. See ``./data/edo_atm_data.csv`` for an
    example.
    
    :param fname: Name of the CSV file to convert
    :type fname: str
    
    :returns: A :class:`numpy.ndarray` with shape ``(n, 6)``, where ``n`` is the
        number of data points in the CSV file. All elements should be of type
        :class:`numpy.float64`.
    :rtype: :class:`numpy.ndarray`
    """
    # read as ndarray of type string. first row (contract codes) is for
    # bookkeeping and is not needed, so drop it.
    data = np.genfromtxt(fname, delimiter = ",", dtype = str,
                         skip_header = 1)[:, 1:]
    # record number of data samples
    n_pts = data.shape[0]
    # yyyy-mm-dd str to date lambda
    Ymd2date = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")
    # get only options price, futures price, strike, and discount factor (float)
    ffour = data[:, :4].astype(float)
    # get call/put flags; reshape into column (as int)
    cpflags = data[:, 4].reshape((n_pts, 1)).astype(int)
    # get option expiration and date of recording, convert to datetime.date
    opt_exps = np.array(list(map(Ymd2date, data[:, -3])))
    rec_dates = np.array(list(map(Ymd2date, data[:, -1])))
    # get time to maturity in years, use 365. reshape into column
    ttms = np.array(list(map(lambda x: x.days, (opt_exps - rec_dates)))
                    ).astype(float).reshape((n_pts, 1)) / 365
    # arrange columns in order: options price, underlying price, strike, ttm,
    # discount factor, call/put flag
    out = np.concatenate((ffour[:, :3],
                          np.concatenate((ttms, ffour[:, 3].reshape((n_pts, 1)),
                                          cpflags), axis = 1)), axis = 1)
    return out


def ndarray2vol_obj_args_tuple(ar):
    """Create a tuple of :class:`vol_obj_args` from :class:`numpy.ndarray`.
    
    ``ar`` should be the output from :func:`options_csv_to_ndarray`.
    
    :param ar: A :class:`numpy.ndarray`, shape ``(n_obs, 6)`` with data type
        :class:`numpy.float64`. This should be the output of
        :func:`options_csv_to_ndarray`.
    :type ar: :class:`numpy.ndarray`
    :returns: A tuple of :class:`vol_obj_args` structs, length ``n_obs``.
    :rtype: tuple
    """
    # number of observations
    n_obs = ar.shape[0]
    # output array; populate using data from ar
    out = [None for _ in range(n_obs)]
    for i in range(n_obs):
        # unpack a row of the ndarray
        price, fwd, strike, ttm, df, is_call = ar[i, :]
        # need to convert is_call to int
        is_call = int(is_call)
        # write new vol_obj_args struct to out
        out[i] = vol_obj_args(price, fwd, strike, ttm , df, is_call)
    # return as tuple
    return tuple(out)


def almost_equal(x, y, tol = 1e-15):
    """``True`` if ``|x - y| <= tol``, ``False`` otherwise.
    
    :param x: First value to compare
    :type x: float
    :param y: Second value to compare
    :type y: float
    :param tol: Tolerance, defaults to ``1e-15``.
    :type tol: float, optional
    """
    return True if abs(x - y) <= tol else False
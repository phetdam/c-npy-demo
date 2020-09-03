__doc__ = "Utilities for the ``c_numpy_demo`` package."

from functools import wraps
import inspect
import numpy as np


def _np_1d_broadcast(f, axis = 0):
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
        if len(args) > 0:
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
        if len(kwargs) > 0:
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


def np_1d_broadcast(f = None, axis = 0):
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
    :type axis: int, optional
    :returns: Function that broadcasts all its args into :class:`numpy.ndarray`.
    :rtype: function
    """
    # if None, return decorator for f, else return decorated f
    if f is None:
        
        def _wrapper(_f):
            return _np_1d_broadcast(_f, axis = axis)
        
        return _wrapper
        
    return _np_1d_broadcast(f, axis = axis)
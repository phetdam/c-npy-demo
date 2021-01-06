__doc__ = """Implied volatility calculations, wrapping ``_ivlib.so``.

Provides Python wrappers for members in ``_ivlib.so`` using ctypes__.

.. __: https://docs.python.org/3/library/ctypes.html
"""

import ctypes
import numpy as np
import os.path


class vol_obj_args(ctypes.Structure):
    """C struct holding the fixed arguments used when solving for implied vol.
    
    The Python analogue to the ``vol_obj_args`` struct defined in
    ``euro_options.h``.
    
    :param price: True option price
    :type price: :class:`~ctypes.c_double`
    :param fwd: Forward level
    :type fwd: :class:`~ctypes.c_double`
    :param strike: Option strike
    :type strike: :class:`~ctypes.c_double`
    :param ttm: Time to maturity, in years
    :type ttm: :class:`~ctypes.c_double`
    :param df: Discount factor in ``(0, 1]``
    :type df: :class:`~ctypes.c_double`
    :param is_call: +/- 1 for call/put
    :type is_call: :class:`~ctypes.c_int`
    """
    _fields_ = [
        ("price", ctypes.c_double), ("fwd", ctypes.c_double),
        ("strike", ctypes.c_double), ("ttm", ctypes.c_double),
        ("df", ctypes.c_double), ("is_call", ctypes.c_int)
    ]


class scl_rf_res(ctypes.Structure):
    """C struct holding results returned by ``_halley_newton``.
    
    The Python analogue to the ``scl_rf_res`` struct defined in ``root_find.c``.
    
    .. note:: ``method`` and ``flag`` should be passed :class:`bytes` objects.
       Convert from :class:`str` to :class:`bytes` with :meth:`~str.encode`.
    
    :param res: Result, i.e. guess of where a root of the function is
    :type res: :class:`~ctypes.c_double`
    :param iters: Number of iterations taken to arrive at ``res``
    :type iters: :class:`~ctypes.c_int`
    :param converged: ``True`` if converged, ``False`` otherwise
    :type converged: :class:`~ctypes.c_bool`
    :param method: Optimization method used, either ``"halley"`` or ``"newton"``
        or ``None`` on error.
    :type method: :class:`~ctypes.c_char_p`
    :param flag: Gives reason for termination of ``_halley_newton``
    :type flag: :class:`~ctypes.c_char_p`
    """
    _fields_ = [
        ("res", ctypes.c_double), ("iters", ctypes.c_int),
        ("converged", ctypes.c_bool), ("method", ctypes.c_char_p),
        ("flag", ctypes.c_char_p)
    ]


# load _ivlib.so and set argument types for its functions
_ivlib = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + "/_ivlib.so")
# set arg and return types for black_price
_ivlib.black_price.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_int
]
_ivlib.black_price.restype = ctypes.c_double
# set arg and return types for bachelier_price
_ivlib.bachelier_price.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_int
]
_ivlib.bachelier_price.restype = ctypes.c_double
# set arg and return types for black_vega and black_volga
_ivlib.black_vega.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_int
]
_ivlib.black_vega.restype = ctypes.c_double
_ivlib.black_volga.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_int
]
_ivlib.black_volga.restype = ctypes.c_double
# set arg and return types for bachelier_vega and bachelier_volga
_ivlib.bachelier_vega.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double
]
_ivlib.bachelier_vega.restype = ctypes.c_double
_ivlib.bachelier_volga.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, 
    ctypes.c_double
]
_ivlib.bachelier_volga.restype = ctypes.c_double
# set arg and return types for _black_vol and _bachelier_vol. note that the
# return type is not ctypes.POINTER(scl_rf_res) but the whole struct itself.
_ivlib._black_vol.argtypes = [
    ctypes.POINTER(vol_obj_args), ctypes.c_int, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_bool
]
_ivlib._black_vol.restype = scl_rf_res
_ivlib._bachelier_vol.argtypes = [
    ctypes.POINTER(vol_obj_args), ctypes.c_int, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_bool
]
_ivlib._bachelier_vol.restype = scl_rf_res
# set arg and return types for black_vol_obj* and bachelier_vol_obj* methods
_ivlib.black_vol_obj.argtypes = [ctypes.c_double, ctypes.c_void_p]
_ivlib.black_vol_obj.restype = ctypes.c_double
_ivlib.black_vol_obj_d1.argtypes = [ctypes.c_double, ctypes.c_void_p]
_ivlib.black_vol_obj_d1.restype = ctypes.c_double
_ivlib.black_vol_obj_d2.argtypes = [ctypes.c_double, ctypes.c_void_p]
_ivlib.black_vol_obj_d2.restype = ctypes.c_double
_ivlib.bachelier_vol_obj.argtypes = [ctypes.c_double, ctypes.c_void_p]
_ivlib.bachelier_vol_obj.restype = ctypes.c_double
_ivlib.bachelier_vol_obj_d1.argtypes = [ctypes.c_double, ctypes.c_void_p]
_ivlib.bachelier_vol_obj_d1.restype = ctypes.c_double
_ivlib.bachelier_vol_obj_d2.argtypes = [ctypes.c_double, ctypes.c_void_p]
_ivlib.bachelier_vol_obj_d2.restype = ctypes.c_double
# set arg and return types for _imp_vol_vec
_ivlib._imp_vol_vec.argtypes = [
    ctypes.POINTER(vol_obj_args), ctypes.POINTER(ctypes.c_double),
    ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool
]
_ivlib._imp_vol_vec.restype = None


def black_vol_vec(
    odata, method = "halley", x0 = 0.5, tol = 1.48e-8, rtol = 0, maxiter = 50,
    n_threads = -1, mask_neg = True, debug = False
):
    """Computes Black implied vol for an array of :class:`vol_obj_args`.
    
    Wraps the C function ``_imp_vol_vec`` with ``vol_type`` set to
    ``BLACK_VOL_FLAG``.
    
    Docstring incomplete; see ``_ivlib/euro_options.c``.
    
    :param odata: A ``ctypes`` array of ``vol_obj_args``.
    :type odata: :class:`__main__.vol_obj_args_Array_*`
    :param method: The implied volatility solving method. Currently supports
        only ``"halley"`` and ``"newton"``.
    :type method: str, optional
    :param x0: Initial guess for implied volatility values, defaults to ``0.5``.
    :type x0: float, optional
    :param tol: Absolute tolerance for solver convergence.
    :type tol: float, optional
    :param rtol: Relative tolerance for solver convergence, i.e. differences
        between guesses. Defaults to ``0``.
    :type rtol: float, optional
    :param maxiter: Maximum number of iterations before convergence, default
        ``50``.
    :type maxiter: int, optional
    :param n_threads: Number of threads to use if computing in parallel. Pass
        ``-1`` to use all threads; defaults to ``1``. Automatically floors the
        number of threads requested to the max number of threads on the system.
    :type n_threads: int, optional
    :param mask_neg: ``True`` to replace negative implied vols with ``NAN``,
        ``False`` to keep all values. Defaults to ``True``.
    :type mark_neg: bool, optional
    :param debug: ``True`` for the solver to print its results for each data
        point, ``False`` for silence. Defaults to ``True``.
    :type debug: bool, optional
    :returns: A :class:`numpy.ndarray` of Black implied volatilities.
    :rtype: :class:`numpy.ndarray`
    """
    # get length of odata; must be positive
    n_pts = len(odata)
    if n_pts == 0:
        raise ValueError("odata must have positive length")
    # note that element type must be vol_obj_args
    if (not hasattr(odata, "_type_")) or (odata._type_ != vol_obj_args):
        raise TypeError("odata must be a ctypes array of vol_obj_args")
    # only allow halley or newton
    method = 0 if method == "halley" else (1 if method == "newton" else -1)
    if method == -1:
        raise ValueError("method must be \"halley\" or \"newton\"")
    # create an array of c_doubles the same length as odata
    _out = (ctypes.c_double * n_pts)()
    # compute implied vols using _imp_vol_vec
    _ivlib._imp_vol_vec(
        odata, _out, n_pts, 0, method, x0, tol, rtol, maxiter, n_threads,
        mask_neg, debug
    )
    # return numpy array from _out
    return np.ctypeslib.as_array(_out)


def bachelier_vol_vec(
    odata, method = "halley", x0 = 0.5, tol = 1.48e-8, rtol = 0, maxiter = 50,
    n_threads = 1, mask_neg = True, debug = False
):
    """Computes Bachelier implied vol for an array of :class:`vol_obj_args`.
    
    Wraps the C function ``_imp_vol_vec`` with ``vol_type`` set to
    ``BACHELIER_VOL_FLAG``.

    :param odata: A ``ctypes`` array of ``vol_obj_args``.
    :type odata: :class:`__main__.vol_obj_args_Array_*`
    :param method: The implied volatility solving method. Currently supports
        only ``"halley"`` and ``"newton"``.
    :type method: str, optional
    :param x0: Initial guess for implied volatility values, defaults to ``0.5``.
    :type x0: float, optional
    :param tol: Absolute tolerance for solver convergence.
    :type tol: float, optional
    :param rtol: Relative tolerance for solver convergence, i.e. differences
        between guesses. Defaults to ``0``.
    :type rtol: float, optional
    :param maxiter: Maximum number of iterations before convergence, default
        ``50``.
    :type maxiter: int, optional
    :param n_threads: Number of threads to use if computing in parallel. Pass
        ``-1`` to use all threads; defaults to ``1``. Automatically floors the
        number of threads requested to the max number of threads on the system.
    :type n_threads: int, optional
    :param mask_neg: ``True`` to replace negative implied vols with ``NAN``,
        ``False`` to keep all values. Defaults to ``True``.
    :type mark_neg: bool, optional
    :param debug: ``True`` for the solver to print its results for each data
        point, ``False`` for silence. Defaults to ``True``.
    :type debug: bool, optional
    :returns: A :class:`numpy.ndarray` of Black implied volatilities.
    :rtype: :class:`numpy.ndarray`
    """
    # get length of odata; must be positive
    n_pts = len(odata)
    if n_pts == 0:
        raise ValueError("odata must have positive length")
    # note that element type must be vol_obj_args
    if (not hasattr(odata, "_type_")) or (odata._type_ != vol_obj_args):
        raise TypeError("odata must be a ctypes array of vol_obj_args")
    # only allow halley or newton
    method = 0 if method == "halley" else (1 if method == "newton" else -1)
    if method == -1:
        raise ValueError("method must be \"halley\" or \"newton\"")
    # create an array of c_doubles the same length as odata
    _out = (ctypes.c_double * n_pts)()
    # compute implied vols using _imp_vol_vec
    _ivlib._imp_vol_vec(
        odata, _out, n_pts, 1, method, x0, tol, rtol, maxiter, n_threads,
        mask_neg, debug
    )
    # return numpy array from _out
    return np.ctypeslib.as_array(_out)
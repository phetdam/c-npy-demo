__doc__ = """Test ``_halley_newton`` function defined in ``root_find.c``.

.. note:: There is not much fear of segmentation faults due to memory allocation
   since the ``_halley_newton`` function only uses auto variables. It doesn't
   mean that a segfault won't happen during execution, however.
"""

import ctypes
import numpy as np
import os.path
import pytest
import scipy.optimize

from .._cwrappers import _ivlib, vol_obj_args
from ..utils import (almost_equal, ndarray2vol_obj_args_tuple,
                     options_csv_to_ndarray)

## -- Fixtures -----------------------------------------------------------------


@pytest.fixture(scope = "module")
def rf_stop_defaults():
    """Default abs + rel tolerance and max iterations for iterative root-finder.
    
    Same as the defaults for :func:`scipy.optimize.newton`.
    
    :returns: Absolute tolerance, relative tolerance, max iterations.
    :rtype: tuple
    """
    return (1.48e-8, 0, 50)


@pytest.fixture(scope = "module")
def options_ntm_data():
    """Creates a small panel of near the money options test data.
    
    Note that the data in each row will be in the parameter order specified by
    :class:`vol_obj_args`. Assumes that there are 365 days in a year. Only uses
    the near the money options data from ``data/edo_ntm_data.csv`` for brevity.
    
    :returns: A :class:`numpy.ndarray` of options data, shape ``(20, 6)``.
    :rtype: :class:`numpy.ndarray`
    """
    return options_csv_to_ndarray(os.path.dirname(__file__) + 
                                  "/../data/edo_ntm_data.csv")


@pytest.fixture(scope = "module")
def options_full_data():
    """Creates a panel of options test data for testing functions with.
    
    Note that the data in each row will be in the parameter order specified by
    :class:`vol_obj_args`. Assumes that there are 365 days in a year. Uses the
    the full options data from ``data/edo_full_data.csv``.
    
    :returns: A :class:`numpy.ndarray` of options data, shape ``(136, 6)``.
    :rtype: :class:`numpy.ndarray`
    """
    return options_csv_to_ndarray(os.path.dirname(__file__) +
                                  "/../data/edo_full_data.csv")


## -- Tests --------------------------------------------------------------------


@pytest.mark.parametrize("method", ["halley", "newton"])
@pytest.mark.parametrize("guess", [0.5, 0.7, 1]) # default guess is 0.5
@pytest.mark.parametrize("max_pts", [np.inf])
@pytest.mark.parametrize("py_debug,c_debug", [(False, False)])
def test_c_ntm_black_vol(options_ntm_data, rf_stop_defaults, method, guess,
                         max_pts, py_debug, c_debug):
    """Test solving for Black volatility using :func:`options_ntm_data` data.
    
    Directly calls the ``_black_vol`` function from ``_ivlib.so``.
    
    :param options_ntm_data: :func:`options_ntm_data` ``pytest`` fixture.
    :type options_ntm_data: :class:`numpy.ndarray`
    :param rf_stop_defaults: :func:``rf_stop_defaults` ``pytest`` fixture.
    :type rf_stop_defaults: tuple
    :param method: Method to use to solve. Only supports ``"halley"`` and
        ``"newton"`` currently. These will be converted to :class:`bytes`
        objects in the function to be suitable for passing to a C function.
    :type method: str
    :param guess: Starting guess for Black implied volatility.
    :type guess: float
    :param max_pts: Maximum number of data points (rows) from
        :func:`options_ntm_data` to run tests with. Must be positive, and if
        larger than ``options_ntm_data.shape[0]``, will be set to
        ``options_ntm_data.shape[0]`` automatically.
    :type max_pts: int
    :param py_debug: ``True`` for debugging output (print statements) from
        Python interpreter, ``False`` for silence. Useful for debugging large
        chunks of data points at once.
    :type py_debug: bool
    :param c_debug: ``True`` for debugging output (print to ``stdout``) from C
        function, ``False`` for silence except upon error. Useful for debugging
        single data points.
    :type c_debug: bool
    """
    # convert ndarray options_ntm_data to tuple of vol_obj_args
    voas = ndarray2vol_obj_args_tuple(options_ntm_data)
    # convert method to int
    method = 0 if "halley" else (1 if "newton" else -1)
    assert method in (0, 1), "method flag must be 0 (halley) or 1 (newton)"
    # get min of max_pts and length of voas
    n_pts = min(max_pts, len(voas))
    # implied volatility results (struct), values (vols), convergences, flags
    volrs = [None for _ in range(n_pts)]
    vols = np.zeros(n_pts)
    successes = np.zeros(n_pts).astype(bool)
    # can't use numpy.zeros(n_pts).astype(str) since the string gets clipped.
    # maybe raise an issue with the numpy maintainers?
    flags = [None for _ in range(n_pts)]
    # iterations per point solved
    iters = np.zeros(n_pts)
    # for the first n_pts, find the black vol
    for i in range(n_pts):
        vol_args = voas[i]
        # get result struct
        volrs[i] = _ivlib._black_vol(ctypes.byref(vol_args), method,
                                     guess, *rf_stop_defaults, c_debug)
        # get actual vol value
        vols[i] = volrs[i].res
        # get convergence information (True if converged)
        successes[i] = volrs[i].converged
        # reason for failure/success (decode bytes object)
        flags[i] = volrs[i].flag.decode()
        # number of iterations required
        iters[i] = volrs[i].iters
    # if py_debug, print vols, convergences, number of iterations, and flags
    if py_debug == True:
        print(f"vols:\n{vols}")
        print(f"successes:\n{successes}")
        print(f"iters:\n{iters}")
        print(f"flags:\n{flags}")
    # test for convergence. must always work with near the money options
    assert sum(successes) == n_pts


@pytest.mark.parametrize("method", ["halley", "newton"])
@pytest.mark.parametrize("guess", [0.5, 0.7, 1]) # default guess is 0.5
@pytest.mark.parametrize("max_pts", [np.inf])
@pytest.mark.parametrize("py_debug,c_debug", [(False, False)])
def test_c_ntm_bachelier_vol(options_ntm_data, rf_stop_defaults, method, guess,
                             max_pts, py_debug, c_debug):
    """Test solving for Bachelier vols using :func:`options_ntm_data` data.
    
    Directly calls the ``_bachelier_vol`` function from ``_ivlib.so``.
    
    :param options_ntm_data: :func:`options_ntm_data` ``pytest`` fixture.
    :type options_ntm_data: :class:`numpy.ndarray`
    :param rf_stop_defaults: :func:``rf_stop_defaults` ``pytest`` fixture.
    :type rf_stop_defaults: tuple
    :param method: Method to use to solve. Only supports ``"halley"`` and
        ``"newton"`` currently. These will be converted to :class:`bytes`
        objects in the function to be suitable for passing to a C function.
    :type method: str
    :param guess: Starting guess for Bachelier implied volatility.
    :type guess: float
    :param max_pts: Maximum number of data points (rows) from
        :func:`options_ntm_data` to run tests with. Must be positive, and if
        larger than ``options_ntm_data.shape[0]``, will be set to
        ``options_ntm_data.shape[0]`` automatically.
    :type max_pts: int
    :param py_debug: ``True`` for debugging output (print statements) from
        Python interpreter, ``False`` for silence. Useful for debugging large
        chunks of data points at once.
    :type py_debug: bool
    :param c_debug: ``True`` for debugging output (print to ``stdout``) from C
        function, ``False`` for silence except upon error. Useful for debugging
        single data points.
    :type c_debug: bool
    """
    # convert ndarray options_ntm_data to tuple of vol_obj_args
    voas = ndarray2vol_obj_args_tuple(options_ntm_data)
    # convert method to int
    method = 0 if "halley" else (1 if "newton" else -1)
    assert method in (0, 1), "method flag must be 0 (halley) or 1 (newton)"
    # get min of max_pts and length of voas
    n_pts = min(max_pts, len(voas))
    # implied volatility results (struct), values (vols), convergences, flags
    volrs = [None for _ in range(n_pts)]
    vols = np.zeros(n_pts)
    successes = np.zeros(n_pts).astype(bool)
    # can't use numpy.zeros(n_pts).astype(str) since the string gets clipped.
    # maybe raise an issue with the numpy maintainers?
    flags = [None for _ in range(n_pts)]
    # iterations per point solved
    iters = np.zeros(n_pts)
    # for the first n_pts, find the bachelier vol
    for i in range(n_pts):
        vol_args = voas[i]
        # get result struct
        volrs[i] = _ivlib._bachelier_vol(ctypes.byref(vol_args), method,
                                         guess, *rf_stop_defaults, c_debug)
        # get actual vol value
        vols[i] = volrs[i].res
        # get convergence information (True if converged)
        successes[i] = volrs[i].converged
        # reason for failure/success (decode bytes object)
        flags[i] = volrs[i].flag.decode()
        # number of iterations required
        iters[i] = volrs[i].iters
    # if py_debug, print vols, convergences, number of iterations, and flags
    if py_debug == True:
        print(f"vols:\n{vols}")
        print(f"successes:\n{successes}")
        print(f"iters:\n{iters}")
        print(f"flags:\n{flags}")
    # test for convergence. must always work on near the money options
    assert sum(successes) == n_pts


@pytest.mark.parametrize("method", ["halley", "newton"])
@pytest.mark.parametrize("guess", [0.5, 0.7, 1]) # default guess is 0.5
@pytest.mark.parametrize("max_pts,py_debug", [(np.inf, False)])
def test_rf_c_against_scipy(options_ntm_data, rf_stop_defaults, method, guess,
                            max_pts, py_debug):
    """Test C root-finding implementation against :func:`scipy.optimize.newton`.
    
    Use both Black and Bachelier objective functions.
    
    :param options_full_data: :func:`options_full_data` ``pytest`` fixture.
    :type options_full_data: :class:`numpy.ndarray`
    :param rf_stop_defaults: :func:``rf_stop_defaults` ``pytest`` fixture.
    :type rf_stop_defaults: tuple
    :param method: Method to use to solve. Only supports ``"halley"`` and
        ``"newton"`` currently.
    :type method: str
    :param guess: Starting guess for Bachelier implied volatility.
    :type guess: float
    :param max_pts: Maximum number of data points (rows) from
        :func:`options_ntm_data` to run tests with. Must be positive, and if
        larger than ``options_full_data.shape[0]``, will be set to
        ``options_full_data.shape[0]`` automatically.
    :type max_pts: int
    :param py_debug: ``True`` for debugging output (print statements) from
        Python interpreter, ``False`` for silence. Useful for debugging large
        chunks of data points at once.
    :type py_debug: bool
    """
    # convert ndarray options_ntm_data to tuple of vol_obj_args
    voas = ndarray2vol_obj_args_tuple(options_ntm_data)
    # method flag to pass to C method
    c_method = 0 if "halley" else (1 if "newton" else -1)
    assert c_method in (0, 1), "c_method flag must be 0 (halley) or 1 (newton)"
    # get min of max_pts and length of voas
    n_pts = min(max_pts, len(voas))
    # implied volatility results (struct) from C functions
    volrs = [None for _ in range(n_pts)]
    # vols from scipy and C implementations; will contain 2-tuples of floats
    vols = [None for _ in range(n_pts)]
    # iterations per point solved
    iters = [None for _ in range(n_pts)]
    # for the first n_pts, find the black vol
    for i in range(n_pts):
        vol_args = voas[i]
        # get result struct from C implementation
        volrs[i] = _ivlib._black_vol(ctypes.byref(vol_args), c_method, guess,
                                     *rf_stop_defaults, False)
        # get RootResults from scipy implementation
        if method == "halley":
            _, spy_res = scipy.optimize.newton(
                _ivlib.black_vol_obj, guess, fprime = _ivlib.black_vol_obj_d1,
                fprime2 = _ivlib.black_vol_obj_d2,
                args = (ctypes.byref(vol_args),), tol = rf_stop_defaults[0],
                rtol = rf_stop_defaults[1], maxiter = rf_stop_defaults[2],
                full_output = True
            )
        elif method == "newton":
            _, spy_res = scipy.optimize.newton(
                _ivlib.black_vol_obj, guess, fprime = _ivlib.black_vol_obj_d1,
                args = (ctypes.byref(vol_args),), tol = rf_stop_defaults[0],
                rtol = rf_stop_defaults[1], maxiter = rf_stop_defaults[2],
                full_output = True
            )
        # get actual vol values and number of iterations required
        vols[i] = (volrs[i].res, spy_res.root)
        # number of iterations required
        iters[i] = (volrs[i].iters, spy_res.iterations)
    # if py_debug, print vols, convergences, number of iterations, and flags
    if py_debug == True:
        print(f"vols:\n{vols}")
        print(f"iters:\n{iters}")
    # test that results are identical for both implementations
    same_roots = list(map(lambda x: almost_equal(*x), vols))
    assert sum(same_roots) == n_pts, "Not all roots results were identical"
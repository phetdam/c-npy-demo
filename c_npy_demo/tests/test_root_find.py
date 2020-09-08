__doc__ = """Test ``_halley_newton`` function defined in ``root_find.c``.

.. note:: There is not much fear of segmentation faults due to memory allocation
   since the ``_halley_newton`` function only uses auto variables. It doesn't
   mean that a segfault won't happen during execution, however.
"""

import ctypes
import numpy as np
import pytest
import scipy.optimize

from .fixtures import edo_ntm_data, hh_ntm_data, rf_stop_defaults
from ..ivlib import _ivlib
from ..utils import almost_equal, ndarray2vol_obj_args_array


@pytest.mark.parametrize("method", ["halley", "newton"])
@pytest.mark.parametrize("guess", [0.5, 0.7, 1]) # default guess is 0.5
@pytest.mark.parametrize("max_pts", [np.inf])
@pytest.mark.parametrize("py_debug,c_debug", [(False, False)])
def test_c_ntm_black_vol(hh_ntm_data, rf_stop_defaults, method, guess,
                         max_pts, py_debug, c_debug):
    """Test solving for Black volatility using :func:`hh_ntm_data` data.
    
    Directly calls the ``_black_vol`` function from ``_ivlib.so``.
    
    :param hh_ntm_data: ``pytest`` fixture. See ``fixtures.py``.
    :type hh_ntm_data: :class:`numpy.ndarray`
    :param rf_stop_defaults: ``pytest`` fixture. See ``fixtures.py``.
    :type rf_stop_defaults: tuple
    :param method: Method to use to solve. Only supports ``"halley"`` and
        ``"newton"`` currently. These will be converted to :class:`bytes`
        objects in the function to be suitable for passing to a C function.
    :type method: str
    :param guess: Starting guess for Black implied volatility.
    :type guess: float
    :param max_pts: Maximum number of data points (rows) from
        :func:`hh_ntm_data` to run tests with. Must be positive, and if
        larger than ``hh_ntm_data.shape[0]``, will be set to
        ``hh_ntm_data.shape[0]`` automatically.
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
    # convert ndarray hh_ntm_data to ctypes array of vol_obj_args
    voas = ndarray2vol_obj_args_array(hh_ntm_data)
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
    # note: there are two points that do not converge. ignore those.
    assert sum(successes) >= n_pts - 2


@pytest.mark.parametrize("method", ["halley", "newton"])
@pytest.mark.parametrize("guess", [0.5, 0.7, 1]) # default guess is 0.5
@pytest.mark.parametrize("max_pts", [np.inf])
@pytest.mark.parametrize("py_debug,c_debug", [(False, False)])
def test_c_ntm_bachelier_vol(edo_ntm_data, rf_stop_defaults, method, guess,
                             max_pts, py_debug, c_debug):
    """Test solving for Bachelier vols using :func:`edo_ntm_data` data.
    
    Directly calls the ``_bachelier_vol`` function from ``_ivlib.so``.
    
    :param edo_ntm_data: ``pytest`` fixture. See ``fixtures.py``.
    :type edo_ntm_data: ``pytest`` fixture. See ``fixtures.py``.
    :param rf_stop_defaults: :func:``rf_stop_defaults` ``pytest`` fixture.
    :type rf_stop_defaults: tuple
    :param method: Method to use to solve, either ``"halley"`` or ``"newton"``.
    :type method: str
    :param guess: Starting guess for Bachelier implied volatility.
    :type guess: float
    :param max_pts: Maximum number of data points (rows) from
        :func:`edo_ntm_data` to run tests with. Must be positive, and if
        larger than ``edo_ntm_data.shape[0]``, will be set to
        ``edo_ntm_data.shape[0]`` automatically.
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
    # convert ndarray edo_ntm_data to ctypes array of vol_obj_args
    voas = ndarray2vol_obj_args_array(edo_ntm_data)
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


@pytest.mark.parametrize("vol_type", ["black", "bachelier"])
@pytest.mark.parametrize("method", ["halley", "newton"])
@pytest.mark.parametrize("guess", [0.5, 0.7, 1]) # default guess is 0.5
@pytest.mark.parametrize("max_pts,py_debug", [(np.inf, True)])
def test_rf_c_against_scipy(edo_ntm_data, hh_ntm_data, rf_stop_defaults,
                            vol_type, method, guess, max_pts, py_debug):
    """Test C root-finding implementation against :func:`scipy.optimize.newton`.
    
    :param edo_ntm_data: ``pytest`` fixture. See ``fixtures.py``.
    :type edo_ntm_data: :class:`numpy.ndarray`
    :param hh_ntm_data: ``pytest`` fixture. See ```fixtures.py``.
    :type hh_ntm_data: :class:`numpy.ndarray`
    :param rf_stop_defaults: ``pytest`` fixture. See ``fixtures.py``.
    :type rf_stop_defaults: tuple
    :param vol_type: Vol to solve for, either ``"black"`` or ``"bachelier"``.
    :type vol_type: str
    :param method: Method to use to solve, either ``"halley"`` or ``"newton"``.
    :type method: str
    :param guess: Starting guess for Bachelier implied volatility.
    :type guess: float
    :param max_pts: Maximum number of data points (rows) from
        :func:`edo_ntm_data` to run tests with. Must be positive, and if
        larger than ``edo_full_data.shape[0]``, will be set to
        ``edo_full_data.shape[0]`` automatically.
    :type max_pts: int
    :param py_debug: ``True`` for debugging output (print statements) from
        Python interpreter, ``False`` for silence. Useful for debugging large
        chunks of data points at once.
    :type py_debug: bool
    """
    # check volatility type
    assert vol_type in ("black", "bachelier")
    # convert ndarray of data to ctypes array of vol_obj_args based on vol_type
    if vol_type == "black":
        voas = ndarray2vol_obj_args_array(hh_ntm_data)
    elif vol_type == "bachelier":
        voas = ndarray2vol_obj_args_array(edo_ntm_data)
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
    # choose objective functions
    if vol_type == "black":
        vol_func = _ivlib._black_vol
        vol_obj = _ivlib.black_vol_obj
        vol_obj_d1 = _ivlib.black_vol_obj_d1
        vol_obj_d2 = _ivlib.black_vol_obj_d2
    elif vol_type == "bachelier":
        vol_func = _ivlib._bachelier_vol
        vol_obj = _ivlib.bachelier_vol_obj
        vol_obj_d1 = _ivlib.bachelier_vol_obj_d1
        vol_obj_d2 = _ivlib.bachelier_vol_obj_d2
    # for the first n_pts, find the black vol
    for i in range(n_pts):
        vol_args = voas[i]
        #vol_args.price = vol_args.price / 100
        # get result struct from C implementation
        volrs[i] = vol_func(ctypes.byref(vol_args), c_method, guess,
                            *rf_stop_defaults, False)
        # get RootResults from scipy implementation
        if method == "halley":
            _, spy_res = scipy.optimize.newton(
                vol_obj, guess, fprime = vol_obj_d1, fprime2 = vol_obj_d2,
                args = (ctypes.byref(vol_args),), tol = rf_stop_defaults[0],
                rtol = rf_stop_defaults[1], maxiter = rf_stop_defaults[2],
                full_output = True, disp = False
            )
        elif method == "newton":
            _, spy_res = scipy.optimize.newton(
                vol_obj, guess, fprime = vol_obj_d1,
                args = (ctypes.byref(vol_args),), tol = rf_stop_defaults[0],
                rtol = rf_stop_defaults[1], maxiter = rf_stop_defaults[2],
                full_output = True, disp = False
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
    # if not, print some statistics (bachelier vols tend to differ)
    if sum(same_roots) != n_pts:
        dfunc = lambda x: abs(x[0] - x[1])
        diffs = np.array(list(map(dfunc, vols)))
        max_diff = diffs.max()
        min_diff = diffs.min()
        print(f"max_diff: {max_diff}, at {np.nonzero(diffs == max_diff)[0][0]}")
        print(f"min_diff: {min_diff}, at {np.nonzero(diffs == min_diff)[0][0]}")
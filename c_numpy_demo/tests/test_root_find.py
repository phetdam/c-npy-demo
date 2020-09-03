__doc__ = """Test ``_halley_newton`` function defined in ``root_find.c``.

.. note:: There is not much fear of segmentation faults due to memory allocation
   since the ``_halley_newton`` function only uses auto variables. It doesn't
   mean that a segfault won't happen during execution, however.
"""

import ctypes
import numpy as np
import pytest

from .._cwrappers import _ivlib, scl_rf_res, vol_obj_args
#from .ctests
from ..demo import (CALL_PRICES, FUT_PRICE, OPT_EXP_DATE, PUT_PRICES,
                    RECORD_DATE, STRIKES)

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
def options_demo_data():
    """Creates a panel of options test data for testing functions with.
    
    Note that the data in each row will be in the parameter order specified by
    :class:`vol_obj_args`. Assumes that there are 365 days in a year.
    
    :returns: A :class:`numpy.ndarray` of options data, shape ``(n_obs, 6)``.
    :rtype: :class:`numpy.ndarray`
    """
    # get all prices together and reshape into column vector
    opt_prices = np.concatenate((CALL_PRICES, PUT_PRICES))
    opt_prices = opt_prices.reshape((opt_prices.shape[0], 1))
    # get forward level (constant) as column vector
    fwds = FUT_PRICE * np.ones(opt_prices.shape)
    # get strikes (need to double count) as column vector
    strikes = np.concatenate((STRIKES, STRIKES)).reshape(opt_prices.shape)
    # get expiration date in years and turn into column array
    ttms = (OPT_EXP_DATE - RECORD_DATE).days / 365 * np.ones(opt_prices.shape)
    # assume discount factors are 0.99765
    dfs = 0.99765 * np.ones(opt_prices.shape)
    # get call/put flags as column vector
    cp_flags = np.concatenate((np.ones((CALL_PRICES.shape[0], 1)),
                               np.ones((PUT_PRICES.shape[0], 1))))
    # get output and return concatenated on axis 1
    return np.concatenate((opt_prices, fwds, strikes, ttms, 
                           dfs, cp_flags), axis = 1)


# note that params
@pytest.fixture(scope = "module")
def vol_obj_args_tuple(options_demo_data):
    """Create a tuple of :class:`vo_obj_args` from :func:`options_demo_data`.
    
    :param options_demo_data: :func:`options_demo_data` ``pytest`` fixture.
    :type options_demo_data: :class:`numpy.ndarray`
    :rtype: tuple
    """
    out = [None for _ in range(options_demo_data.shape[0])]
    for i in range(options_demo_data.shape[0]):
        price, fwd, strike, ttm, df, is_call = options_demo_data[i]
        # need to convert is_call to int
        is_call = int(is_call)
        # write to out
        out[i] = vol_obj_args(price, fwd, strike, ttm , df, is_call)
    # return as tuple
    return tuple(out)


## -- Tests --------------------------------------------------------------------


@pytest.mark.parametrize("method", ["halley", "newton"])
@pytest.mark.parametrize("guess", [0.5])
@pytest.mark.parametrize("max_pts", [np.inf])
@pytest.mark.parametrize("debug", [False]) # True to allow C function printf
def test_c_black_vol(vol_obj_args_tuple, rf_stop_defaults, method, guess,
                     max_pts, debug):
    """Test solving for Black volatility using :func:`vol_obj_args_tuple`.
    
    Directly calls the ``_black_vol`` function from ``_ivlib.so``.
    
    :param vol_obj_args_tuple: :func:`vol_obj_args_tuple` ``pytest`` fixture.
    :type vol_obj_args_tuple: tuple
    :param rf_stop_defaults: :func:``rf_stop_defaults` ``pytest`` fixture.
    :type rf_stop_defaults: tuple
    :param method: Method to use to solve. Only supports ``"halley"`` and
        ``"newton"`` currently. These will be converted to :class:`bytes`
        objects in the function to be suitable for passing to a C function.
    :type method: str
    :param guess: Starting guess for Black implied volatility.
    :type guess: float
    :param max_pts: Maximum number of data points from
        :func:`vol_obj_args_tuple` to run tests with. Must be positive, and if
        larger than ``len(vol_obj_args_tuple)``, will be set to
        ``len(vol_obj_args_tuple)`` automatically.
    :type max_pts: int
    :param debug: ``True`` for debugging output (print to ``stdout``) from C
        function, ``False`` for silence except upon error.
    :type debug: bool
    """
    method = 0 if "halley" else (1 if "newton" else -1)
    assert method in (0, 1), "method flag must be 0 (halley) or 1 (newton)"
    # change method to int since method is determined with flags
    # get min of max_pts and length of vol_obj_args_tuple
    n_pts = min(max_pts, len(vol_obj_args_tuple))
    # implied volatility results (struct), values (vols), convergences, flags
    vol_structs = [None for _ in range(n_pts)]
    vols = np.zeros(n_pts)
    successes = np.zeros(n_pts).astype(bool)
    # can't use numpy.zeros(n_pts).astype(str) since the string gets clipped.
    # maybe raise an issue with the numpy maintainers?
    flags = [None for _ in range(n_pts)]
    # iterations per point solved
    iters = np.zeros(n_pts)
    # for the first n_pts, find the black vol
    for i in range(n_pts):
        vol_args = vol_obj_args_tuple[i]
        # get result struct
        vol_structs[i] = _ivlib._black_vol(ctypes.byref(vol_args), method,
                                           guess, *rf_stop_defaults, debug)
        # get actual vol value
        vols[i] = vol_structs[i].res
        # get convergence information (True if converged)
        successes[i] = vol_structs[i].converged
        # reason for failure/success (decode bytes object)
        flags[i] = vol_structs[i].flag.decode()
        # number of iterations required
        iters[i] = vol_structs[i].iters
    # print vol values, convergences, number of iterations, and flags
    print(f"vols:\n{vols}")
    print(f"successes:\n{successes}")
    print(f"iters:\n{iters}")
    print(f"flags:\n{flags}")
    # test for convergence (might fail sometimes)
    assert sum(successes) == n_pts
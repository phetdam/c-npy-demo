__doc__ = """Test ``_imp_vol_vec function defined in ``euro_options.c``.

Function may have OpenMP support enabled for multithreading on local machine.
"""

import numpy as np
import pytest
import scipy.optimize

from .fixtures import options_full_data, options_ntm_data, rf_stop_defaults
from ..ivlib import bachelier_vol_vec, black_vol_vec
from ..utils import almost_equal, ndarray2vol_obj_args_array


@pytest.mark.parametrize("n_threads", [-1])
@pytest.mark.parametrize("method", ["halley", "newton"])
@pytest.mark.parametrize("guess", [0.5, 0.7, 1]) # default guess is 0.5
@pytest.mark.parametrize("mult", [1, 10, 100])
@pytest.mark.parametrize("py_debug", [False])
def test_ntm_imp_vol_vec(options_ntm_data, rf_stop_defaults, method, guess,
                         n_threads, mult, py_debug):
    """Test ``_imp_vol_vec`` function from ``_ivlib.so`` on near-the-money data.
    
    Near-the-money data is better behaved and less likely to give weird results.
    
    :param options_ntm_data: ``pytest`` fixture. See ``fixtures.py``.
    :type options_ntm_data: :class:`numpy.ndarray`
    :param options_full_data: ``pytest`` fixture. See ``fixtures.py``.
    :param rf_stop_defaults: ``pytest`` fixture. See ``fixtures.py``.
    :type rf_stop_defaults: tuple
    :param method: Method to use to solve, either ``"halley"`` or ``"newton"``.
    :type method: str
    :param guess: Starting guess for implied volatility solving.
    :type guess: float
    :param n_threads: Number of threads to use when multithreading.
    :type n_threads: int
    :param mult: Number of times to duplicate input data. For scaling up.
    :type mult: int
    :param py_debug: ``True`` to print values to :class:`sys.stdout`, ``False``
        for silence.
    :type py_debug: bool

    black_vol_vec(odata, method = "halley", x0 = 0.5, tol = 1.48e-8, rtol = 0,
                  maxiter = 50, n_threads = -1, mask_neg = True, debug = False):
    """
    # convert ndarray options data to ctypes array of vol_obj_args
    voas = ndarray2vol_obj_args_array(options_ntm_data, mult = mult)
    # test running black_vol_vec and bachelier_vol_vec on these inputs
    ntm_out_bl = black_vol_vec(
        voas, method = method, x0 = guess, tol = rf_stop_defaults[0],
        rtol = rf_stop_defaults[1], maxiter = rf_stop_defaults[2],
        n_threads = n_threads, mask_neg = False
    )
    if py_debug == True: print(ntm_out_bl)
    ntm_out_ba = bachelier_vol_vec(
        voas, method = method, x0 = guess, tol = rf_stop_defaults[0],
        rtol = rf_stop_defaults[1], maxiter = rf_stop_defaults[2],
        n_threads = n_threads, mask_neg = False
    )
    if py_debug == True: print(ntm_out_ba)
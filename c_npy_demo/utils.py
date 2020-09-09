__doc__ = "Utilities for the ``c_numpy_demo`` package."

import datetime
import numpy as np

from .ivlib import vol_obj_args
        
    
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


def ndarray2vol_obj_args_array(ar, mult = 1):
    """Get :class:`vol_obj_args` array from :class:`numpy.ndarray`.
    
    ``ar`` should be given by :func:`options_csv_to_ndarray`. The returned array
    is a ``ctypes`` array built from the :class:`vol_obj_args` type. Optionally,
    ``mult`` can be used to repeat the original array multiple times.
    
    :param ar: A :class:`numpy.ndarray`, shape ``(n_obs, 6)`` with data type
        :class:`numpy.float64`. This should be the output of
        :func:`options_csv_to_ndarray`.
    :type ar: :class:`numpy.ndarray`
    :param mult: Number of times to repeat original data received from ``ar``.
        This is useful if existing data is limited but scale needs to be tested.
    :type mult: int, optional
    :returns: A ``ctypes`` array of ``n_obs`` :class:`vol_obj_args` structs.
    :rtype: :class:`__main__.vol_obj_args_Array_*`
    """
    # number of observations
    n_obs = ar.shape[0]
    # mult must be positive
    if mult < 1: raise ValueError("mult must be a positive int")
    # output array; populate using data from ar mult times
    # note: need to wrap n_obs and mult in parentheses or else the array ends
    # up becoming a mult-size array of n_obs arrays, which is NOT what we want!
    out = (vol_obj_args * (n_obs * mult))()
    for m in range(mult):
        for i in range(n_obs):
            # unpack a row of the ndarray
            price, fwd, strike, ttm, df, is_call = ar[i, :]
            # need to convert is_call to int
            is_call = int(is_call)
            # write new vol_obj_args struct to out
            out[m * n_obs + i] = vol_obj_args(price, fwd, strike, ttm , df,
                                              is_call)
    return out


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
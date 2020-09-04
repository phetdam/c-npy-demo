__doc__ = "Module for ``_ivlib.so`` and C extension demonstrations."

import datetime
import numpy as np

from ._cwrappers import _ivlib
from .utils import np_1d_broadcast


@np_1d_broadcast(axis = 1)
def black_price(fwd, strike, ttm, ivol, df = 1, is_call = True):
    """Return the Black model option price.
    
    See ``euro_options.c`` for C implementation of ``black_price``. Broadcasts
    args to :class:`numpy.ndarray`, so is slower with scalars.
    
    :param fwd: Current level of the forward (underlying) in units of price.
    :type fwd: float or iterable
    :param strike: Option strike, must be same units as fwd
    :type fwd: float or iterable
    :param ttm: Time to maturity in years
    :type ttm: float or iterable
    :param ivol: Black implied volatility / 100, i.e. percentage / 100
    :type ivol: float or iterable
    :param df: Optional discount factor(s) in ``(0, 1]``.
    :type df: float or iterable, optional
    :param is_call: ``True`` if call option, ``False`` otherwise.
    :type is_call: bool or iterable, optional
    :returns: Black implied volatilities.
    :rtype: float or :class:`numpy.ndarray`
    """
    ## decorated, so treat all args as numpy.ndarray ##
    # flatten and convert is_call to int
    is_call = np.array(list(map(lambda x: 1 if x == True else -1,
                                is_call.ravel())))
    # concatenate along axis 1
    data = np.concatenate((fwd, strike, ttm, ivol, df), axis = 1)
    # common length
    our_len = data.shape[0]
    # allocate some memory for results
    out = np.zeros(our_len)
    # get black price for each element (slow loop)
    for i in range(our_len):
        out[i] = _ivlib.black_price(*data[i], is_call[i])
    return out


@np_1d_broadcast(axis = 1)
def bachelier_price(fwd, strike, ttm, ivol, df = 1, is_call = True):
    """Return the Bachelier model option price.
    
    See ``euro_options.c`` for C implementation of ``black_price``. Broadcasts
    args to :class:`numpy.ndarray`, so is slower with scalars.
    
    :param fwd: Current level of the forward (underlying) in units of price.
    :type fwd: float or iterable
    :param strike: Option strike, must be same units as fwd
    :type fwd: float or iterable
    :param ttm: Time to maturity in years
    :type ttm: float or iterable
    :param ivol: Black implied volatility / 100, i.e. percentage / 100
    :type ivol: float or iterable
    :param df: Optional discount factor(s) in ``(0, 1]``.
    :type df: float or iterable, optional
    :param is_call: ``True`` if call option, ``False`` otherwise.
    :type is_call: bool or iterable, optional
    :returns: Bachelier implied volatilities.
    :rtype: float or :class:`numpy.ndarray`
    """
    ## decorated, so treat all args as numpy.ndarray ##
    # flatten and convert is_call to int
    is_call = np.array(list(map(lambda x: 1 if x == True else -1,
                                is_call.ravel())))
    # concatenate along axis 1
    data = np.concatenate((fwd, strike, ttm, ivol, df), axis = 1)
    # common length
    our_len = data.shape[0]
    # allocate some memory for results
    out = np.zeros(our_len)
    # get black price for each element (slow loop)
    for i in range(our_len):
        out[i] = _ivlib.bachelier_price(*data[i], is_call[i])
    return out
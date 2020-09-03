__doc__ = "Module for ``ctypes`` and C extension demos."

import datetime
import numpy as np

from ._cwrappers import _ivlib
from .utils import np_1d_broadcast

# demo data for implied volatility computation, recorded on 2020-08-31.

FUT_PRICE = 99.72
"""EDH23 price on Aug 31, 2020, in IMM index points [#]_.

EDH23 is the March 2023 expiration Eurodollar futures contract. The futures
contract expires on March 13, 2023, the second London bank business day before
the third Wednesday of the month.

.. [#] IMM index points have the same units as percentage.
"""

RECORD_DATE = datetime.date(2020, 8, 31)
"Date when the data was recorded."

OPT_EXP_DATE = datetime.date(2021, 3, 12)
"Options' expiration date."

FUT_EXP_DATE = datetime.date(2023, 3, 13)
"Options' underlying futures' expiration date."

CALL_PRICES = np.array([61.5, 49.5, 38, 27, 17.5, 9.5, 5, 3, 2, 1.5])
"""2EH21 call prices on Aug 31, 2020, in IMM index points.

2EH21 is the March 2021 expiration 2-year Eurodollar mid-curve option contract.
The options contract expires on March 12, 2021, the Friday immediately before
the third Wednesday of the month.
"""

STRIKES = np.array([99.125, 99.25, 99.375, 99.5, 99.625, 99.75, 99.875, 100,
                    100.125, 100.25])
"""2EH21 at the money strikes on Aug 31, 2020, in IMM index points.

2EH21 expires on March 12, 2021.
"""

PUT_PRICES = np.array([1, 1.5, 2.5, 4, 7, 11.5, 19.5, 30, 41.5, 53.5])
"""2EH21 put prices on Aug 31, 2020, in IMM index points.

2EH21 expires on March 12, 2021.    
"""

YIELD_CURVE = np.array([0.08, 0.10, 0.11, 0.13, 0.12, 0.14, 0.15, 0.28, 0.50,
                        0.72, 1.26, 1.49])
"""The Treasury constant maturity yield curve on Aug 31, 2020.

Maturities are, in order: 1 Mo, 2 Mo, 3 Mo, 6 Mo, 1 Yr, 2 Yr, 3 Yr, 5 Yr, 7 Yr,
10 Yr, 20 Yr, 30 Yr.
"""


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
    :rtype: :class:`numpy.ndarray`
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
    :rtype: :class:`numpy.ndarray`
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
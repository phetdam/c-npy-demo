__doc__ = "Module for ``ctypes`` and C extension demos."

import ctypes
import numpy as np

from ._cwraps import _ivlib

# demo data for implied volatility computation, recorded on 2020-31-08.

FUT_PRICE = 99.72
"""EDH23 price on Aug 31, 2020, in IMM index points [#]_.

EDH23 is the March 2023 expiration Eurodollar futures contract. The futures
contract expires on March 13, 2023, the second London bank business day before
the third Wednesday of the month.

.. [#] IMM index points have the same units as percentage.
"""

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


def black_price(fwd, strike, ttm, ivol, df = 1, is_call = True):
    """Return the Black model option price.
    
    See ``implied_vol.c`` for details.
    
    :param fwd:
    :param strike:
    :param ttm:
    :param ivol:
    :param df: Discount factor.
    :type df: float, optional
    :param is_call: ``True`` if call option, ``False`` otherwise.
    :type is_call: bool, optional
    """
    # convert is_call to int
    is_call = 1 if True else -1
    # return black price
    return _ivlib.black_price(fwd, strike, ttm, ivol, df, is_call)
__doc__ = """Provide Python wrappers for ``_ivlib.so``.

Sets arg and return types for the C functions in ``_ivlib.so`` that are loaded
using the ctypes__ foreign function interface and provides a Python wrappper for
the ``vol_obj_args`` struct defined in ``euro_options.h``.

.. __: https://docs.python.org/3/library/ctypes.html
"""

import ctypes
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
    _fields_ = [("price", ctypes.c_double), ("fwd", ctypes.c_double),
                ("strike", ctypes.c_double), ("ttm", ctypes.c_double),
                ("df", ctypes.c_double), ("is_call", ctypes.c_int)]


# load _ivlib.so and set argument types for its functions
_ivlib = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + "/_ivlib.so")
# set arg and return types for black_price
_ivlib.black_price.argtypes = [ctypes.c_double, ctypes.c_double,
                               ctypes.c_double, ctypes.c_double,
                               ctypes.c_double, ctypes.c_int]
_ivlib.black_price.restype = ctypes.c_double
# set arg and return types for bachelier_price
_ivlib.bachelier_price.argtypes = [ctypes.c_double, ctypes.c_double,
                                   ctypes.c_double, ctypes.c_double,
                                   ctypes.c_double, ctypes.c_int]
_ivlib.bachelier_price.restype = ctypes.c_double
# set arg and return types for black_vega and black_volga
_ivlib.black_vega.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                              ctypes.c_double, ctypes.c_double, ctypes.c_int]
_ivlib.black_vega.restype = ctypes.c_double
_ivlib.black_volga.argtypes = [ctypes.c_double, ctypes.c_double,
                               ctypes.c_double, ctypes.c_double,
                               ctypes.c_double, ctypes.c_int]
_ivlib.black_volga.restype = ctypes.c_double
# set arg and return types for bachelier_vega and bachelier_volga
_ivlib.bachelier_vega.argtypes = [ctypes.c_double, ctypes.c_double,
                                  ctypes.c_double, ctypes.c_double,
                                  ctypes.c_double]
_ivlib.bachelier_vega.restype = ctypes.c_double
_ivlib.bachelier_volga.argtypes = [ctypes.c_double, ctypes.c_double,
                                   ctypes.c_double, ctypes.c_double, 
                                   ctypes.c_double]
_ivlib.bachelier_volga.restype = ctypes.c_double
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
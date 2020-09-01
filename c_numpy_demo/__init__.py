__doc__ = """Top-level ``c_numpy_demo`` ``__init__.py``.

Performs some extra work by setting arg and return types for the C functions
in ``implied_vol.so`` that are loaded using the ctypes__ foreign function
interface.

.. __: https://docs.python.org/3/library/ctypes.html
"""

import ctypes
import os.path

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
# set arg and rturn types for bachelier_vega and bachelier_volga
_ivlib._bachelier_vega.argtypes = [ctypes.c_double, ctypes.c_double,
                                   ctypes.c_double, ctypes.c_double,
                                   ctypes.c_double, ctypes.c_int]
_ivlib._bachelier_vega.restype = ctypes.c_double
_ivlib._bachelier_volga.argtypes = [ctypes.c_double, ctypes.c_double,
                                    ctypes.c_double, ctypes.c_double, 
                                    ctypes.c_double, ctypes.c_int]
_ivlib._bachelier_volga.restype = ctypes.c_double
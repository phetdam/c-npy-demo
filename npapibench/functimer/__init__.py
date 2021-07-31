"""An internal C extension function timing module.

Inspired by ``timeit`` but times callables with arguments without using
executable string statements. This way, timings of several callables that share
the same args avoids multiple setup statement calls.

Times obtained by default using time.perf_counter. Functional API only.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

# pylint: disable=no-name-in-module
from ._timeapi import autorange, timeit_plus, timeit_once, timeit_repeat
from ._timeresult import TimeResult
from ._timeunit import MAX_PRECISION, VALID_UNITS, VALID_UNIT_BASES
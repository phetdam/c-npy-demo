__doc__ = """Global fixtures used when running tests with ``pytest``.

Why not use a ``conftest.py``? Because I might forget where fixtures are from...
"""

import pytest
import os.path

from ..utils import options_csv_to_ndarray


@pytest.fixture(scope = "session")
def rf_stop_defaults():
    """Default abs + rel tolerance and max iterations for iterative root-finder.
    
    Same as the defaults for :func:`scipy.optimize.newton`.
    
    :returns: Absolute tolerance, relative tolerance, max iterations.
    :rtype: tuple
    """
    return (1.48e-8, 0, 50)


@pytest.fixture(scope = "session")
def options_ntm_data():
    """Creates a small panel of near the money options test data.
    
    Note that the data in each row will be in the parameter order specified by
    :class:`vol_obj_args`. Assumes that there are 365 days in a year. Only uses
    the near the money options data from ``data/edo_ntm_data.csv`` for brevity.
    
    :returns: A :class:`numpy.ndarray` of options data, shape ``(80, 6)``.
    :rtype: :class:`numpy.ndarray`
    """
    return options_csv_to_ndarray(os.path.dirname(__file__) + 
                                  "/../data/edo_ntm_data.csv")


@pytest.fixture(scope = "session")
def options_full_data():
    """Creates a panel of options test data for testing functions with.
    
    Note that the data in each row will be in the parameter order specified by
    :class:`vol_obj_args`. Assumes that there are 365 days in a year. Uses the
    the full options data from ``data/edo_full_data.csv``.
    
    :returns: A :class:`numpy.ndarray` of options data, shape ``(422, 6)``.
    :rtype: :class:`numpy.ndarray`
    """
    return options_csv_to_ndarray(os.path.dirname(__file__) +
                                  "/../data/edo_full_data.csv")
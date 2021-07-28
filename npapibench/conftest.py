"""Top-level conftest.py for global pytest fixtures.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def default_seed():
    """Default numpy.random.Generator seed.

    Returns
    -------
    int
    """
    return 7


@pytest.fixture
def default_rng(default_seed):
    """Default PRNG to use with any method that requires numpy Generator.

    Since this fixture is function scope, there is fresh PRNG per test.

    Returns
    -------
    numpy.random.Generator
    """
    return np.random.default_rng(default_seed)


@pytest.fixture(scope="session")
def tuple_replace():
    """Return a new tuple from an existing tuple with element modifications.

    Changes to the new tuple are specified with (idx, value) pairs passed
    after the tuple. The new tuple will be same length as the original tuple.

    Parameters
    ----------
    orig : tuple
        The original tuple whose elements are to be replaced
    *args
        Pairs of tuples in format (idx, value) where idx indexes orig and value
        gives the value that orig[i] should be replaced with.

    Returns
    -------
    tuple
        New tuple with all the specified modifications.
    """
    def _tuple_replace(orig, *args):
        orig_ = list(orig)
        for idx, val in args:
            orig_[idx] = val
        return tuple(orig_)

    _tuple_replace.__doc__ = tuple_replace.__doc__

    return _tuple_replace


@pytest.fixture(scope="session")
def pyvalue_raise():
    """A pytest.raises wrapper for catching ValueErrors.

    Parameters
    ----------
    match : str, default=None
        Regular expression to match exception error text against.

    Returns
    -------
    RaisesContext
        pytest context manager for catching exception-raising blocks.
    """
    def _pyvalue_raise(match=None):
        return pytest.raises(ValueError, match=match)

    _pyvalue_raise.__doc__ = pyvalue_raise.__doc__

    return _pyvalue_raise


@pytest.fixture(scope="session")
def pytype_raise():
    """A pytest.raises wrapper for catching TypeErrors.

    Parameters
    ----------
    match : str, default=None
        Regular expression to match exception error text against.

    Returns
    -------
    RaisesContext
        pytest context manager for catching exception-raising blocks.
    """
    def _pytype_raise(match=None):
        return pytest.raises(TypeError, match=match)

    _pytype_raise.__doc__ = pyvalue_raise.__doc__

    return _pytype_raise
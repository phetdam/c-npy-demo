__doc__ = """pytest test fixtures for the unit tests.

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


@pytest.fixture
def test_mat(default_rng):
    """The test array used for unit tests.

    Values are randomly sampled from standard Gaussian distribution.

    Returns
    -------
    numpy.ndarray
        Shape (10, 3, 10), entries sampled from standard normal distribution.
    """
    return default_rng.normal(size=(10, 3, 10))

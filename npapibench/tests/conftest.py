"""pytest fixtures for benchmark script and cimpl unit tests.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import pytest


@pytest.fixture
def test_mat(default_rng):
    """The test array used for unit tests.

    Values are randomly sampled from standard Gaussian distribution. See the
    top-level package conftest.py for the default_rng fixture.

    Returns
    -------
    numpy.ndarray
        Shape (10, 3, 10), entries sampled from standard normal distribution.
    """
    return default_rng.normal(size=(10, 3, 10))

"""Unit tests for benchmarking script.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import pytest

# pylint: disable=relative-beyond-top-level
from ..bench import comma_list_to_shape


def test_comma_list_to_shape_sanity():
    """Test input sanity of bench.comma_list_to_shape."""
    with pytest.raises(TypeError, match="s must be a string"):
        comma_list_to_shape(-100)
    with pytest.raises(ValueError, match="s is empty"):
        comma_list_to_shape("")


def test_comma_list_to_shape_split():
    """Test bench.comma_list_to_shape splitting behavior.

    ValueError raised if conversion fails.
    """
    # non-int element in split list
    with pytest.raises(ValueError, match="invalid literal"):
        comma_list_to_shape("4,thebigboi,100")
    # empty string in split list (too many commas)
    with pytest.raises(ValueError, match="invalid literal"):
        comma_list_to_shape(",100,10")


def test_comma_list_to_shape_validity():
    """Check that bench.comma_list_to_shape shape is valid.

    Shape must contain only positive integers.
    """
    # negative integer
    with pytest.raises(ValueError, match="axis [0-9]+ of shape"):
        comma_list_to_shape("100,-19,90,1")
    # zero
    with pytest.raises(ValueError, match="axis [0-9]+ of shape"):
        comma_list_to_shape("100,0,4,99")
    # note: comma_list_to_shape silently converts comma-separated lists of
    # float-like strings to lists. this isn't a problem in practice.
"""Python implementation of zero mean unit variance scaling function.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""


def stdscale(ar, ddof=0):
    """Center and scale numpy.ndarray to zero mean, unit variance.

    Treats the array like a single flattened array and computes the mean and
    standard deviation over all the elements.

    Parameters
    ----------
    ar : numpy.ndarray
        Arbitrary numpy.ndarray that can be converted to NPY_DOUBLE type
    ddof : int, default=0
        Delta degrees of freedom, i.e. so that the divisor used in standard
        deviation computation is ``n_obs - ddof``.

    Returns
    -------
    numpy.ndarray
        A new numpy.ndarray centered and scaled with zero mean, unit variance,
        with type NPY_DOUBLE, flags NPY_ARRAY_CARRAY, same shape as ar.
    """
    return (ar - ar.mean()) / ar.std(ddof=ddof)
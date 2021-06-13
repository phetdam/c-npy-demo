__doc__ = "Python implementation of zero mean unit variance scaling function."


def stdscale(ar, ddof=0):
    """Center and scale :class:`numpy.ndarray` to zero mean, unit variance.

    Treats the array like a single flattened array and computes the mean and
    standard deviation over all the elements.

    :param ar: Arbitrary :class:`numpy.ndarray`
    :type ar: :class:`numpy.ndarray`
    :param ddof: Delta degrees of freedom, i.e. so that the divisor used in
        standard deviation computation is ``n_obs - ddof``.
    :type ddof: int, optional
    :returns: A :class:`numpy.ndarray` centered and scaled to have zero mean
        and unit variance.
    :rtype: :class:`numpy.ndarray`
    """
    return (ar - ar.mean()) / ar.std(ddof=ddof)
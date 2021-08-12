"""Benchmarking module for npapibench.cimpl and npapibench.pyimpl.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from functools import partial
import numpy as np

# pylint: disable=no-name-in-module,relative-beyond-top-level
from . import cimpl, pyimpl
from .functimer import timeit_plus

_BENCH_DESC = """\
Benchmarking script comparing the Python and C stdscale implementations.

Compares the speed of the Python and C stdscale implementations on a relatively
large random multidimensional numpy.ndarray using the timeit module. Random
ndarray is created with a call to numpy.random.normal.

Timing is performed with the npapibench.functimer subpackage, a purpose-built
timing library implemented using C extensions. The reason timeit is not used is
because subsequent calls to timeit cannot be used to time different functions
with shared arguments. Since the requested numpy.ndarray can be rather large,
using timeit would result in double allocation. The npapibench.functimer
library allows sharing arguments between separate timing calls for different
functions and so avoids this double allocation issue.\
"""
_HELP_SHAPE = """\
The shape of the random ndarray to allocate, default 40,5,10,10,20,5. Shape
must be specified with a comma-separated list of positive integers.\
"""
_HELP_NUMBER = """\
Number of times to execute each function in a trial. If not specified, this is
automatically determined by npapibench.functimer.timeit_plus using the same
strategy employed by timeit.Timer.autorange.\
"""
_HELP_REPEAT = "Number of timing trials for each function, default 5"
_HELP_UNIT = """\
Time unit to display result with. If not specified, this is automatically
determined by npapibench.functimer.timeit_plus using the same strategy employed
by timeit.main, the method invoked by `python3 -m timeit`. Available options
are sec, msec, usec, nsec, the same options given by timeit.main.\
"""
_HELP_PRECISION = "Number of decimals printed in output, default 1"


def comma_list_to_shape(s):
    """Parse a string of comma-separated ints into a valid numpy shape.

    Trailing commas will raise an error.

    Parameters
    ----------
    s : str
        A string of comma-separated positive integers.

    Returns
    -------
    tuple
    """
    if not isinstance(s, str):
        raise TypeError("s must be a string")
    if s == "":
        raise ValueError("s is empty")
    # split by comma into an array of strings and try to convert to int
    shape = tuple(map(int, s.split(",")))
    # check that each shape member is valid (positive int), return if valid
    for i in range(len(shape)):
        if shape[i] < 1:
            raise ValueError(f"axis {i} of shape {shape} must be positive")
    return shape


def main(args=None):
    """Main entry point for the benchmarking script.

    Parameters
    ----------
    args : list, default=None
        List of string arguments to pass to argparse.ArgumentParser.parse_args.
    """
    # instantiate ArgumentParse and add arguments. help width is set to 80 cols
    # although we are technically using the private argparse API.
    arp = ArgumentParser(
        description=_BENCH_DESC,
        formatter_class=partial(RawDescriptionHelpFormatter, width=80)
    )
    arp.add_argument(
        "-s", "--shape", default=(40, 5, 10, 10, 20, 5),
        type=comma_list_to_shape, help=_HELP_SHAPE
    )
    arp.add_argument("-n", "--number", type=int, help=_HELP_NUMBER)
    arp.add_argument("-r", "--repeat", type=int, help=_HELP_REPEAT)
    arp.add_argument("-u", "--unit", help=_HELP_UNIT)
    # use count and default 0 to count verbosity levels
    arp.add_argument(
        "-p", "--precision", default=1, type=int, help=_HELP_PRECISION
    )
    # parse arguments
    args = arp.parse_args(args=args)
    # collect named args for npapibench.functimer.timeit_plus that are not None
    # except for the shape argument. timeit_args will be directly unpacked
    # into the npapibench.functimer.timeit_plus function.
    timeit_args = {}
    for k, v in vars(args).items():
        if (k != "shape") and (v is not None):
            timeit_args[k] = v
    # print shape and number of elements in array + allocate random array
    print(f"numpy.ndarray shape {args.shape}, size {np.prod(args.shape)}")
    ar = np.random.normal(size=args.shape)
    # get results for pyimpl.stdscale and cimpl.stdscale + print results
    py_res = timeit_plus(pyimpl.stdscale, (ar,), **timeit_args)
    print(f"pyimpl.stdscale -- {py_res.brief}")
    c_res = timeit_plus(cimpl.stdscale, (ar,), **timeit_args)
    print(f" cimpl.stdscale -- {c_res.brief}")
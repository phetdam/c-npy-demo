__doc__ = "Benchmarking :mod:`c_npy_demo.cscale` and :mod:`c_npy_demo.pyscale`."

import argparse
import gc
import numpy as np
import timeit

_BENCH_DESC = """\
Benchmarking script comparing the Python and C stdscale implementations.

Compares the speed of the Python and C stdscale implementations on a relatively
large random multidimensional numpy.ndarray using the timeit module. Random
ndarray is created with a call to numpy.random.normal.

Timing with timeit uses the main method exposed when timeit is run as a module
from the command line, i.e. with `python3 -m timeit [args] [statements]`.

Note: Using the timeit module here is quite inefficient because the setup, i.e.
the ndarray allocation, can be quite expensive depending on the size of the
ndarray. Therefore, for large ndarrays, there will be a delay before each of the
two calls to timeit.main when the array is [re-]allocated. In subsequent
releases, a dedicating comparative timeit-based timing implementation will be
written to prevent this double allocation from being performed.\
"""
_HELP_SHAPE = """\
The shape of the random ndarray to allocate, default 40,5,10,10,50,5. Shape
must be specified with a comma-separated list of positive integers.\
"""
_HELP_NUMBER = "Number of times to execute the statement, passed to timeit.main"
_HELP_REPEAT = "Number of times to repeat statement, passed to timeit.main"
_HELP_UNIT = "Time unit for timer output, passed to timeit.main"
_HELP_VERBOSE = "Print raw timing results, passed to timeit.main"


def comma_list_to_shape(s):
    """Tries to parse a comma-separated list of ints into a valid numpy shape.

    Trailing commas will raise an error.

    :param s: A string of comma-separated positive integers.
    :type s: str
    :rtype: tuple
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


def main(args = None):
    """Main entry point for the benchmarking script.

    :param args: List of string arguments to pass to
        :meth:`argparse.ArgumentParser.parse_args`
    :type args: list, optional
    """
    # instantiate ArgumentParse and add arguments
    arp = argparse.ArgumentParser(
        description = _BENCH_DESC,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )
    arp.add_argument(
        "-s", "--shape", default = (40, 5, 10, 10, 50, 5),
        type = comma_list_to_shape, help = _HELP_SHAPE
    )
    arp.add_argument("-n", "--number", help = _HELP_NUMBER)
    arp.add_argument("-r", "--repeat", help = _HELP_REPEAT)
    arp.add_argument("-u", "--unit", help = _HELP_UNIT)
    # use count and default 0 to count verbosity levels
    arp.add_argument(
        "-v", "--verbose", action = "count", default = 0, help = _HELP_VERBOSE
    )
    # parse arguments
    args = arp.parse_args(args = args)
    # collect args for timeit.main
    timeit_args = []
    if args.number is not None:
        timeit_args += ["-n", args.number]
    if args.repeat is not None:
        timeit_args += ["-r", args.repeat]
    if args.unit is not None:
        timeit_args += ["-u", args.unit]
    if args.verbose is not None:
        timeit_args += ["-v"] * args.verbose
    # print shape and number of elements in array
    print(f"numpy.ndarray shape {args.shape}, size {np.prod(args.shape)}")
    # setup of the random ndarray and execution statement for stdscale
    ex_setup = f"import numpy as np\nar = np.random.normal(size = {args.shape})"
    ex_snippet = "stdscale(ar)"
    # call timeit.main with timeit_args for both pyscale and cscale stdscale
    timeit.main(
        timeit_args + [
            "-s", "from c_npy_demo.pyscale import stdscale\n" + ex_setup,
            ex_snippet
        ]
    )
    gc.collect()
    timeit.main(
        timeit_args + [
            "-s", "from c_npy_demo.cscale import stdscale\n" + ex_setup,
            ex_snippet
        ]
    )
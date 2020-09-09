__doc__ = "Run the benchmarks through their main methods + pass some options."

import pytest

from ..bench import bench_ext_main, bench_vol_main


@pytest.mark.parametrize("seed", [7])
@pytest.mark.parametrize("nvecs", [3, 7])
@pytest.mark.parametrize("ncons", [2, 6])
@pytest.mark.parametrize("vlen", [100000]) # 1000000 takes way too long (tested)
@pytest.mark.parametrize("ntrs", [5, 7])
def test_bench_ext(seed, nvecs, ncons, vlen, ntrs):
    """Run :func:`~c_npy_demo.bench.bench_ext_main` verbosely with args.
    
    :param seed: Seed for the :class:`~numpy.random.RandomState` used for
        pseudo-random number generation.
    :type seed: int
    :param nvecs: Number of input vectors to generate
    :type nvecs: int
    :param ncons: Number of input constants to broadcast to generate
    :type ncons: int
    :param vlen: Length of the vectors
    :type ncons: int
    :param ntrs: Number of trials to run for measuring execution time
    :type ntrs: int
    """
    # argument string
    arg_str = f"-s {seed} -nv {nvecs} -nc {ncons} -vl {vlen} -nt {ntrs} -v"
    # run bench_ext_main with specified arguments
    bench_ext_main(args = arg_str.split())


def test_bench_vol():
    """Run :func:`~c_npy_demo.bench.bench_vol_main` verbosely with args.
    
    Doesn't do anything interesting yet.
    """
    # argument string
    #arg_str = f""
    # run bench_vol_main with specified arguments
    #bench_vol_main(arg_str.split())
    bench_vol_main()
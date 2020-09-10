#!/usr/bin/bash
# run tests on built package using pytest, install into root with pip3, and then
# run the setuptools-generated scripts verbosely as test.

# exit if command has non-zero exit status
set -e

# test built package with pytest and then test setuptools-generated benchmarks
run_tests() {
    # run test suite; tests for setuptools-generated scripts are skipped
    pytest
    # install
    make install
    # run extension and vol benchmarks, verbosely (with defaults)
    c_npy_demo.bench.ext -v
    c_npy_demo.bench.vol -v
}


# check if on travis or not
if ! [ $TRAVIS ]
then
    echo "WARNING: script intended to be run on Travis CI"
else
    # check that pkg_test directory exists; if not print error and exit
    if ! [ -d pkg_test ]
    then
        echo "error: cannot find directory pkg_test. exiting"
    else
        # run all tests
        run_tests
    fi
fi
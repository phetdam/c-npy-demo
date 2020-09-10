#!/usr/bin/bash
# builds c_numpy_demo._np_bcast, the C extension part of the package, as well as
# the non-Python extension C code as a shared object.

# check if on travis or not
if ! [ $TRAVIS ]
then
    echo "WARNING: script intended to be run on Travis CI. exiting"
else
    # build c_numpy_demo._np_bcast (see Makefile). completed package will be
    # located in directory pkg_test. then pytest is run
    make build
fi
#!/usr/bin/bash
# builds c_numpy_demo.cext, the C extension part of the package, as well as
# the non-Python extension C code as a shared object.
# note: DO NOT source this script.

# check if on travis or not
if ! [ $TRAVIS ]
then
    echo "WARNING: script intended to be run on Travis CI"
    exit
fi

# build c_numpy_demo.cext, where package is in directory pkg_test (see Makefile)
make build
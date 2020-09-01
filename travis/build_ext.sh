#!/usr/bin/bash
# builds c_numpy_demo.cext, the C extension part of the package, as well as
# the non-Python extension C code as a shared object.

# check if on travis or not
if ! [ $TRAVIS ]
then
    echo "WARNING: script intended to be run on Travis CI"
    exit
fi
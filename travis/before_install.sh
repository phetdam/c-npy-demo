#!/usr/bin/bash
# upgrade required packages. use in before_install in .travis.yml
# note: DO NOT source this script.

# if not on travis ci, exit
if ! [ $TRAVIS ]
then
    echo "WARNING: intended for execution on Travis CI only. exiting"
    exit
fi

# install newest gcc
sudo apt-get update
sudo apt-get install gcc
# check python version and upgrade required tools
python3 --version
pip3 install --upgrade pip
pip3 install --upgrade pytest
pip3 install --upgrade setuptools
pip3 install --upgrade wheel
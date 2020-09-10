#!/usr/bin/bash
# build source and wheel distributions for manylinux1 using docker and then
# upload to either test PyPI or real PyPI depending on DEPLOY_DRY + DEPLOY_WET.
# we can't use travis' built in PyPI deployment since we need the wheel to be
# built on the docker image. travis deploy is fine for pure Python project.

# quit if command has nonzero status
set -e

# check if on travis or not
if ! [ $TRAVIS ]
then
    echo "WARNING: script intended to be run on Travis CI. exiting"
# if on travis, then check if we can deploy
elif [ -e DEPLOY_DRY ] || [ -e DEPLOY_WET ]
then
    echo "deployment sequence started"
    # upgrade setuptools, wheel, and twine
    pip3 install --upgrade setuptools
    pip3 install --upgrade wheel
    pip3 install --upgrade twine
    # run script to build wheels on manylinux docker image. PLAT, DOCKER_IMAGE,
    # and DOCKER_MNT are defined in .travis.yml.
    docker container run --rm -e PLAT=$PLAT DOCKER_MNT=$DOCKER_MNT \
        -v `pwd`:$DOCKER_MNT $DOCKER_IMAGE $DOCKER_MNT/travis/build_dist.sh
    # deploy to test pypi and/or pypi, depending on DEPLOY_DRY or DEPLOY_WET
    if [ -e DEPLOY_DRY ]
    then
        twine upload -r testpypi -u __token__ -p $TOK_TEST dist/*
        echo "uploaded to Test PyPI"
    fi
    if [ -e DEPLOY_WET ]
    then
        twine upload --u __token__ -p $TOK_PYPI dist/*
        echo "uploaded to [real] PyPI"
    fi
fi
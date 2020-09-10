#!/usr/bin/bash
# run tests on built wheels in docker using pytest, install into root in venv
# with pip3, and then run the setuptools-generated scripts verbosely as test.

# exit if command has non-zero exit status
set -e

# test wheels and run setuptools-generated benchmark scripts
run_venv_tests() {
    # directory for python executables
    PY_BIN=$1
    # create virtual environment and activate it
    $PY_BIN/python3 -m venv .venv
    source .venv/bin/activate
    echo "running `python3 --version` tests and benchmarks"
    # install requirements from requirements.txt and generated wheel
    $PY_BIN/pip3 install -r $DOCKER_MNT/travis/requirements.txt
    $PY_BIN/pip3 install c_npy_demo --no-index --only-binary :all: \
        -f $DOCKER_MNT/dist
    # run test suite; tests for setuptools-generated scripts are skipped
    pytest -rsxXP $DOCKER_MNT/c_npy_demo/tests
    # run extension and vol benchmarks, verbosely (with defaults)
    c_npy_demo.bench.ext -v
    c_npy_demo.bench.vol -v
    # deactivate and delete virtual environment
    deactivate
    rm -rvf .venv
}


# exit if we don't have PLAT or DOCKER_MNT defined
if ! [ $PLAT ]
then
    echo "variable PLAT not defined; exiting"
elif ! [ $DOCKER_MNT ]
then
    echo "variable DOCKER_MNT not defined; exiting"
# also exit if we are on travis
elif [ $TRAVIS ]
then
    echo "should not be run on travis but on manylinux docker image. exiting"
# else for python versions 3.6 to 3.8, run pytest on the built wheels
else
    # cd to $HOME so pytest cannot read pytest.ini
    cd $HOME
    # for python versions 3.6-3.8 (control using regex)
    for PY_BIN in /opt/python/cp3*/bin:
    do
        if $PY_BIN/python3 --version | grep "3\.[6-8]\.[0-9]"
        then
            # create + activate venv, install requirements and wheel, run test
            # suite with pytest, run benchmark scripts, then deactivate/clean up
            run_venv_tests $PY_BIN
        else
            echo "no tests/benchmarks to run for `$PY_BIN/python3 --version`"
        fi
    done
fi
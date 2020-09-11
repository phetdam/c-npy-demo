#!/usr/bin/bash
# run tests on built wheels in docker using pytest, install into root in venv
# with pip3, and then run the setuptools-generated scripts verbosely as test.

# exit if command has non-zero exit status
set -e

# test wheels and run setuptools-generated benchmark scripts
run_venv_tests() {
    # directory for python executables, directory for wheels
    PY_BIN=$1
    WHEEL_DIR=$2
    # create virtual environment in ~ and activate it
    $PY_BIN/python3 -m venv ~/.venv
    source .venv/bin/activate
    echo "running `python3 --version` tests and benchmarks"
    # install requirements from requirements.txt
    $PY_BIN/pip3 install -r $DOCKER_MNT/travis/requirements.txt
    # get last major + minor python version number
    PY_VER=$PY_BIN/python3 --version | sed -e s/"Python "// -e s/"\.[0-9]$"// \
        -e s/"\."//
    # try to find the wheel; if it can't be found, skip
    if find $WHEEL_DIR/c_npy_demo-*$PY_VER-$PY_VER*-manylinux*.whl
    then
        # get the name of the wheel
        PY_WHL=`find $WHEEL_DIR/c_npy_demo-*$PY_VER-$PY_VER*-manylinux*.whl`
        echo "found wheel $PY_WHL for `$PY_BIN/python3 --version`"
        # install using pip3 directly from the wheel
        $PY_BIN/pip3 install $PY_WHL
        # run test suite; tests for setuptools-generated scripts are skipped
        pytest -rsxXP $DOCKER_MNT/c_npy_demo/tests
        # run extension and vol benchmarks, verbosely (with defaults)
        c_npy_demo.bench.ext -v
        c_npy_demo.bench.vol -v
    else
        echo "couldn't find `$PY_BIN/python3 --version` manylinux wheel"
    fi
    # deactivate and delete virtual environment
    deactivate
    rm -rf ~/.venv
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
    # cd to home so pytest cannot read pytest.ini
    cd ~
    # for python versions 3.6-3.8 (control using regex)
    for PY_BIN in /opt/python/cp3*/bin
    do
        if $PY_BIN/python3 --version | grep "3\.[6-8]\.[0-9]"
        then
            # create + activate venv, install requirements and wheel, run test
            # suite with pytest, run benchmark scripts, then deactivate/clean up
            run_venv_tests $PY_BIN $DOCKER_MNT/dist
        else
            echo "no tests/benchmarks to run for `$PY_BIN/python3 --version`"
        fi
    done
fi
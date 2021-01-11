#!/usr/bin/bash
# install wheels venvs on docker + run tests and benchmark on the wheels

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
    $PY_BIN/pip3 install -r $DOCKER_MNT/tools/requirements.txt
    # get last major + minor python version number
    PY_VER=`$PY_BIN/python3 --version | sed s/"Python "//`
    PY_VER=`echo $PY_VER | cut -c 1-3 | sed s/"\."//`
    # try to find the wheel; if it can't be found, skip. example wheel is
    # c_npy_demo-0.0.1.dev1-cp36-cp36m-manylinux1_x86_64.whl, for x86_64.
    if find $WHEEL_DIR/c_npy_demo-*cp$PY_VER-cp$PY_VER*-manylinux*.whl
    then
        # get the name of the wheel
        PY_WHL=`find $WHEEL_DIR/c_npy_demo-*cp$PY_VER-cp$PY_VER*-manylinux*.whl`
        echo "found wheel $PY_WHL for `$PY_BIN/python3 --version`"
        # install using pip3 directly from the wheel
        $PY_BIN/pip3 install $PY_WHL
        # run pytest
        PY_WHL_BASE=`$PY_BIN/pip3 show c-npy-demo | grep "Location:"`
        PY_WHL_BASE=`echo $PY_WHL_BASE | sed s/"Location: "//`
        # run test suite using --pyargs x.y, x.y a python package. note that
        # manylinux image does not have PATH properly configured, so we need to
        # run pytest from the directory the python interpreter are located in.
        $PY_BIN/pytest -rsxXP --pyargs c_npy_demo.tests
        # run the benchmark script to verify installation. again note use of
        # absolute path; also pass small shape so runtime is shorter
        ls $PY_BIN # debug: check that the benchmark really is installed there
        #$PY_BIN/c_npy_demo.bench -s 20,10
    else
        echo "couldn't find `$PY_BIN/python3 --version` manylinux wheel"
    fi
    # deactivate and delete virtual environment
    deactivate
    rm -rf ~/.venv
}


# exit if we don't have DOCKER_MNT or DOCKER_IMAGE defined
if [ ! $DOCKER_MNT ]
then
    echo "DOCKER_MNT not defined; exiting"
elif [ ! $DOCKER_IMAGE ]
then
    echo "DOCKER_IMAGE not defined; exiting"
# also exit if on github actions runner
elif [ $GITHUB_ACTIONS ]
then
    echo "should only be run on manylinux docker image. exiting"
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
            echo "no tests to run for `$PY_BIN/python3 --version`"
        fi
    done
fi
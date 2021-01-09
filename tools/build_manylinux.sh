#!/usr/bin/bash
# build manylinux wheels. this should be run on manylinux1 docker image. passed
# DOCKER_MNT and DOCKER_IMAGE from github actions.

# exit if command has nonzero exit status
set -e

# subroutine to build cpython 3 wheels. note we use separate venvs to build,
build_cp3_wheels() {
    echo "building cpython3 wheels"
    # on the manylinux1 images, python installed in /opt/python/*/bin. see
    # https://github.com/pypa/manylinux README.md for more details.
    for PY_BIN in /opt/python/cp3*/bin
    do
        # only accept python versions 3.6-3.8
        if $PY_BIN/python3 --version | grep "3\.[6-8]\.[0-9]"
        then
            # start virtual environment in home
            $PY_BIN/python3 -m venv ~/.venv
            source $HOME/.venv/bin/activate
            # build wheel for this python version. first install dependencies 
            # from tools/requirements.txt, then run sdist bdist_wheel. this is
            # because we mounted repo home to DOCKER_MNT.
            $PY_BIN/pip3 install -r $DOCKER_MNT/tools/requirements.txt
            # use absolute path for python3 and absolute dist path
            make dist PYTHON=$PY_BIN/python3 \
                DIST_FLAGS="--dist-dir $DOCKER_MNT/dist"
            # deactivate and remove virtual environment
            deactivate
            rm -rf ~/.venv
        else
            echo "no wheel will be built for `$PY_BIN/python3 --version`"
        fi
    done
}

# subroutine to repair wheels using auditwheel (installed on manylinux1 images)
# and then remove the original linux wheels
repair_cp3_wheels() {
    echo "repairing cpython3 wheels"
    # take first argument as name of distribution directory
    DIST_DIR=$1
    # get manylinux tag from DOCKER_IMAGE
    PLAT=`echo $DOCKER_IMAGE | sed s/"quay.io\/pypa\/"//g`
    # for each of the wheels in DIST_DIR, repair
    for PY_WHL in $DIST_DIR/*.whl
    do
        # if auditwheel doesn't return 0, then skip (non-platform wheel)
        if ! auditwheel show $PY_WHL
        then
            echo "skipping non-platform wheel $PY_WHL"
        # else repair wheel with auditwheel and remove original linux wheel
        else
            echo "repairing wheel $PY_WHL"
            auditwheel repair $PY_WHL --plat $PLAT -w $DIST_DIR
            rm -vf $PY_WHL
        fi
    done
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
# else run the wheel building and install process on manylinux1 docker image
else
    # cd to $DOCKER_MNT for convenience (can use relative paths, etc.)
    cd $DOCKER_MNT
    # build wheels using Makefile
    build_cp3_wheels
    # repair wheels using auditwheel in manylinux1 image
    repair_cp3_wheels $DOCKER_MNT/dist
    echo "wheel building and repair has been completed"
    # show contents of $DOCKER_MNT/dist
    ls $DOCKER_MNT/dist
fi
# Makefile for c_npy_demo build + install.

# package name
pkg_name       = c_npy_demo
# directory for libcheck test runner code
check_dir      = check
# c compiler, of course
CC             = gcc
# dependencies for the extension modules
ext_deps       = $(wildcard $(pkg_name)/*.c)
# dependencies for test running code. since we are testing helper functions in
# functimer.c, we include it as a dependency.
check_deps     = $(wildcard $(check_dir)/*.c) $(pkg_name)/functimer.c
# required Python source files in the package (modules and tests)
py_deps        = $(wildcard $(pkg_name)/*.py) $(wildcard $(pkg_name)/*/*.py)
# set python; on docker specify PYTHON value externally using absolute path
PYTHON        ?= python3
# flags to pass to setup.py build, sdist, bdist_wheel
BUILD_FLAGS    =
# directory to save distributions to; use absolute path on docker
DIST_FLAGS     =
# python compiler and linker flags for use when linking debug python into
# external C code (our test runner). gcc requires -fPIE.
PY_CFLAGS      = -fPIE $(shell $(PYTHON)d-config --cflags)
# ubuntu needs --embed, else -lpythonx.y is omitted by --ldflags, which is a
# linker error. libpython3.8 is in /usr/lib/x86_64-linux-gnu for me.
PY_LDFLAGS     = $(shell $(PYTHON)d-config --embed --ldflags)
# base installation directory of libcheck
CHECK_PATH     = /usr/local
# compile flags for compiling test runner. my libcheck is in /usr/local/lib.
# note we define C_NPY_DEMO_DEBUG macro so that the helper functions are
# non-static and can be accessed by the test runner.
check_cflags   = -DC_NPY_DEMO_DEBUG -I$(CHECK_PATH)/include $(PY_CFLAGS)
# linker flags for compiling test runner
check_ldflags  = -L$(CHECK_PATH)/lib -lcheck $(PY_LDFLAGS)
# flags to pass to the libcheck test runner
RUNNER_FLAGS   =

# phony targets
.PHONY: clean dummy dist

# triggered if no target is provided
dummy:
	@echo "Please specify a target to build."

# removes local build, dist, egg-info directories
clean:
	@rm -vrf build
	@rm -vrf $(pkg_name).egg-info
	@rm -vrf dist

# build extension module locally in ./build from source files with setup.py
# triggers when any of the files that are required are touched/modified.
build: $(py_deps) $(ext_deps)
	$(PYTHON) setup.py build $(BUILD_FLAGS)

# build in-place with build_ext --inplace. shared object for extension module
# will show up in the directory pkg_name.
inplace: $(ext_deps)
	$(PYTHON) setup.py build_ext --inplace

# build test runner and run unit tests using check. show flags passed to gcc
check: $(check_deps) inplace
	$(CC) $(check_cflags) -o runner $(check_deps) $(check_ldflags)
	./runner $(RUNNER_FLAGS)

# make source and wheel
dist: build
	$(PYTHON) setup.py sdist bdist_wheel $(DIST_FLAGS)

# make just wheels
bdist_wheel: build
	$(PYTHON) setup.py bdist_wheel $(DIST_FLAGS)

# make only source tar.gz (doesn't require full build)
sdist: $(py_deps) $(ext_deps)
	$(PYTHON) setup.py sdist $(DIST_FLAGS)

# perform root install by default (intended for use with venv)
install: install_root

# user install in site-packages directory for importing. builds if necessary.
# under ubuntu, you will need to manually remove zipped egg to uninstall.
install_user: build
	$(PYTHON) setup.py install --user

# install to root; don't use unless you have a venv set up!
install_root: build
	$(PYTHON) setup.py install
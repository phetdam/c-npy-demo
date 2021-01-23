# Makefile for c_npy_demo build + install.

# package name and source folder for extension source
PKG_NAME       = c_npy_demo
_EXT_DIR       = $(PKG_NAME)/cscale
# source folder for the timing module (also a C extension)
_TIMER_DIR     = $(PKG_NAME)/functimer
# directory for libcheck test runner code
CHECK_DIR      = check
# c compiler, of course
CC             = gcc
# dependencies for the extension modules
XDEPS          = $(wildcard $(_EXT_DIR)/*.c) $(wildcard $(_TIMER_DIR)/*.c)
# dependencies for test running code. since we are testing a helper function in
# $(_TIMER_DIR)/timeitresult.c, we include it as a dependency.
CHECK_DEPS     = $(wildcard $(CHECK_DIR)/*.c) $(_TIMER_DIR)/timeitresult.c
# required Python source files in the package (modules and tests)
PYDEPS         = $(wildcard $(PKG_NAME)/*.py) $(wildcard $(PKG_NAME)/*/*.py)
# set python; on docker specify PYTHON value externally using absolute path
PYTHON        ?= python3
# flags to pass to setup.py build
BUILD_FLAGS    =
# directory to save distributions to; use absolute path on docker
DIST_FLAGS    ?= --dist-dir ./dist
# python compiler and linker flags for use when linking python into external C
# code (our test runner); can be externally specified. gcc requires -fPIE.
PY_CFLAGS     ?= -fPIE $(shell python3-config --cflags)
# ubuntu needs --embed, else -lpythonx.y is omitted by --ldflags, which is a
# linker error. libpython3.8 is in /usr/lib/x86_64-linux-gnu for me.
PY_LDFLAGS    ?= $(shell python3-config --embed --ldflags)
# compile flags for compiling test runner. my libcheck is in /usr/local/lib
CHECK_CFLAGS   = $(PY_CFLAGS) -I$(_TIMER_DIR)
# linker flags for compiling test runner
CHECK_LDFLAGS  = $(PY_LDFLAGS) -L$(PKG_NAME) -lcheck
# flags to pass to the libcheck test runner
RUNNER_FLAGS   =

# phony targets (need to look into why build sometimes doesn't trigger)
.PHONY: build clean dummy dist

# triggered if no target is provided
dummy:
	@echo "Please specify a target to build."

# removes emacs autosave files and local build, dist, egg-info, test directories
clean:
	@rm -vf *~
	@rm -vrf build
	@rm -vrf $(PKG_NAME).egg-info
	@rm -vrf dist

# build extension module locally in ./build from source files with setup.py
# triggers when any of the files that are required are touched/modified.
build: $(PYDEPS) $(XDEPS)
	@$(PYTHON) setup.py build $(BUILD_FLAGS)

# build in-place with build_ext --inplace. shared object for extension module
# will show up in the directory PKG_NAME.
inplace: $(XDEPS)
	@$(PYTHON) setup.py build_ext --inplace

# build test runner and run unit tests using check. show flags passed to gcc
check: $(CHECK_DEPS) inplace
	$(CC) $(CHECK_CFLAGS) -o runner $(CHECK_DEPS) $(CHECK_LDFLAGS)
	@./runner $(RUNNER_FLAGS)

# make source and wheel
dist: build
	@$(PYTHON) setup.py sdist bdist_wheel $(DIST_FLAGS)

# perform root install by default (intended for use with venv)
install: install_root

# user install in site-packages directory for importing. builds if necessary.
# under ubuntu, you will need to manually remove zipped egg to uninstall.
install_user: build
	@$(PYTHON) setup.py install --user

# install to root; don't use unless you have a venv set up!
install_root: build
	@$(PYTHON) setup.py install
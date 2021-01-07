# Makefile for c_numpy_demo build + install.

# package name and source folder for extension source
PKG_NAME       = c_npy_demo
_EXT_DIR       = $(PKG_NAME)/cscale
# directory for libcheck test runner code
CHECK_DIR      = check
# c compiler, of course
CC             = gcc
# dependencies for the extension module
XDEPS          = $(wildcard $(_EXT_DIR)/*.c)
# dependencies for test running code
CHECK_DEPS     = $(wildcard $(CHECK_DIR)/*.c)
# required Python source files in the package (modules and tests)
PYDEPS         = $(wildcard $(PKG_NAME)/*.py) $(wildcard $(PKG_NAME)/*/*.py)
# set python; on docker specify PYTHON value externally using absolute path
PYTHON        ?= python3
# flags to pass to setup.py build
BUILD_FLAGS    =
# directory to save distributions to; use absolute path on docker
DIST_FLAGS    ?= --dist-dir ./dist
# python compiler and linker flags for use when linking python into external C
# code (our test runner); can be externally specified. note -fPIE needs to be
# passed because pythonx.y-config might pass a spec file to the -specs option
# that essentially passes -fno-PIE if -r, -fpie, -fPIE, -fpic, -fPIC not passed
# to gcc (i.e. no specification for some kind of position-independent code).
PY_CFLAGS     ?= -fPIE $(shell python3.8-config --cflags)
PY_LDFLAGS    ?= $(shell python3.8-config --ldflags)
# linker flags specifically for compiling the test runner
CHECK_LDFLAGS  = $(PY_LDFLAGS) -lcheck

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
build_inplace: $(XDEPS)
	@$(PYTHON) setup.py build_ext --inplace

# build test runner and run unit tests using check. no optimization needed. note
# -lcheck to link libcheck, which for me is installed at /usr/local/lib.
# need to pass -I/usr/include/python3.8 to correctly include Python.h and
# -lpython3.8 as linker args to link to libpython3.8, which for me is located
# in /usr/lib/x86_64-linux-gnu (system dir).
check: $(CHECK_DEPS)
	$(CC) $(PY_CFLAGS) -o $(CHECK_DIR)/runner $(CHECK_DEPS) $(CHECK_LDFLAGS)
	./$(CHECK_DIR)/runner

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
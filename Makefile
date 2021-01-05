# Makefile for c_numpy_demo build + install.

# test dir where for build binaries and .py files; used for local testing
TEST_DIR     = pkg_test
# package name, _ivlib.so source folder, source folder for extension source
PKG_NAME     = c_npy_demo
_IVLIB_DIR   = $(PKG_NAME)/_ivlib
_EXT_DIR     = $(PKG_NAME)/_np_bcast

CC           = gcc
# required C files for building our shared object.
CDEPS        = $(wildcard $(_IVLIB_DIR)/*.c)
# dependencies for the extension module
XDEPS        = $(wildcard $(_EXT_DIR)/*.c)
# required Python source files in the package (modules and tests)
PYDEPS       = $(wildcard $(PKG_NAME)/*.py) $(wildcard $(PKG_NAME)/*/*.py)
# specifically specify standard for gcc 4.8.2 on older linux distros
CFLAGS       = -o $(PKG_NAME)/_ivlib.so -shared -fPIC -fopenmp -lgomp -std=gnu11
# set python; on docker specify PYTHON value externally using absolute path
PYTHON      ?= python3
BUILD_FLAGS  = --build-lib $(TEST_DIR)
# directory to save distributions to; use absolute path on docker
DIST_FLAGS  ?= --dist-dir ./dist

# phony targets
.PHONY: build clean dummy dist

dummy:
	@echo "Please specify a target to build."

# removes emacs autosave files and local build, dist, egg-info, test directories
clean:
	@rm -vf *~
	@rm -vrf build
	@rm -vrf $(PKG_NAME).egg-info
	@rm -vrf dist
	@rm -vrf $(TEST_DIR)

# build external _ivlib.so from required C files (ff -> foreign function)
build_ff: $(CDEPS)
	@$(CC) $(CFLAGS) $(CDEPS)

# build np_touch module locally (in ./build) from source files with setup.py
# and build standalone shared object and move into pkg_test. currently
# configured to move built package into directory pkg_test. note that this
# target is more or less triggered to execute when any of the files that are
# required for the package are touched/modified (including data files).
build: build_ff $(PYDEPS) $(XDEPS) $(wildcard $(PKG_NAME)/data/*.csv)
	@$(PYTHON) setup.py build $(BUILD_FLAGS)

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
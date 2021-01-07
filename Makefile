# Makefile for c_numpy_demo build + install.

# package name and source folder for extension source
PKG_NAME     = c_npy_demo
_EXT_DIR     = $(PKG_NAME)/cscale
# c compiler, of course
CC           = gcc
# dependencies for the extension module
XDEPS        = $(wildcard $(_EXT_DIR)/*.c)
# required Python source files in the package (modules and tests)
PYDEPS       = $(wildcard $(PKG_NAME)/*.py) $(wildcard $(PKG_NAME)/*/*.py)
# set python; on docker specify PYTHON value externally using absolute path
PYTHON      ?= python3
# flags to pass to setup.py build
BUILD_FLAGS  =
# directory to save distributions to; use absolute path on docker
DIST_FLAGS  ?= --dist-dir ./dist

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
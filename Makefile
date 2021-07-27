# Makefile for numpy-api-bench build + install.

# Python package name, not the same as the project name
pkg_name = npapibench
# C compiler, of course
CC = gcc
# dependencies for the extension modules
ext_deps = $(wildcard $(pkg_name)/*.c)
# required Python source files in the package (modules, tests, setup.py)
py_deps = setup.py $(wildcard $(pkg_name)/*.py) \
	$(wildcard $(pkg_name)/tests/*.py)
# set python; on docker specify PYTHON value externally using absolute path
PYTHON ?= python3
# flags to pass to setup.py build, sdist, bdist_wheel
BUILD_FLAGS =
# flags to pass to setup.py dist
DIST_FLAGS =
# flags to pass when invoking pytest
PYTEST_ARGS ?= -rsxXP

# to force setup.py to rebuild, add clean as a target. note clean is phony.
ifeq ($(REBUILD), 1)
py_deps += clean
endif

# phony targets. note sdist just copies files.
.PHONY: check clean dummy sdist

# triggered if no target is provided
dummy:
	@echo "Please specify a target to build."

# removes local build, dist, egg-info directories, local shared objects
clean:
	@rm -vrf build
	@rm -vrf $(pkg_name).egg-info
	@rm -vrf dist
	@rm -vrf $(pkg_name)/*.so
	@rm -vrf $(pkg_name)/functimer/*.so

# build extension module locally in ./build from source files with setup.py
# triggers when any of the files that are required are touched/modified.
build: $(py_deps) $(ext_deps)
	$(PYTHON) setup.py build $(BUILD_FLAGS)

# build in-place with build_ext --inplace. shared object for extension module
# will show up in the package directory pkg_name.
inplace: build
	$(PYTHON) setup.py build_ext --inplace $(BUILD_FLAGS)

# just run pytest with arguments given by PYTEST_ARGS
check: inplace
	pytest $(PYTEST_ARGS)

# make source and wheel
dist: build
	$(PYTHON) setup.py sdist bdist_wheel $(DIST_FLAGS)

# make just wheels
bdist_wheel: build
	$(PYTHON) setup.py bdist_wheel $(DIST_FLAGS)

# make only source tar.gz (doesn't require full build)
sdist: $(py_deps) $(ext_deps)
	$(PYTHON) setup.py sdist $(DIST_FLAGS)
# Makefile for c_numpy_demo build + install.

# test dir where all our build binaries and .py files come together
TEST_DIR    = pkg_test
# package name, _ivlib.so source
PKG_NAME    = c_npy_demo
_IVLIB_DIR  = $(PKG_NAME)/_ivlib

CC          = gcc
# for building shared object. this is deprecated; we will use Extension later
# to build the shared object (although it will not be a Python module)
CDEPS       = $(addprefix $(_IVLIB_DIR)/,gauss.c euro_options.c root_find.c)
CFLAGS      = -o $(PKG_NAME)/_ivlib.so -shared -fPIC -fopenmp -lgomp
PYTHON      = python3
SETUP_FLAGS = --build-lib $(TEST_DIR)

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

# build np_touch module locally (in ./build) from source files with setup.py
# and build standalone shared object and move into pkg_test. force clean build.
# currently configured to move built package into directory pkg_test.
build: clean $(CDEPS)
	@$(CC) $(CFLAGS) $(CDEPS)
	@$(PYTHON) setup.py build $(SETUP_FLAGS)

# make source and wheel
dist: build
	@$(PYTHON) setup.py sdist bdist_wheel	

# perform root install by default (intended for use with venv)
install: install_root

# user install in site-packages directory for importing. builds if necessary.
# under ubuntu, you will need to manually remove zipped egg to uninstall.
install_user: build
	@$(PYTHON) setup.py install --user

# install to root; don't use unless you have a venv set up!
install_root: build
	@$(PYTHON) setup.py install
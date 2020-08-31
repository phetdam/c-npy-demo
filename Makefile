# Makefile for c_numpy_demo build + install.

CC          = gcc
CFLAGS      =
PYTHON      = python3
SETUP_FLAGS = #--compiler=mingw32

# phony targets
.PHONY: build clean dummy dist

dummy:
	@echo "Please specify a target to build."

# removes emacs autosave files and local build, dist, and egg-info directories
clean:
	@rm -vf *~
	@rm -vrf ./build
	@rm -vrf ./c_numpy_demo.egg-info
	@rm -vrf ./dist

# build np_touch module locally (in ./build) from source files, running setup.py
build: $(CDEPS)
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
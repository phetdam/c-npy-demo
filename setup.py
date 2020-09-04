# setup.py for building c_numpy_demo package

from setuptools import Extension, setup
from numpy import get_include

# package name and path for _ivlib and _ivmod source code
_PACKAGE_NAME = "c_numpy_demo"
_IVLIB_SRC_PATH = _PACKAGE_NAME + "/_ivlib"
_IVMOD_SRC_PATH = _PACKAGE_NAME + "/_ivmod"


def _get_ext_modules():
    """Returns a list of :class:`~setuptools.Extension` modules to build.
    
    .. note:: The extensions need not even be true modules, i.e. define a
       ``PyInit_*`` function. This allows building of shared objects that use
       the Python and NumPy C API without actually being loadable modules.
    
    :rtype: list
    """
    # use get_include to get numpy include directory
    _ivmod = Extension(name = "_ivmod",
                     sources = [_IVMOD_SRC_PATH + "/_modinit.c",
                                _IVMOD_SRC_PATH + "/np_demo.c",
                                _IVLIB_SRC_PATH + "/euro_options.c",
                                _IVLIB_SRC_PATH + "/gauss.c",
                                _IVLIB_SRC_PATH + "/root_find.c",
                                _IVMOD_SRC_PATH + "/np_euro_options.c"],
                     include_dirs = [get_include()])
    return [_ivmod]


def _setup():
    # get version
    with open("VERSION", "r") as vf: version = vf.read().rstrip()
    # short and long descriptions
    short_desc = ("A Python package demoing the combined use of ctypes, an "
                  "extension module, and the NumPy C API.")
    with open("README.rst", "r") as rf: long_desc = rf.read()
    # perform setup
    setup(name = _PACKAGE_NAME,
          version = version,
          description = short_desc,
          long_description = long_desc,
          long_description_content_type = "text/x-rst",
          author = "Derek Huang",
          author_email = "djh458@stern.nyu.edu",
          license = "MIT",
          url = "https://github.com/phetdam/c_numpy_demo",
          classifiers = ["License :: OSI Approved :: MIT License",
                         "Operating System :: OS Independent",
                         "Programming Language :: Python :: 3.6",
                         "Programming Language :: Python :: 3.7",
                         "Programming Language :: Python :: 3.8"],
          project_urls = {
              "Source": "https://github.com/phetdam/c_numpy_demo"
          },
          python_requires = ">=3.6",
          packages = ["c_numpy_demo", "c_numpy_demo.tests"],
          # adds implied vol shared object, 
          package_data = {
              _PACKAGE_NAME: ["_ivlib.so", "data/*.csv", "data/*.rst"]
          },
          install_requires = ["numpy>=1.15"],
          ext_package = _PACKAGE_NAME,
          ext_modules = _get_ext_modules()
    )

if __name__ == "__main__":
    _setup()

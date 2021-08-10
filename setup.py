"""setup.py for building numpy-api-bench package.

Extension modules require setup.py so we can't use PEP 517 setup.cfg.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from numpy import get_include
import platform
from setuptools import Extension, find_packages, setup

from npapibench import __package__, __version__

# package name (underscores converted to dashes anyways in PyPI) + short desc
_PACKAGE_NAME = "numpy-api-bench"
_SHORT_DESC = (
    "A small Python package showcasing speed differences between NumPy's "
    "Python and C APIs."
)
# include paths for functimer subpackage header files and for NumPy headers
_FUNCTIMER_INCLUDE_DIR = f"{__package__}/functimer/include"
_NUMPY_INCLUDE_DIR = get_include()

# extra extension compilation args. must specify C99+ for older Linux gccs.
if platform.system() == "Windows":
    _EXTRA_COMPILE_ARGS = ["/std:c11"]
else:
    _EXTRA_COMPILE_ARGS = ["-std=c11"]


def _get_ext_modules():
    """Returns a list of setuptools.Extension modules to build.
    
    .. note::

       The extensions must be true modules, i.e. define a ``PyInit_*``
       function. Use alternate means to build foreign C code.
    """
    # use get_include to get numpy include directory + add -std=gnu11 so that
    # the extension will build on older distros with old gcc like 4.8.2
    return [
        # C implementation of the stdscale function in pyimpl
        Extension(
            name="cimpl",
            sources=[f"{__package__}/cimpl.c"],
            include_dirs=[_NUMPY_INCLUDE_DIR],
            extra_compile_args=_EXTRA_COMPILE_ARGS
        ),
        # _timeapi module of the functimer subpackage
        Extension(
            name="functimer._timeapi",
            sources=[f"{__package__}/functimer/_timeapi.c"],
            include_dirs=[_FUNCTIMER_INCLUDE_DIR, _NUMPY_INCLUDE_DIR],
            extra_compile_args=_EXTRA_COMPILE_ARGS
        ),
        # _timeresult module of the functimer subpackage
        Extension(
            name="functimer._timeresult",
            sources=[f"{__package__}/functimer/_timeresult.c"],
            include_dirs=[_FUNCTIMER_INCLUDE_DIR, _NUMPY_INCLUDE_DIR],
            extra_compile_args=_EXTRA_COMPILE_ARGS
        ),
        # _timeunit module of the functimer subpackage
        Extension(
            name="functimer._timeunit",
            sources=[f"{__package__}/functimer/_timeunit.c"],
            include_dirs=[_FUNCTIMER_INCLUDE_DIR],
            extra_compile_args=_EXTRA_COMPILE_ARGS
        )
    ]


def _setup():
    # get long description from README.rst
    with open("README.rst", "r") as rf:
        long_desc = rf.read()
    # perform setup
    setup(
        name=_PACKAGE_NAME,
        version=__version__,
        description=_SHORT_DESC,
        long_description=long_desc,
        long_description_content_type="text/x-rst",
        author="Derek Huang",
        author_email="djh458@stern.nyu.edu",
        license="MIT",
        url="https://github.com/phetdam/numpy-api-bench",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9"
        ],
        project_urls={"Source": "https://github.com/phetdam/numpy-api-bench"},
        python_requires=">=3.6",
        packages=find_packages(),
        # benchmarking script
        entry_points={
            "console_scripts": [f"{__package__} = {__package__}.bench:main"]
        },
        install_requires=["numpy>=1.19"],
        extras_require={"tests": ["pytest>=6.0.1"]},
        ext_package=__package__,
        ext_modules=_get_ext_modules()
    )


if __name__ == "__main__":
    _setup()
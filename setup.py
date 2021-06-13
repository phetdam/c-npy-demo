# setup.py for building c_npy_demo package. since extension modules can't be
# built without setup.py, we can't use the new PEP 517 format.

from numpy import get_include
from setuptools import Extension, setup

from c_npy_demo import __version__

# package name
_PACKAGE_NAME = "c_npy_demo"
# extra compilation arguments for extension modules
_EXTRA_COMPILE_ARGS = ["-std=gnu11"]


def _get_ext_modules():
    """Returns a list of :class:`~setuptools.Extension` modules to build.
    
    .. note:: The extensions must be true modules, i.e. define a ``PyInit_*``
       function. Use alternate means to build foreign C code.
    
    :rtype: list
    """ 
    # use get_include to get numpy include directory + add -std=gnu11 so that
    # the extension will build on older distros with old gcc like 4.8.2
    return [
        Extension(
            name="cscale",
            sources=[_PACKAGE_NAME + "/cscale.c"],
            include_dirs=[get_include()],
            extra_compile_args=_EXTRA_COMPILE_ARGS
        ),
        Extension(
            name="functimer",
            sources=[_PACKAGE_NAME + "/functimer.c"],
            extra_compile_args=_EXTRA_COMPILE_ARGS
        )
    ]


def _setup():
    # short and long descriptions
    short_desc = (
        "A tiny Python package showcasing speed differences between NumPy's "
        "Python and C APIs."
    )
    with open("README.rst", "r") as rf:
        long_desc = rf.read()
    # perform setup
    setup(
        name=_PACKAGE_NAME,
        version=__version__,
        description=short_desc,
        long_description=long_desc,
        long_description_content_type="text/x-rst",
        author="Derek Huang",
        author_email="djh458@stern.nyu.edu",
        license="MIT",
        url="https://github.com/phetdam/c_npy_demo",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8"
        ],
        project_urls={
            "Source": "https://github.com/phetdam/c_npy_demo"
        },
        python_requires=">=3.6",
        packages=[_PACKAGE_NAME, _PACKAGE_NAME + ".tests"],
        # benchmarking script
        entry_points={
            "console_scripts": [
                _PACKAGE_NAME + ".bench = " + _PACKAGE_NAME + ".bench:main"
            ]
        },
        install_requires=["numpy>=1.19"],
        ext_package=_PACKAGE_NAME,
        ext_modules=_get_ext_modules()
    )


if __name__ == "__main__":
    _setup()
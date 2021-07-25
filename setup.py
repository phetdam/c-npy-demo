"""setup.py for building c-npy-demo package.

Extension modules require setup.py so we can't use PEP 517 format.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from numpy import get_include
import platform
from setuptools import Extension, setup

from c_npy_demo import __package__, __version__

# package name (underscores converted to dashes anyways in PyPI) + short desc
_PACKAGE_NAME = "c-npy-demo"
_SHORT_DESC = (
    "A tiny Python package showcasing speed differences between NumPy's "
    "Python and C APIs."
)
# extra compilation arguments for extension modules. C99+ required for gcc.
if platform.system() == "Linux":
    _EXTRA_COMPILE_ARGS = ["-std=gnu11"]
else:
    _EXTRA_COMPILE_ARGS = None


def _get_ext_modules():
    """Returns a list of setuptools.Extension modules to build.
    
    .. note::

       The extensions must be true modules, i.e. define a ``PyInit_*``
       function. Use alternate means to build foreign C code.
    """
    # use get_include to get numpy include directory + add -std=gnu11 so that
    # the extension will build on older distros with old gcc like 4.8.2
    return [
        Extension(
            name="cscale",
            sources=[f"{__package__}/cscale.c"],
            include_dirs=[get_include()],
            extra_compile_args=_EXTRA_COMPILE_ARGS
        ),
        Extension(
            name="functimer",
            sources=[f"{__package__}/functimer.c"],
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
        url="https://github.com/phetdam/c-npy-demo",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8"
        ],
        project_urls={
            "Source": "https://github.com/phetdam/c-npy-demo"
        },
        python_requires=">=3.6",
        packages=[__package__, f"{__package__}.tests"],
        # benchmarking script
        entry_points={
            "console_scripts": [
                f"{__package__}.bench = {__package__}.bench:main"
            ]
        },
        install_requires=["numpy>=1.19"],
        ext_package=__package__,
        ext_modules=_get_ext_modules()
    )


if __name__ == "__main__":
    _setup()
.. README for c_numpy_demo

c_numpy_demo
============

.. image:: https://img.shields.io/travis/phetdam/c_numpy_demo?logo=travis
   :target: https://travis-ci.org/github/phetdam/c_numpy_demo
   :alt: Travis (.org)

A demo Python package including a small extension module using the NumPy C API.
I personally struggled to figure out how to integrate extension modules written
in C, pure Python code, and the NumPy C API, so I hope this will be useful for
anyone interested in doing something similar. Applications of this Python
development paradigm could be for scientific computing or anything that needs to
be fast or do a lot of low-level stuff [#]_.

Includes a test suite [#]_ that also illustrates the difference in speed between
operating on NumPy array elements directly in C versus iterating through the
elements in Python and using ctypes__ to call the C function.

In the future, there should also be a small pytest__ test suite that can be run.

.. [#] Other options could be the use of ctypes, Cython__ or numba__.

.. [#] This does not exist yet.

.. __: https://docs.python.org/3/library/ctypes.html

.. __: https://docs.pytest.org/en/stable/contents.html

.. __: https://cython.readthedocs.io/en/latest/index.html

.. __: https://numba.readthedocs.io/en/stable/index.html

Installation
------------

To be added.

Building from this (unstable) repo will probably only work on Linux systems.
Local extension builds are done on WSL Ubuntu 18.04 with gcc 9.3 while builds on
Travis CI virtual machines are done on Ubuntu 18.04 with gcc 7.4. There is also
an implicit dependency on the gcc version being high enough such that an OpenMP
implementation is included [#]_. For example, I have ``libgomp.so.1.0.0`` in
``/usr/lib/x86_64-linux-gnu/``, with appropriate symbolic links.

.. [#] OpenMP may not be involved in this project. This is not final.
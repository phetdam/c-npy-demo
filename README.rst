.. README for c_numpy_demo

c_numpy_demo
============

.. image:: https://img.shields.io/travis/phetdam/c_numpy_demo?logo=travis
   :target: https://travis-ci.org/github/phetdam/c_numpy_demo
   :alt: Travis (.org)

A demo Python package illustrating how to combine a compiled Python C extension,
foreign compiled C code, ctypes__, and the NumPy and Python C APIs. I personally
struggled to figure out how to integrate these components together, through a
lot of trial error and reading of terrible documentation, so I hope this will be
useful for anyone interested in doing something similar. This development
paradigm could be applied to scientific computing or anything that needs to be
fast or do a lot of things close to the metal [#]_.

Includes a demo script [#]_ that performs a comparison of execution speed for
the iterative computation of Black and Bachelier implied volatility using
Halley's and Newton's methods. The script compares the time taken to solve the
Black and Bachelier implied volatilities of 10 million European option prices
for a pure Python implementation using ``scipy.optimize.minimize``, a mixed 
implementation where a minimalistic C implementation of Halley's/Newton's method
is used to solve for the price but iteration through prices is still done in
Python, and a ``ctypes`` wrapped pure C implementation. The script illustrates
the difference in speed between converting data types and looping directly in C
versus iterating through the prices in Python and using ``ctypes`` to call the C
function for each price [#]_.

There is a small test suite that can be run with pytest__ after installation.

.. [#] Other options could be the use of ctypes, Cython__ or numba__.

.. [#] This does not exist yet.

.. [#] There might also be a demonstration of how true parallelism can speed up
   such tasks, but the inclusion of OpenMP in this project is not finalized.

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

Contents
--------

TBA. Currently looks like some pure Python code, ``pytest`` test suite,
demo module to run benchmarks, separate C shared library for implied volatility
calculations, and a Python extension module written in C.

Lessons
-------

I learned a few lessons the hard way from doing this project. However, they are
valuable not just in the context of mixed Python/C development, as they can be
generalized to other mixed-language projects where higher and lower level
languages need to be used together.

1. Keep Python and C stuff as far away from each other as possible.
2. The GIL breathes much harder down your neck when you start using the Python
   C API, especially if you are trying to venture into multithreading with
   external libraries like OpenMP. See 1.
3. The NumPy C API is unforgiving. Be sure to check for ``NULL`` pointers.
4. If you aren't fluent in both languages, you will spend a lot of time
   frustrated.
5. Conversely, the satisfaction gained from building code that not only gains
   the speed of C but also exposes a pretty Python API may be more than enough
   to make up for all your struggles.

This project ended up being relatively large, and I will be porting a lot of the
code written here for one of my future planned projects, although I won't be
able to get started on that for a while.
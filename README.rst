.. README for c_numpy_demo

c_npy_demo
============

.. image:: https://img.shields.io/travis/com/phetdam/c_npy_demo?logo=travis
   :target: https://travis-ci.com/github/phetdam/c_npy_demo
   :alt: Travis (.com)

.. note::

   CI/build/deploy workflow is being migrated to GitHub actions after the
   changes made to `Travis CI's pricing plans`__.

A demo Python package illustrating how to combine a compiled Python C extension,
foreign compiled C code, ctypes__, and the NumPy and Python C APIs. I personally
struggled to figure out how to integrate these components together through a
lot of trial, error, and reading of dense and sometimes confusing documentation,
so I hope this will be useful for anyone interested in doing something similar.
This development paradigm could be applied to scientific computing or anything
that needs to be fast or do a lot of things close to the metal [#]_.

Includes a demo script [#]_ that performs a comparison of execution speed for
the iterative computation of Black and Bachelier implied volatility using
Halley's and Newton's methods. The script compares the time taken to solve the
Black and Bachelier implied volatilities of ~1 million European option prices
for a pure Python implementation using `scipy.optimize.newton`__, a mixed 
implementation where a minimalistic C implementation of Halley's/Newton's method
is used to solve for the price but iteration through prices is still done in
Python, and a ``ctypes`` wrapped pure C implementation. The script illustrates
the difference in speed between converting data types and looping directly in C
versus iterating through the prices in Python and using ``ctypes`` to call the C
function for each price [#]_.

There is a small test suite that can be run with pytest__ after installation.

.. [#] Other options could be the use of ctypes, Cython__ or numba__.

.. [#] This does not exist yet.

.. [#] Should include a demo of how multithreading helps with large inputs.

.. __: https://www.jeffgeerling.com/blog/2020/travis-cis-new-pricing-plan-threw-
   wrench-my-open-source-works

.. __: https://docs.python.org/3/library/ctypes.html

.. __: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.
   newton.html

.. __: https://docs.pytest.org/en/stable/contents.html

.. __: https://cython.readthedocs.io/en/latest/index.html

.. __: https://numba.readthedocs.io/en/stable/index.html

Installation
------------

From source
~~~~~~~~~~~

Building from this (unstable) repo will probably only work on Linux systems.
Local extension builds are done on WSL Ubuntu 18.04 with gcc 9.3 while builds on
Travis CI virtual machines are done on Ubuntu 18.04 with gcc 7.4. There is also
an implicit dependency on the gcc version being high enough such that an OpenMP
implementation is included [#]_. For example, I have ``libgomp.so.1.0.0`` in
``/usr/lib/x86_64-linux-gnu/``, with appropriate symbolic links.

.. [#] Fun exercise: Find where I have (sparingly) used OpenMP directives.

From PyPI
~~~~~~~~~

Although this package is not on PyPI (yet), I have successfully built
``manylinux1`` wheels using Travis CI on the ``manylinux1`` Docker images
provided by PyPA, of which more information can be found at the
`manylinux GitHub`__.

.. __: https://github.com/pypa/manylinux

Contents
--------

TBA. Currently looks like some pure Python code, ``pytest`` test suite,
demo module to run benchmarks, separate C shared library for implied volatility
calculations, and a Python extension module written in C.

Lessons
-------

Remarks on a few lessons I learned the hard way from mixing Python code,
foreign C code, the Python and NumPy C APIs, and Python C extension modules. It
was definitely a difficult but rewarding journey.

- Python C extension modules are loaded in a very specific way, are under the
  jurisdiction of the Python interpreter, and are subject to the GIL. When
  writing C code for an extension module, make sure you at least have **really**
  understood how to perform reference counting and exception handling from the
  `C API documentation`__. It is also useful to read about
  `initialization, finalizaton, and threads`__ if you are going to be using
  multiple threads. You should not need to read too much of the
  `memory management`__ documentation, as the Python interpreter should be
  managing the memory of the Python objects for you.
- The `NumPy C API documentation`__ [#]_ can be quite confusing sometimes, even
  when compared to the Python C API documentation. However, there are some rules
  of thumb that work pretty well. First, it is quite easy to use a function like
  `PyArray_FROM_OTF`_ to create a new NumPy array from an existing Python
  object. You just need to call `Py_DECREF`_ on the NumPy array if you are not
  returning it. If the Python object you called `PyArray_FROM_OTF`_ on was newly
  created in your function, then its reference also needs to be decremented,
  typically by calling `PyArray_XDECREF`_ before `Py_DECREF`_ on the
  ``PyObject *`` or ``PyArrayObject *`` pointing the NumPy array. If you want to
  create a NumPy array from scratch, however, **do not** first ``malloc`` some
  memory and then create a NumPy array around it using something like
  `PyArray_SimpleNewFromData`_! That memory is **not** tracked by the Python
  garbage collector and will result in a memory leak. Instead, use something
  like `PyArray_SimpleNew`_ to get an uninitialized NumPy array, use the
  `PyArray_DATA`_ macro to get a pointer to the first data element, and then
  fill in the data buffer, using stride information from `PyArray_STRIDES`_
  if necessary.
- Writing code in two different languages can be very frustrating, especially
  when it's something you are trying the first time. Be patient and don't rush;
  with languages such as C that are not as beginner-friendly as Python and with
  the added challenge of also having to understand Python design choices [#]_,
  there will likely be many missteps. I don't think I am particularly sharp or
  dull, but either way it was a humbling and trying experience.
- However, there is a lot of satisfaction to be gained from building code that
  not only gains the speed of C but also exposes a pretty Python API. The payoff
  was very work intensive, but for me it made up for all the struggle. I learned
  far more about CPython than I expected and have had my eyes opened to a new,
  and much more interesting, development paradigm.

.. __: https://docs.python.org/3/c-api/index.html

.. __: https://docs.python.org/3/c-api/init.html

.. __: https://docs.python.org/3/c-api/memory.html

.. __: https://numpy.org/doc/stable/reference/c-api/

.. _PyArray_FROM_OTF: https://numpy.org/doc/stable/reference/c-api/array.html#c.
   PyArray_FROM_OTF

.. _Py_DECREF: https://docs.python.org/3/c-api/refcounting.html#c.Py_DECREF

.. _PyArray_XDECREF: https://numpy.org/doc/stable/reference/c-api/array.html#c.
   PyArray_XDECREF

.. _PyArray_SimpleNewFromData: https://numpy.org/doc/stable/reference/c-api/
   array.html#c.PyArray_SimpleNewFromData

.. _PyArray_SimpleNew: https://numpy.org/doc/stable/reference/c-api/array.html#
   c.PyArray_SimpleNew

.. _PyArray_DATA: https://numpy.org/doc/stable/reference/c-api/array.html#c.
   PyArray_DATA

.. _PyArray_STRIDES: https://numpy.org/doc/stable/reference/c-api/array.html#c.
   PyArray_STRIDES

This project ended up being relatively large, and I will be porting a lot of the
code written here for one of my future planned projects, although I won't be
able to get started on that for a while.

.. [#] There is a newer documentation version for the dev version of NumPy 1.20,
   which may be found `here`__. To be fair, I did find `this page`__ in the
   NumPy documentation to be quite clear and helpful.

.. [#] Case in point: reference counting.

.. __: https://numpy.org/devdocs/reference/c-api/

.. __: https://numpy.org/doc/stable/user/c-info.how-to-extend.html
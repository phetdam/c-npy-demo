.. README for c_numpy_demo

c_npy_demo
==========

.. image:: https://img.shields.io/github/workflow/status/phetdam/c_npy_demo/
   build?logo=github
   :target: https://github.com/phetdam/c_npy_demo/actions
   :alt: GitHub Workflow Status

*We should forget about small efficiencies, say about 97% of the time: premature
optimization is the root of all evil* [#]_.

.. leave note as comment

.. The contents of this repository will see significant change in the near
   future, as I have decided to greatly simplify the code being used. The
   implied volatility stuff will be moved to a new repository, whose name will
   be yet another play on snake-related stuff. There is more code than I
   initially wanted, however, since I wrote my own alternative to `timeit`__
   as a C extension module along with its necessary unit tests since using
   ``timeit.main`` results in double allocation of a ``numpy`` array in the
   benchmarking script.

.. .. __: https://docs.python.org/3/library/timeit.html

A tiny demo Python package comparing speed differences between NumPy's Python
and C APIs that also serves as an example project for writing a C extension
module that uses the `NumPy C API`__.

.. [#] Attributed to Sir Tony Hoare, popularized by Donald Knuth.

.. __: https://numpy.org/devdocs/user/c-info.html


Installation
------------

From source
~~~~~~~~~~~

Building from source using this repo will probably only work on Linux systems.
Local extension builds are done on WSL Ubuntu 18.04 with gcc 9.3 while builds on
Github Actions runners were done within the `manylinux1 Docker images`__
provided by PyPA. To build, you will need ``numpy>=1.19`` and the latest
`setuptools`__ [#]_ installed on your system. Your C compiler should be a recent
version of gcc, but ``setuptools`` will sort out the details.

First, use ``git clone`` or download + unzip to get the repo source code and
install ``numpy>=1.19``. With the current working directory the repository root,
you can build the C extension modules and install directly with

.. code:: bash

   make inplace && pip3 install .

If you don't have or don't wish to use ``make``, you may instead use

.. code:: bash

   python3 setup.py build_ext --inplace && pip3 install .

.. [#] ``setuptools`` has seen a lot of change, especially post `PEP 517`__, but
   since a C extension modules have to be built in this package the legacy
   ``setup.py`` method of building distributions still has to be used. Note that
   the `distutils.core.Extension`__ class is present in ``setuptools`` as the
   ``setuptools.extension.Extension`` class.

.. __: https://github.com/pypa/manylinux

.. __: https://setuptools.readthedocs.io/en/latest/

.. __: https://www.python.org/dev/peps/pep-0517/

.. __: https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension

From PyPI
~~~~~~~~~

Although this package is not on PyPI (yet), I have successfully built
``manylinux1`` wheels using Github Actions on the ``manylinux1`` Docker images
provided by PyPA, of which more information can be found at the
`manylinux GitHub`__. Once I finish the source, I will try to build wheels for
Windows and Mac using GitHub Actions workflows.

.. __: https://github.com/pypa/manylinux

Contents
--------

The Python package contains three modules, two of which are C extension
modules. The first module is ``c_npy_demo.pyscale``, which is a pure Python
module containing one function that is two lines of ``numpy``\ -enabled Python
code. It is the "benchmark" for the C extension module ``c_npy_demo.cscale``
as both extension modules contain a single function that centers and scales to
unit variance a ``numpy.ndarray``. The last module is ``c_npy_demo.functimer``,
a purpose-written C extension module with a callable API for timing the
execution of a function with optional arguments in a `timeit`__\ -like
fashion [#]_.

On installation, ``setuptools`` will also create an entry point titled
``c_npy_demo.bench`` to access the benchmarking code. Just typing the name of
the entry point in the terminal should produce the ``timeit``\ -like output

.. code:: text

   numpy.ndarray shape (40, 5, 10, 10, 50, 5), size 5000000
   pyscale.stdscale -- 2 loops, best of 5: 157.1 msec per loop
   cscale.stdscale -- 5 loops, best of 5: 57.9 msec per loop

For usage details, try ``c_npy_demo.bench --help``.

.. __: https://docs.python.org/3/library/timeit.html

.. [#] Previously, I had used `timeit.main`__ for its pretty output, but
   unlike the callable API provided by ``timeit``, there is no way to pass in
   a global symbol table. Therefore, the ``numpy.ndarray`` allocated in the
   benchmarking script ``c_npy_demo.bench`` has to be allocated twice. I took
   the liberty of writing an alternative that provides ``timeit.main``\ -like
   capabilities with a callable API, intended for use with functions. The
   result was ``c_npy_demo.functimer``, written as a C extension module to
   reduce the timing measurement error resulting from timing ``n`` executions
   of a statement within a Python loop, which has a higher per-loop overhead
   than a C for loop.

.. __: https://docs.python.org/3/library/timeit.html#command-line-interface

Documentation
-------------

In progress, and will be eventually be hosted on `Read the Docs`__.

.. __: https://readthedocs.org/

Unit tests
----------

The unit test requirements for a C extension module are rather unique. Although
one is writing C code, the resulting shared object built by ``setuptools`` is
to be loaded by the Python interpreter, so it easier to conduct unit tests for
the Python-accessible functions by using Python unit testing tools. However, it
is possible that the extension module also contains some C functions that don't
use the Python C API and should be tested using a C unit testing framework.
It's also very possible that incorrectly written C code loaded as an extension
module may cause a segmentation fault and crash the interpreter. Ideally, unit
tests should be run in a separate address space so that the test runner doesn't
get killed by the operating system if a particular test causes a segfault.

For this project, I used `pytest`__ and `Check`__, embedding the Python
interpreter into and using Check unit tests inside a test runner to test both
from the Python interpreter and directly from C. Check runs unit tests in a
separate address space so the test runner doesn't get killed when a unit test
segfaults, but this can be disabled so that ``gdb`` can be used on the test
runner to debug C extension module behavior when its members are accessed by
the Python interpreter.

To build the test runner, you will need ``pytest`` and Check. ``pytest`` can be
easily installed with ``pip`` but Check is best built from source as the
versions available on some platforms are rather outdated. To build Check,
download the source from the `Check GitHub releases page`__ [#]_ and follow
the installation instructions in `the homepage`__ ``README.md`` [#]_. Then,
with the working directory the repository root, the test runner can be built
and run with

.. code:: bash

   make check

Type ``./runner --help`` for details on additional options that can be passed.

.. [#] `Check 0.15.2`__ was used in this project.

.. [#] I built ``libcheck`` using the standard ``./configure && make`` method
   with automake/autoconf.

.. __: https://pytest.readthedocs.io/

.. __: https://libcheck.github.io/check/

.. __: https://github.com/libcheck/check/releases

.. __: https://github.com/libcheck/check

.. __: https://github.com/libcheck/check/releases/tag/0.15.2

Lessons
-------

Remarks on a few lessons I learned the hard way from mixing Python code,
foreign C code, the Python and NumPy C APIs, and Python C extension modules. It
was definitely a difficult but rewarding journey.

TBA. I learned a great lesson on using ``tp_new`` and ``tp_dealloc`` by having
the unpleasant experience of having a double ``Py_DECREF`` lead to a
segmentation fault during ``pytest`` test discovery.

.. leave remarks on C/C++/Python mixing practices as comment

.. I personally went through a decent amount of pain, sweat, and tears to get
   this working, so I hope this will be useful example for one interested in
   doing something similar. However, I think it's generally best to decouple
   C/C++ and Python code as much as possible, so for example, if you to do
   computations in C/C++ code for speed increases, you should allocate memory
   in Python, pass pointers to your C/C++ code using `ctypes`__, and then have
   your C/C++ function write to the memory allocated by the Python interpreter.
   Since the `GIL`__ is released when calling foreign C/C++ code, you can
   then multithread using OpenMP, etc.

..   .. __: https://docs.python.org/3/library/ctypes.html

.. .. __: https://docs.python.org/3/glossary.html#term-global-interpreter-lock
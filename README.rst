.. README for numpy-api-bench

numpy-api-bench
===============

.. .. image:: https://img.shields.io/pypi/v/numpy-api-bench
   :target: https://pypi.org/project/numpy-api-bench/
   :alt: PyPI

.. .. image:: https://img.shields.io/pypi/wheel/numpy-api-bench
   :target: https://pypi.org/project/numpy-api-bench/
   :alt: PyPI - Wheel

.. .. image:: https://img.shields.io/pypi/pyversions/numpy-api-bench
   :target: https://pypi.org/project/numpy-api-bench/
   :alt: PyPI - Python Version

.. image:: https://img.shields.io/github/workflow/status/phetdam/
   numpy-api-bench/build?logo=github
   :target: https://github.com/phetdam/numpy-api-bench/actions
   :alt: GitHub Workflow Status

*We should forget about small efficiencies, say about 97% of the time:
premature optimization is the root of all evil* [#]_.

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

A small Python package comparing speed differences between NumPy's Python and
C APIs that also serves as an example project for writing a C extension
modules that use the `NumPy C API`__ [#]_.

.. [#] Attributed to Sir Tony Hoare, popularized by Donald Knuth.

.. __: https://numpy.org/devdocs/user/c-info.html

.. [#] This package is a fork of the latest version of `c-npy-demo`__
   with more concise and efficient code as well as a new name. ``c-npy-demo``
   was written at a time when I was still struggling with the Python and NumPy
   C APIs and had no idea how to test them, but now that I am much more
   practiced with both APIs, I felt it right to use my new knowledge to rework
   my old code.

   .. __: https://pypi.org/project/c-npy-demo/


Installation
------------

From source
~~~~~~~~~~~

Local extension builds are done on WSL Ubuntu 18.04 with gcc 9.3 while builds
on Github Actions runners were done within the `manylinux1 Docker images`__
provided by PyPA. To build, you will need ``numpy>=1.19`` and the latest
`setuptools`__ [#]_ installed on your system. Your C compiler should be
appropriate for your platform, ex. gcc for Linux, MSVC for Windows, but
``setuptools`` will (hopefully) sort out the details.

First, use ``git clone`` or download + unzip to get the repo source code and
install ``numpy>=1.19``. With the current working directory the repository
root, you can build the C extension modules and install directly with

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

.. `manylinux1`__ and Windows wheels may be installed directly from PyPI with

.. .. code:: bash

..    pip3 install numpy-api-bench

.. Currently wheels support only Python 3.6-3.8, but Python 3.9 wheels will be
   available soon.

.. .. __: https://github.com/pypa/manylinux

Not yet available.

Package contents
----------------

The ``numpy-api-bench`` package contains a pure Python module and two C
extension modules. The pure Python module is ``npapibench.pyscale``, which
contains one function that is two lines of ``numpy``\ -enabled Python code. It
is the "benchmark" for the C extension module ``npapibench.cscale`` as both
modules contain a single function that centers and scales to unit variance a
``numpy.ndarray``. The other C extension module is ``npapibench.functimer``,
which provides a callable API for timing the execution of a function with
optional arguments in a `timeit`__\ -like fashion [#]_.

On installation, ``setuptools`` will also create an entry point titled
``npapibench`` [#]_ to access the benchmarking code. Just typing the name
of the entry point in the terminal should produce the ``timeit``\ -like output

.. code:: text

   numpy.ndarray shape (40, 5, 10, 10, 50, 5), size 5000000
   pyscale.stdscale -- 2 loops, best of 5: 157.1 msec per loop
    cscale.stdscale -- 5 loops, best of 5: 57.9 msec per loop

For usage details, try ``npapibench --help``.

.. __: https://docs.python.org/3/library/timeit.html

.. [#] Previously, I had used `timeit.main`__ for its pretty output, but
   unlike the callable API provided by ``timeit``, one cannot pass in a global
   symbol table to avoid repeated setup. Therefore, the ``numpy.ndarray``
   allocated in the benchmarking code is allocated twice. I thus wrote
   ``npapibench.functimer``, which provides ``timeit.main``\ -like capabilities
   with a callable API intended for use with functions. It is written as a C
   extension module to reduce the timing measurement error resulting from
   timing ``n`` executions of a statement within a Python loop, which has a
   higher per-loop overhead than a C for loop.

.. __: https://docs.python.org/3/library/timeit.html#command-line-interface

.. [#] ``npapibench`` is the ``numpy-api-bench`` analogue to
   ``c_npy_demo.bench``, provided by ``c-npy-demo``.

.. Unit tests
.. ----------

.. The unit test requirements for a C extension module are rather unique. Although
   one is writing C code, the resulting shared object built by ``setuptools`` is
   to be loaded by the Python interpreter, so it easier to conduct unit tests for
   the Python-accessible functions by using Python unit testing tools. However, it
   is possible that the extension module also contains some C functions that don't
   use the Python C API and should be tested using a C unit testing framework.
   It's also very possible that incorrectly written C code loaded as an extension
   module may cause a segmentation fault and crash the interpreter. Ideally, unit
   tests should be run in a separate address space so that the test runner doesn't
   get killed by the operating system if a particular test causes a segfault.

.. For this project, I used `pytest`__ and `Check`__, embedding the Python
   interpreter into and using Check unit tests inside a test runner to test both
   from the Python interpreter and directly from C. Check runs unit tests in a
   separate address space so the test runner doesn't get killed when a unit test
   segfaults, but this can be disabled so that ``gdb`` can be used on the test
   runner to debug C extension module behavior when its members are accessed by
   the Python interpreter.

.. To build the test runner, you will need ``pytest`` and Check. ``pytest`` can be
   easily installed with ``pip`` but Check is best built from source as the
   versions available on some platforms are rather outdated. To build Check,
   download the source from the `Check GitHub releases page`__ [#]_ and follow
   the installation instructions in `the homepage`__ ``README.md`` [#]_. Then,
   with the working directory the repository root, the test runner can be built
   and run with

.. .. code:: bash

..    make check

.. Type ``./runner --help`` for details on additional options that can be passed.

.. .. [#] `Check 0.15.2`__ was used in this project.

.. .. [#] I built ``libcheck`` using the standard ``./configure && make`` method
      with automake/autoconf.

.. .. __: https://pytest.readthedocs.io/

.. .. __: https://libcheck.github.io/check/

.. .. __: https://github.com/libcheck/check/releases

.. .. __: https://github.com/libcheck/check

.. .. __: https://github.com/libcheck/check/releases/tag/0.15.2

.. Lessons
.. -------

.. Testing Python C extensions
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. Remarks on a few lessons I learned the hard way from mixing Python code,
   foreign C code, the Python and NumPy C APIs, and Python C extension modules. It
   was definitely a difficult but rewarding journey.

.. TBA, but I learned a great lesson on using ``tp_new`` and ``tp_dealloc`` by
   having the unpleasant experience of having a double ``Py_DECREF`` lead to a
   segmentation fault during ``pytest`` test discovery. This was caused by the
   fact that the `PyArg_ParseTupleAndKeywords`__ call in the ``tp_new`` function
   was parsing a `PyObject *`__. If parsing the ``PyObject *`` failed due to an
   earlier argument failing to parse correctly, the address in my C struct that
   the ``PyObject *`` was supposed to be written to will contain garbage. Then,
   the ``tp_dealloc`` function `Py_XDECREF`__\ 's the garbage pointer value at
   that address and boom, segmentation fault. The fix is to set the pointer value
   at the address in my C struct to ``NULL`` so on error, the ``Py_XDECREF`` has
   no effect since it will be passed ``NULL``.

.. .. __: https://docs.python.org/3/c-api/arg.html#c.PyArg_ParseTupleAndKeywords

.. .. __: https://docs.python.org/3/c-api/structures.html#c.PyObject

.. .. __: https://docs.python.org/3/c-api/refcounting.html#c.Py_XDECREF

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

.. Renaming projects
.. ~~~~~~~~~~~~~~~~~

.. big pain when it comes to changing names; changing releases, removing old
   version tags, deleting PyPI project... might have been better to simply make
   a new repository instead of renaming the old one. but too late rip
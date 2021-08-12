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

A small Python package comparing speed differences between NumPy's Python and
C APIs that also serves as an example project for writing C extension modules
that make use of the `NumPy C API`__ [#]_.

.. [#] Attributed to Sir Tony Hoare, popularized by Donald Knuth.

.. __: https://numpy.org/devdocs/user/c-info.html

.. [#] This package is a fork of the latest version of an old PyPI project of
   mine called ``c-npy-demo`` with more concise and efficient code as well as a
   new name. ``c-npy-demo`` was written at a time when I was still struggling
   with the Python and NumPy C APIs and had no idea how to test them, but now
   that I am much more practiced with both APIs, I felt it right to use my new
   knowledge to rework my old code.


Installation
------------

From source
~~~~~~~~~~~

Linux, Mac, and Windows binary wheels have been built from source on Github
Actions runners using the excellent `cibuildwheel`__ tool, which eases the
process of building binary wheels from compiled code for different platforms.
``cibuildwheel`` especially helps with building `manylinux`__ wheels.

.. __: https://cibuildwheel.readthedocs.io/en/stable/

.. __: https://github.com/pypa/manylinux

To build locally, you will need ``numpy>=1.19`` and the latest
`setuptools`__ [#]_ installed. Your C compiler should be appropriate for your
platform, ex. GCC for Linux, MSVC for Windows, but let ``setuptools`` do the
work.

.. __: https://setuptools.readthedocs.io/en/latest/

First, use ``git clone`` or download + unzip to get the repo source code and
install the requirements with [#]_

.. code:: bash

   pip3 install -r install_requires.txt

After you ``cd`` into the repository root, you can build the C extensions
in-place and install the package files with

.. code:: bash

   make inplace && pip3 install .

If you don't have or don't wish to use ``make``, you may instead use

.. code:: bash

   python3 setup.py build_ext --inplace && pip3 install .

.. [#] ``setuptools`` has seen a lot of change, especially post `PEP 517`__,
   but since C extension modules have to be built in this package the legacy
   ``setup.py`` method of building distributions still has to be used. Note
   that the `distutils.core.Extension`__ class is present in ``setuptools`` as
   the ``setuptools.extension.Extension`` class.

.. [#] Only Linux users need worry about using ``pip3``. Use ``pip`` for
   Windows and Mac.

.. __: https://www.python.org/dev/peps/pep-0517/

.. __: https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension

From PyPI
~~~~~~~~~

64-bit Python 3.6-3.9 binary wheels for Windows, MacOS, manylinux1, and
manylinux2010 can be installed from PyPI, with 32-bit wheels for Windows (x86)
and Linux (i686) also available. Install with

.. code:: bash

   pip3 install numpy-api-bench


Package contents
----------------

The ``numpy-api-bench`` package contains a pure Python module and two C
extension modules. The pure Python module is ``npapibench.pyimpl``, which
contains one function that is two lines of ``numpy``\ -enabled Python code. It
is the "benchmark" for the C extension module ``npapibench.cimpl`` as both
modules contain a single function that centers and scales to unit variance a
``numpy.ndarray``. The other C extension module is ``npapibench.functimer``,
which provides a callable API for timing the execution of a function with
optional arguments in a `timeit`__\ -like fashion [#]_.

On installation, ``setuptools`` will also create an entry point titled
``npapibench`` [#]_ to access the benchmarking code. Just typing the name
of the entry point in the terminal should produce the ``timeit``\ -like output

.. code:: text

   numpy.ndarray shape (40, 5, 10, 10, 20, 5), size 2000000
   pyimpl.stdscale -- 10 loops, best of 5: 31.9 msec per loop
    cimpl.stdscale -- 50 loops, best of 5: 13.6 msec per loop

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

Unit tests
----------

Testing internal functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

The unit testing requirements for a C extension module are rather unique.
Although one is writing C code, the resulting shared object built by
``setuptools`` is loaded by the Python interpreter, so it easier to test
Python-accessible functions by using Python unit testing tools. However, it is
likely that the C extension module, which `by convention`__ is a single file
with all members static except the module initialization function, may contain
some internal functions that cannot be accessed directly from Python. So far,
there does not seem to be a widely accepted approach to unit testing code in
Python C extensions, especially these internal C functions.

.. __: https://docs.python.org/3/extending/extending.html#
   providing-a-c-api-for-an-extension-module

For this project, in separate C extension modules, I wrote Python wrappers for
the internal functions I wanted to test, providing a C API for other extension
modules by using the header file and ``PyCapsule`` method described in the
`official tutorial`__ on writing Python C extensions. Then, I wrote unit tests
in Python using the `pytest`__ API and simply invoked ``pytest`` to collect and
run all unit tests, as it produces far better unit test output compared to
most C unit testing frameworks and is aware of Python objects. If there were
any segmentation faults or need to more closely debug, I would just then invoke
``gdb`` on the Python interpreter running ``pytest`` [#]_ with

.. code:: bash

   gdb --args python3 -m pytest

Together, ``pytest`` and ``gdb`` allowed me to hammer out a significant number
of bugs.

.. [#] The ``pytest`` entry point is a Python script run by the interpreter so
   ``gdb pytest`` does not work.

.. __: https://docs.python.org/3/extending/extending.html#
   providing-a-c-api-for-an-extension-module

.. __: https://docs.pytest.org/en/stable/

For users
~~~~~~~~~

To run the unit tests in the package, ``pytest>=6.0.1`` must be installed. If
installing the wheel from PyPI, you can install ``pytest`` as an optional
dependency alongside the package code with

.. code:: bash

   pip3 install numpy-api-bench[tests]

The unit tests are located in ``npapibench.tests`` and
``npapibench.functimer.tests`` and can be run with

.. code:: bash

   pytest --pyargs npapibench.tests && pytest --pyargs npapibench.functimer.tests

Other desired flags can be passed to ``pytest`` before the ``--pyargs`` flag.

If building from source, follow the steps in `From source`_ but replace the
final ``pip3 install .`` with ``pip3 install .[tests]``. The unit tests can
be run after ``cd``\ ing to the repository root by simply calling ``pytest``.

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
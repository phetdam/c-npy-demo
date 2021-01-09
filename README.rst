.. README for c_numpy_demo

c_npy_demo
==========

.. image:: https://img.shields.io/github/workflow/status/phetdam/c_npy_demo/
   build?logo=github
   :target: https://github.com/phetdam/c_npy_demo/actions
   :alt: GitHub Workflow Status

*We should forget about small efficiencies, say about 97% of the time: premature
optimization is the root of all evil* [#]_

   Note:

   The contents of this repository will see significant change in the near
   future, as I have decided to greatly simplify the code being used. The
   implied volatility stuff will be moved to a new repository, whose name will
   be yet another play on snake-related stuff.

A tiny demo Python package comparing speed differences between NumPy's Python
and C APIs that also serves as an example project for writing a C extension
module that uses the `NumPy C API`__. I personally went through a decent amount
of trial, error, and reading of dense and sometimes confusing documentation, so
I hope this will be useful example for one interested in doing something
similar. In my opinion, however, it's generally best to decouple C and Python
code as much as possible, so if you want to use C to speed up your computations
you are best off allocating memory in Python, passing that to your foreign C
code using `ctypes`__, and then having your C function write to the memory
allocated by the Python interpreter. Since the `GIL`__ is released when calling
foreign C code, you can then multithread using OpenMP, etc.

.. [#] Attributed to Sir Tony Hoare, popularized by Donald Knuth.

.. __: https://www.jeffgeerling.com/blog/2020/travis-cis-new-pricing-plan-threw-
   wrench-my-open-source-works

.. __: https://numpy.org/devdocs/user/c-info.html

.. __: https://docs.python.org/3/library/ctypes.html

.. __: https://docs.python.org/3/glossary.html#term-global-interpreter-lock

Installation
------------

From source
~~~~~~~~~~~

Building from this (unstable) repo will probably only work on Linux systems.
Local extension builds are done on WSL Ubuntu 18.04 with gcc 9.3 while builds on
GIthub Actions runners were done within the `manylinux1 Docker images`__
provided by PyPA. To build, you will need to have the latest `setuptools`__ [#]_
installed on your system, with `wheel`__ also installed if you like to create a
prebuilt wheel for your own specific platform.

.. [#] ``setuptools`` has seen a lot of change, especially post `PEP 517`__, but
   since a (tiny) C extension module has to be built in this package the legacy
   ``setup.py`` method of building distributions still has to be used.

.. __: https://github.com/pypa/manylinux

.. __: https://setuptools.readthedocs.io/en/latest/

.. __: https://wheel.readthedocs.io/en/stable/

.. __: https://www.python.org/dev/peps/pep-0517/

From PyPI
~~~~~~~~~

Although this package is not on PyPI (yet), I have successfully built
``manylinux1`` wheels using Github Actions on the ``manylinux1`` Docker images
provided by PyPA, of which more information can be found at the
`manylinux GitHub`__.

.. __: https://github.com/pypa/manylinux

Contents
--------

TBA.

Documentation
-------------

In progress, and will be eventually be hosted on `Read the Docs`__. For now,
the ``doc`` directory probably only has ``conf.py`` and ``index.rst``.

.. __: https://readthedocs.org/

Unit tests
----------

TBA.

Lessons
-------

Remarks on a few lessons I learned the hard way from mixing Python code,
foreign C code, the Python and NumPy C APIs, and Python C extension modules. It
was definitely a difficult but rewarding journey.

TBA.
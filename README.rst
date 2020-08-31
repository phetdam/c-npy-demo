.. README for c_numpy_demo

c_numpy_demo
============

A demo Python package including a toy extension module using the NumPy C API. I
personally struggled for a bit on how to integrate extension modules written in
C, pure Python code, and the NumPy C API, so I hope this will be useful for
anyone interested in doing something similar. Applications of this Python
development paradigm could be for scientific computing or anything that needs to
be fast or do a lot of low-level stuff [#]_.

Includes a test suite that also illustrates the difference in speed between
operating on NumPy array elements directly in C versus iterating through the
elements in Python and using ctypes__ to call the C function.

In the future, there should also be a small pytest__ test suite that can be run.

.. [#] Other options could be the use of ctypes, Cython__ or numba__.

.. __: https://docs.python.org/3/library/ctypes.html

.. __: https://docs.pytest.org/en/stable/contents.html

.. __: https://cython.readthedocs.io/en/latest/index.html

.. __: https://numba.readthedocs.io/en/stable/index.html
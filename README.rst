.. README for c_numpy_demo

c_numpy_demo
============

A demo Python package including a toy extension module using the NumPy C API. I
personally struggled for a bit on how to integrate extension modules written in
C, pure Python code, and the NumPy C API, so I hope this will be useful for
anyone interested in doing something similar. Applications of this Python
development paradigm could be for scientific computing or anything that needs to
be fast or do a lot of low-level stuff [#]_.

.. [#] Other options could be the use of Cython__ or numba__.

.. __: https://cython.readthedocs.io/en/latest/index.html

.. __: https://numba.readthedocs.io/en/stable/index.html
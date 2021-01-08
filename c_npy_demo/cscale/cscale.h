/**
 * @file cscale.h
 * @brief Contains declaration for function in `_cscale.c` and useful macros.
 * @note This should be included after definition of `PY_ARRAY_UNIQUE_SYMBOL`
 *     and `numpy/arrayobject.h` in a source file. Has header guards.
 */

#ifndef CSCALE_H
#define CSCALE_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#endif
// don't include deprecated numpy C API. avoid re-defining in same file
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
// no re-definition of PY_ARRAY_UNIQUE_SYMBOL and re-importing of
// numpy/arrayobject.h in file if PY_ARRAY_UNIQUE_SYMBOL already defined in file
#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL CSCALE_ARRAY_API
// arrayobject.h gives access to the array API
#include "numpy/arrayobject.h"
#endif

// prints warning if NpyIter_Deallocate fails on PyArrayObject ar's NpyIter iter
#define NpyIter_DeallocAndWarn(iter, ar) if (NpyIter_Deallocate(iter) == \
  NPY_FAIL) { PyErr_WarnEx(PyExc_RuntimeWarning, "unable to deallocate " \
  "iterator of " #ar, 1); }

PyObject *stdscale(PyObject *self, PyObject *args, PyObject *kwargs);

#endif /* CSCALE_H */
/**
 * @file cscale.c
 * @brief Core function to broadcast arbitrary Python inputs into 1D ndarray.
 */

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif

#include "Python.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL CSCALE_ARRAY_API
#include "numpy/arrayobject.h"

#include "cscale.h"

/**
 * Centers and scale a `numpy.ndarray` to zero mean, unit variance.
 * 
 * @param args Positional arguments
 * @param kwargs Keyword arguments
 * @returns `PyArrayObject *` cast to `PyObject *` 
 */
PyObject *stdscale(PyObject *self, PyObject *args, PyObject *kwargs) {
  // numpy ndarray and delta degrees of freedom
  PyArrayObject *ar;
  int ddof = 0;
  // argument names
  char *argnames[] = {"ar", "ddof", NULL};
  // check args and kwargs. | indicates that all args after it are optional
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", argnames, &ar, &ddof)) {
    PyErr_SetString(
      PyExc_ValueError, "argument parsing failure. check that the positional "
      "arg ar has been passed to the function and that ddof is an int"
    );
    return NULL;
  }
  // check that ar is a numpy ndarray
  if (!PyArray_Check(ar)) {
    PyErr_SetString(PyExc_TypeError, "ar must be of type numpy.ndarray");
    return NULL;
  }
  // check that ar is of the correct types
  if (!PyArray_ISINTEGER(ar) && !PyArray_ISFLOAT(ar)) {
    PyErr_SetString(PyExc_TypeError, "ar must have dtype int or float");
    return NULL;
  }
  // check that ddof is nonnegative
  if (ddof < 0) {
    PyErr_SetString(PyExc_ValueError, "ddof must be a nonnegative int");
    return NULL;
  }
  // new output array, with same values. always aligned and in C major order.
  // unlike original array, will have dtype float64
  PyArrayObject *out_ar = (PyArrayObject *) PyArray_FromArray(
    ar, PyArray_DescrFromType(NPY_FLOAT64), NPY_ARRAY_CARRAY
  );
  // return centered and scaled out_ar
  return (PyObject *) out_ar;
}
/** 
 * @file np_demo.c
 * @brief Contains functions demonstrating the NumPy C API.
 */

#include <stdio.h>
#include <stdlib.h>

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif

#include <math.h>
#include "Python.h"

/** 
 * must define NO_IMPORT_ARRAY if NumPy API is used but import_array is not 
 * called in a C file. import_array only called by module init function.
 * */
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL _IVMOD_ARRAY_API
#include "numpy/arrayobject.h"

#include "np_demo.h"

/**
 * Returns the type of the Python object as a (Python) string.
 *
 * @param self
 * @param args Python tuple of args, should only have one element (obj)
 * @return None if obj is NULL or a Python Unicode string (both PyObject *)
 */ 
PyObject *PyObject_type(PyObject *self, PyObject *args) {
  PyObject *obj;
  // parse arguments with PyArg_ParseTuple, positional args only
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    /*
    PyErr_SetString(PyExc_RuntimeError, "PyObject_type: expected only one arg" \
		    " of type PyObject *");
    */
    Py_INCREF(Py_None);
    return Py_None;
  }
  /* if NULL, return None */
  if (obj == NULL) {
    fprintf(stderr, "PyObject_type: obj is NULL\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  /* get object type name*/
  const char *type_name;
  type_name = Py_TYPE(obj)->tp_name;
  /* get new reference to Python string from C string and return */
  PyObject *py_type_name;
  py_type_name = PyUnicode_FromString(type_name);
  return py_type_name;
}

/**
 * Returns the sum of the elements in an (int, float) ndarray.
 *
 * @param self
 * @param args PyObject * args tuple, arg type numpy.ndarray
 * @return Sum of the elements in an ndarray, as Python float
 */
PyObject *PyArrayObject_sum(PyObject * self, PyObject *args) {
  PyArrayObject *ar;
  PyObject *obj;
  // check args
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    PyErr_SetString(PyExc_TypeError, "expected exactly one numpy.ndarray");
    Py_INCREF(Py_None);
    return Py_None;
  }
  ar = (PyArrayObject *) obj;
  // error checking for incref
  if (ar == NULL) {
    fprintf(stderr, "PyArrayObject_sum: ar is NULL\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  // check type
  if (!PyArray_ISINTEGER(ar) && !PyArray_ISFLOAT(ar)) {
    PyErr_SetString(PyExc_TypeError, "numpy.ndarray must contain int or float" \
		    " members");
    Py_INCREF(Py_None);
    return Py_None;
  }
  // compute and return the sum of the elements across all axes
  PyObject *ar_sum;
  ar_sum = PyArray_Sum(ar, NPY_MAXDIMS, PyArray_TYPE(ar), NULL);
  return ar_sum;
}

/**
 * (Reusing old name) Return new numpy array from object. allow only list/tuple.
 */
PyObject *loop(PyObject *self, PyObject *args) {
  PyObject *obj;
  // only parse a single integer object
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    PyErr_SetString(PyExc_ValueError, "expected only one positional arg");
    Py_INCREF(Py_None);
    return Py_None;
  }
  /*
  // check for list/tuple type
  if (!PyList_Check(obj) && !PyTuple_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "arg must be list- or tuple-like");
    Py_INCREF(Py_None);
    return Py_None;
  }
  */
  // convert into NumPy array and return
  PyArrayObject *ar;
  PyArray_Converter(obj, (PyObject **) &ar);
  PyArray_XDECREF(ar);
  return PyFloat_FromDouble(NAN);
}
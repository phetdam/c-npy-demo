/**
 * @file np_broadcast.c
 * @brief Core function to broadcast arbitrary Python inputs into 1D ndarray.
 */

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif

#include "Python.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL _IVMOD_ARRAY_API
#include "numpy/arrayobject.h"

#include "np_broadcast.h"

/**
 * Broadcasts Python objects into float64 1D ndarrays, if possible.
 * 
 * Method internal to the extension module.
 * 
 * Arguments should have already been checked by PyArg_ParseTuple or an
 * equivalent function before calling this function. args is borrowed ref.
 * Resulting arrays are guaranteed aligned and C-contiguous.
 * 
 * @note It appears that functions not directly called from the Python
 * interpreter are unable to set the current thread's error indicator. This
 * function is meant to be internal to the module anyways, so that's not too
 * much of a problem.
 * 
 * @param args Pointer to array of PyObject *
 * @param nargs Number of PyObject * in the array
 * @param axis 0 for row vectors, 1 for column vectors
 * @returns Pointer to array of numpy ndarrays, if successful
 */
PyObject **np_float64_bcast_1d(PyObject **args, Py_ssize_t nargs,
  Py_ssize_t axis) {
  // check nargs and axis
  if (nargs <= 0) {
    PyErr_SetString(PyExc_ValueError, "nargs must be positive int");
    return NULL;
  }
  // check value of axis; must be 0 or 1
  if ((axis != NP_BROADCAST_ROWS) && (axis != NP_BROADCAST_COLS)) {
    PyErr_SetString(PyExc_ValueError, "axis must be 0 or 1");
    return NULL;
  }
  /**
   * if all objects are numeric, just return without making any changes.
   */
  Py_ssize_t n_scalar;
  n_scalar = 0;
  long i;
  for (i = 0; i < nargs; i++) {
    if(PyObject_is_numeric(args[i])) {
      n_scalar = n_scalar + 1;
    }
  }
  if (n_scalar == nargs) {
    return args;
  }
  /**
   * [try to] convert all data elements to numpy arrays of type numpy.float64.
   * use the NPY_ARRAY_IN_ARRAY flag to guarantee alignment and C contiguous
   * layout. scalars are first wrapped in a tuple to get a ndarray shape (1,).
   * each new ndarray has its shape checked; must be one-dimensional.
   */
  for (long _i = 0; _i < nargs; _i++) {
    // if args[_i] is scalar, wrap in tuple.
    if (PyObject_is_numeric(args[_i])) {
      PyObject *tup;
      tup = PyTuple_New(1);
      PyTuple_SetItem(tup, 0, args[_i]);
      // tuple has ownership of old scalar (stole reference)
      args[_i] = tup;
    }
    // convert args[_i] to ndarray and check for conversion error
    args[_i] = (PyObject *) PyArray_FROM_OTF(args[_i], NPY_FLOAT64,
      NPY_ARRAY_IN_ARRAY);
    // don't set exception; let numpy function set the error indicator
    if (args[_i] == NULL) {
      return NULL;
    }
    // check that number of dimensions is 1
    if (PyArray_NDIM(args[_i]) != 1) {
      PyErr_Format(PyExc_ValueError, "Positional arg %d cannot be converted"
        " to 1D numpy.ndarray\n", _i + 1);
      PyArray_XDECREF(args[_i]);
      return NULL;
    }
  }
  /**
   * manually broadcast. iterate to make sure any objects that aren't length 1
   * have an equal length; if not, return on error. else, extend 1D arrays
   * by creating broadcasted tuples and then ndarrays.
   */
  npy_intp shared_len, local_len;
  shared_len = -1;
  // shared_len will be the max of the lengths
  for (i = 0; i < nargs; i++) {
    local_len = PyArray_SHAPE((PyArrayObject *) args[i])[0];
    shared_len = (local_len > shared_len) ? local_len : shared_len;
  }
  // check that shapes are broadcastable. broadcast length 1 ndarrays.
  for (long _i = 0; _i < nargs; _i++) {
    local_len = PyArray_SHAPE(args[_i])[0];
    if (local_len == 1) {
      // get base value as float, XDECREF original ndarray, create tuple
      double base;
      base = *((double *) PyArray_DATA(args[_i]));
      PyArray_XDECREF((PyArrayObject *) args[_i]);
      args[_i] = PyTuple_New((Py_ssize_t) shared_len);
      for (Py_ssize_t j = 0; j < shared_len; j++) {
        PyTuple_SetItem(args[_i], j, PyFloat_FromDouble(base));
      }
      // convert to ndarray with C order and aligned memory + check if error
      args[_i] = (PyObject *) PyArray_FROM_OTF(args[_i], NPY_FLOAT64,
        NPY_ARRAY_IN_ARRAY);
      // don't set exception; let numpy function set the error indicator
      if (args[_i] == NULL) {
        Py_DECREF(args[_i]);
        return NULL;
      }
    }
    else if (local_len == shared_len) { ; }
    else {
      PyErr_Format(PyExc_ValueError, "Could not broadcast (%ld,) to (%ld,)",
        (long) local_len, (long) shared_len);
      PyArray_XDECREF((PyArrayObject *) args[_i]);
      return NULL;
    }
  }
  // if reshape into column is specified, reshape flat arrays into columns
  if (axis == 1) {
    // new shape; need to pass in PyArray_Dims struct
    npy_intp new_shape[2];
    new_shape[0] = shared_len;
    new_shape[1] = 1;
    PyArray_Dims dims;
    dims.ptr = new_shape;
    dims.len = 2;
    for (i = 0; i < nargs; i++) {
      args[i] = PyArray_Newshape(args[i], &dims, NPY_CORDER);
      if (args[i] == NULL) {
        PyErr_Format(PyExc_RuntimeError, "Failed to reshape arg %d\n", i + 1);
        return NULL;
      }
    }
  }
  // if conversion was successful, return
  return args;
}

/**
 * Broadcasts Python objects into float64 1D ndarrays, if possible.
 * 
 * This is the Python user facing method. Note that np_float64_bcast_1d also
 * raise Python exceptions, so when it errors we return NULL and don't raise.
 *
 * Resulting arrays are guaranteed aligned and C-contiguous. If the input
 * iterable is empty, then it is simply returned.
 * 
 * @param self
 * @param args A list, tuple, or dict, and an integer in {0, 1}
 * @returns The original iterable, with members now ndarray
 */
PyObject *np_float64_bcast_1d_ext(PyObject *self, PyObject *args) {
  // expect two arguments; one is tuple/list/dict the other is int
  PyObject *obj;
  Py_ssize_t axis;
  if (!PyArg_ParseTuple(args, "On", &obj, &axis)) {
    PyErr_SetString(PyExc_ValueError, "First argument must be a list/tuple/"
      "dict while the second argument must be an integer");
    return NULL;
  }
  // check value of axis; must be 0 or 1
  if ((axis != NP_BROADCAST_ROWS) && (axis != NP_BROADCAST_COLS)) {
    PyErr_SetString(PyExc_ValueError, "axis must be 0 or 1");
    return NULL;
  }
  /**
   * unpack the objects in the iterable. for simplicity, only allow list, tuple,
   * and dict (bare minimum needed). the separate keys PyObject ** is for the
   * dictionary, since that's how we access it.
   */
  PyObject **objs, **keys;
  keys = NULL;
  Py_ssize_t n_objs, i;
  if (PyTuple_Check(obj)) {
    n_objs = PyTuple_Size(obj);
    /**
     * if empty, just return, but also increase reference. this is because it is
     * possible there is no other reference to obj outside the function (ex. no
     * name binding to variable) so it will get collected upon return.
     */
    if (n_objs == 0) {
      Py_INCREF(obj);
      return obj;
    }
    objs = (PyObject **) malloc(n_objs * sizeof(PyObject *));
    if (objs == NULL) {
      return PyErr_NoMemory();
    }
    for (i = 0; i < n_objs; i++) {
      objs[i] = PyTuple_GetItem(obj, i);
    }
    // perform conversions using np_float64_bcast_1d
    PyObject **new_objs;
    new_objs = np_float64_bcast_1d(objs, n_objs, axis);
    // on error, free memory, and return (np_float64_bcast_1d raised exception)
    if (new_objs == NULL) {
      free(objs);
      return NULL;
    }
    // make a new tuple
    obj = PyTuple_New(n_objs);
    for (i = 0; i < n_objs; i++) {
      PyTuple_SetItem(obj, i, objs[i]);
    }
  }
  else if (PyDict_Check(obj)) {
    // get the keys and values as lists first; this lets us get n_objs
    PyObject *_keys, *_vals;
    _keys = PyDict_Keys(obj);
    _vals = PyDict_Values(obj);
    n_objs = PyList_Size(_keys);
    // if empty, just decref and return (incref obj)
    if (n_objs == 0) {
      Py_DECREF(_keys);
      Py_DECREF(_vals);
      Py_INCREF(obj);
      return obj;
    }
    // malloc and copy over the keys and values.
    objs = (PyObject **) malloc(n_objs * sizeof(PyObject *));
    if (objs == NULL) {
      return PyErr_NoMemory();
    }
    keys = (PyObject **) malloc(n_objs * sizeof(PyObject *));
    if (keys == NULL) {
      return PyErr_NoMemory();
    }
    for (i = 0; i < n_objs; i++) {
      keys[i] = PyList_GetItem(_keys, i);
      objs[i] = PyList_GetItem(_vals, i);
    }
    // perform conversions using np_float64_bcast_1d
    PyObject **new_objs;
    new_objs = np_float64_bcast_1d(objs, n_objs, axis);
    // on error, free memory, and return (np_float64_bcast_1d raised exception)
    if (new_objs == NULL) {
      free(objs);
      free(keys);
      return NULL;
    }
    // make a new dict
    obj = PyDict_New();
    for (i = 0; i < n_objs; i++) {
      PyDict_SetItem(obj, keys[i], objs[i]);
    }
    // Py_DECREF both _keys and _vals (they are new references)
    Py_DECREF(_keys);
    Py_DECREF(_vals);
    // free memory used for PyObject * keys
    free(keys);
  }
  else if (PyList_Check(obj)) {
    n_objs = PyList_Size(obj);
    // if empty, just return (incref obj)
    if (n_objs == 0) {
      Py_INCREF(obj);
      return obj;
    }
    objs = (PyObject **) malloc(n_objs * sizeof(PyObject *));
    if (objs == NULL) {
      return PyErr_NoMemory();
    }
    for (i = 0; i < n_objs; i++) {
      objs[i] = PyList_GetItem(obj, i);
    }
    // perform conversions using np_float64_bcast_1d
    PyObject **new_objs;
    new_objs = np_float64_bcast_1d(objs, n_objs, axis);
    // on error, free memory, and return (np_float64_bcast_1d raised exception)
    if (new_objs == NULL) {
      free(objs);
      return NULL;
    }
    // make a new list
    obj = PyList_New(n_objs);
    for (i = 0; i < n_objs; i++) {
      PyList_SetItem(obj, i, objs[i]);
    }
  }
  // don't support other types
  else {
    PyErr_SetString(PyExc_TypeError, "First argument must be only be list, "
      "tuple, or dict");
    return NULL;
  }
  // free objs and return obj
  free(objs);
  return obj;
}
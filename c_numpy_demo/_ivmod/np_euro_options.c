/**
 * @file np_euro_options.c
 * @brief NumPy-enabled European option price + implied volatility functions.
 */

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif

#include <math.h>
#include <stdbool.h>

#include "Python.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL _IVMOD_ARRAY_API
#include "numpy/arrayobject.h"

// needed to call _black_vol and related functions
#include "../_ivlib/euro_options.h"
#include "../_ivlib/root_find.h"

// convenience method for checking if something is a numeric Python type
#define PyObject_is_numeric(x) (PyLong_CheckExact(x) || PyFloat_CheckExact(x))

// volatility type flags
#define BLACK_VOL_FLAG 0
#define BACHELIER_VOL_FLAG 1

// typedef since the implied volatility computation functions have many params
typedef scl_rf_res (*scl_vol_func)(vol_obj_args *, scl_opt_flag, double, double,
  double, int, bool);

/**
 * Computes implied volatilities for arguments broadcastable to ndarrays.
 * 
 * Do not call directly in production code.
 * 
 * @remark This function would have been impossible to write without reading the
 * NumPy C API documentation on array iterators like it was the Bible. The
 * latest version is https://numpy.org/devdocs/reference/c-api/iterator.html.
 * 
 * After broadcasting prices, forwards, strikes, ttms, discount factors, and
 * call/put flags to 1D ndarrays of the same shape, will apply _black_vol or
 * _bachelier_vol to each data point vector to compute Black implied vol or
 * Bachelier implied vol, respectively.
 * 
 * See docstrings of c_numpy_demo.ivmod.black_vol or
 * c_numpy_demo.ivmod.bachelier_vol for more details. An example of calling this
 * function directly on scalar data in the Python interpreter would be
 * 
 * >>> from c_numpy_demo._ivmod import _implied_vol_np
 * >>> _implied_vol_np(1.5, 99.72, 100.25, 0.6, 0.99765, 1, 0, 0, 0.5, 1.48e-8,
 * ... 0, 50, True)
 * 
 * @param self
 * @param args Positional-only argument vector containing prices, forwards,
 * strikes, ttms, discount factors, call/put flags, an int volatility type,
 * an int method flag, the initial vol guess, absolute and relative tolerance,
 * max iterations, and true/false on treating negative vols as NAN values.
 */
PyObject *_implied_vol_np(PyObject *self, PyObject *args) {
  /**
   * original objs for price, forward, strike, ttm, discount factor, call/put.
   * need to leave an additional slot for computed vols for multi-iterator.
   */
  PyObject *data[7];
  // vol type, method flag, and guess
  int vflag, mflag;
  double guess;
  // tolerance, relative tolerance, max iterations
  double tol, rtol;
  int maxiter;
  // whether to treat negative implied vols as NAN
  bool mask_neg;
  /**
   * parse arguments. note that only the data arguments will be broadcast; the
   * others must all match their types correctly.
   */
  if (!PyArg_ParseTuple(args, "OOOOOOiidddip", data, data + 1, data + 2,
    data + 3, data + 4, data + 5, &vflag, &mflag, &guess, &tol, &rtol, &maxiter,
    &mask_neg)) {
    // vol will be converted to vflag and method to mflag, both int
    PyErr_SetString(PyExc_RuntimeError, "Argument parsing failed. The first "
      "6 arguments must be broadcastable to ndarray, vol and method must be "
      "strings, guess, tol, and rtol must be floats, and maxiter must be int.");
    return NULL;
  }
  // use vflag to determine volatility function to use for computation
  scl_vol_func vol_func;
  if (vflag == BLACK_VOL_FLAG) {
    vol_func = &_black_vol;
  }
  else if (vflag == BACHELIER_VOL_FLAG) {
    vol_func = &_bachelier_vol;
  }
  else {
    PyErr_SetString(PyExc_ValueError,
      "vol must be either \"halley\" or \"newton\"");
    return NULL;
  }
  // loop index, volatility result, argument struct for _black_vol
  long i;
  vol_obj_args voa;
  scl_rf_res res;
  // doubles for individual options data post-conversion
  double f_data[6];
  // hold exception in case of double conversion error
  PyObject *exc;
  /**
   * if all elements whose pointers are in data are scalars, call _black_vol
   * once and return the value. mask_neg determines if negative values are kept
   * or instead replaced with NAN upon return.
   */
  if (PyObject_is_numeric(data[0]) && PyObject_is_numeric(data[1]) &&
    PyObject_is_numeric(data[2]) && PyObject_is_numeric(data[3]) &&
    PyObject_is_numeric(data[4]) && PyObject_is_numeric(data[5])) {
    // convert into C doubles. PyErr_Occurred used to check for errors.
    for (i = 0; i < 6; i++) {
      f_data[i] = PyFloat_AsDouble(data[i]);
      exc = PyErr_Occurred();
      if (exc != NULL) {
        PyErr_Format(exc, "Failed to convert arg %d to double\n", i + 1);
        return NULL;
      }
    }
    // populate vol_obj_args struct
    vol_obj_args_afill(voa, f_data[0], f_data[1], f_data[2], f_data[3],
      f_data[4], f_data[5]);
    // get result from chosen volatility function
    res = (*vol_func)(&voa,
      (scl_opt_flag) mflag, guess, tol, rtol, maxiter, false);
    // if mask_eg is true, then return NAN if res.res < 0
    if (mask_neg && (res.res < 0)) {
      return PyFloat_FromDouble(NAN);
    }
    // else return Python float
    return PyFloat_FromDouble(res.res);
  }
  /**
   * convert all data elements to numpy arrays of type numpy.float64. use the
   * NPY_ARRAY_IN_ARRAY flag to guarantee alignment and C contiguous layout.
   * scalars are first wrapped in a tuple to get a ndarray shape (1,). each new
   * ndarray has its shape checked; must be one-dimensional.
   */
  for (i = 0; i < 6; i++) {
    // if data[i] is scalar, wrap in tuple.
    if (PyObject_is_numeric(data[i])) {
      PyObject *tup;
      tup = PyTuple_New(1);
      PyTuple_SetItem(tup, 0, data[i]);
      // tuple has ownership of old scalar (stole reference)
      data[i] = tup;
    }
    // convert data[i] to ndarray and check for conversion error
    data[i] = (PyObject *) PyArray_FROM_OTF(data[i], NPY_FLOAT64,
      NPY_ARRAY_IN_ARRAY);
    exc = PyErr_Occurred();
    if (exc != NULL) {
      PyErr_Format(exc, "Failed to convert arg %d to ndarray\n", i + 1);
      return NULL;
    }
    // check that number of dimensions is 1
    if (PyArray_NDIM(data[i]) != 1) {
      PyErr_Format(PyExc_ValueError, "Positional arg %d cannot be converted"
        " to 1D numpy.ndarray\n", i + 1);
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
  for (i = 0; i < 6; i++) {
    local_len = PyArray_SHAPE((PyArrayObject *) data[i])[0];
    shared_len = (local_len > shared_len) ? local_len : shared_len;
  }
  // check that shapes are broadcastable. broadcast length 1 ndarrays.
  for (i = 0; i < 6; i++) {
    local_len = PyArray_SHAPE((PyArrayObject *) data[i])[0];
    if (local_len == 1) {
      // get base value as float, XDECREF original ndarray, create tuple
      double base;
      base = *((double *) PyArray_DATA(data[i]));
      PyArray_XDECREF((PyObject *) data[i]);
      data[i] = PyTuple_New((Py_ssize_t) shared_len);
      int j;
      for (j = 0; j < shared_len; j++) {
        PyTuple_SetItem(data[i], j, PyFloat_FromDouble(base));
      }
      // convert to ndarray with C order and aligned memory + check if error
      data[i] = (PyObject *) PyArray_FROM_OTF(data[i], NPY_FLOAT64,
        NPY_ARRAY_IN_ARRAY);
      if (exc != NULL) {
        PyErr_Format(exc, "Failed to convert arg %d to ndarray\n", i + 1);
        return NULL;
      }
    }
    else if (local_len == shared_len) { ; }
    else {
      PyErr_Format(PyExc_ValueError, "Could not broadcast (%ld,) to (%ld,)",
        (long) local_len, (long) shared_len);
      return NULL;
    }
  }
  // allocate tuple length shared_len as output
  data[6] = PyTuple_New(shared_len);
  /**
   * for each element in the first 6 input ndarrays, compute the implied vol and
   * save it to the apprpriate data location in data[6].
   */
  for (i = 0; i < shared_len; i++) {
    // initialize vol_obj_args struct
    vol_obj_args_afill(voa, ((double *) PyArray_DATA(data[0]))[i],
      ((double *) PyArray_DATA(data[1]))[i],
      ((double *) PyArray_DATA(data[2]))[i],
      ((double *) PyArray_DATA(data[3]))[i],
      ((double *) PyArray_DATA(data[4]))[i],
      (int) ((double *) PyArray_DATA(data[5]))[i]);
    /*
    printf("%f, %f, %f, %f, %f, %d\n", voa.price, voa.fwd, voa.strike,
      voa.ttm, voa.df, voa.is_call);
    */
    /**
     * compute result and save to scl_f_res res. if mask_neg is true, then
     * replace negative implied volatilities with NAN. then write to data[6].
     */
    res = (*vol_func)(&voa,
      (scl_opt_flag) mflag, guess, tol, rtol, maxiter, false);
    if (mask_neg && (res.res < 0)) {
      res.res = NAN;
    }
    PyTuple_SetItem(data[6], i, PyFloat_FromDouble(res.res));
  }
  // now that we are complete, return converted ndarray
  return PyArray_FROM_OTF(data[6], NPY_FLOAT64, NPY_ARRAY_OUT_ARRAY);
}

PyObject *_black_price_np(PyObject *self, PyObject *args) {
  Py_INCREF(Py_None);
  return Py_None;
}

PyObject *_bachelier_price_np(PyObject *self, PyObject *args) {
  Py_INCREF(Py_None);
  return Py_None;
}
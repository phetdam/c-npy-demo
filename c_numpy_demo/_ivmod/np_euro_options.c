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

/**
 * Computes the Black implied vol for arguments broadcastable to ndarrays.
 * 
 * After broadcasting prices, forwards, strikes, ttms, discount factors, and
 * call/put flags to 1D ndarrays of the same shape, will apply _black_vol to
 * each data point vector to compute Black implied vol.
 * 
 * See docstring of c_numpy_demo.ivmod.black_vol for more details. An example of
 * a call on scalar data in the Python interpreter would be
 * 
 * >>> from c_numpy_demo._ivmod import _black_vol_np
 * >>> _black_vol_np(1.5, 99.72, 100.25, 0.6, 0.99765, 1, 0, 0.5, 1.48e-8, 0,
 * ... 50, True)
 * 
 * @param self
 * @param args Positional-only argument vector containing prices, forwards,
 * strikes, ttms, discount factors, call/put flags, an int method flag, the
 * initial vol guess, absolute and relative tolerance, max iterations, and
 * true/false for whether to treat negative implied vols as NAN values.
 */
PyObject *_black_vol_np(PyObject *self, PyObject *args) {
  // original objs for price, forward, strike, ttm, discount factor, call/put
  PyObject *data[6];
  // method flag and guess
  int mflag;
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
  if (!PyArg_ParseTuple(args, "OOOOOOidddip", data, data + 1, data + 2,
    data + 3, data + 4, data + 5, &mflag, &guess, &tol, &rtol, &maxiter,
    &mask_neg)) {
    // method will be converted to mflag (int)
    PyErr_SetString(PyExc_RuntimeError, "Argument parsing failed. The first "
      "6 arguments must be broadcastable to ndarray, method must be a string, "
      "guess, tol, and rtol must be floats, and maxiter must be an int.");
    Py_INCREF(Py_None);
    return Py_None;
  }
  // loop index, volatility result, argument struct for _black_vol
  int i;
  vol_obj_args voa;
  scl_rf_res res;
  /**
   * if all elements whose pointers are in data are scalars, call _black_vol
   * once and return the value. mask_neg determines if negative values are kept
   * or instead replaced with NAN upon return.
   */
  if (PyObject_is_numeric(data[0]) && PyObject_is_numeric(data[1]) &&
    PyObject_is_numeric(data[2]) && PyObject_is_numeric(data[3]) &&
    PyObject_is_numeric(data[4]) && PyObject_is_numeric(data[5])) {
    // convert into C doubles. PyErr_Occurred used to check for errors.
    double f_data[6];
    PyObject *exc;
    int i;
    for (i = 0; i < 6; i++) {
      f_data[i] = PyFloat_AsDouble(data[i]);
      exc = PyErr_Occurred();
      if (exc != NULL) {
        PyErr_Format(exc, "Failed to convert positional arg %d to double\n", i);
        Py_INCREF(Py_None);
        return Py_None;
      }
    }
    // populate vol_obj_args struct
    vol_obj_args_afill(voa, f_data[0], f_data[1], f_data[2], f_data[3],
      f_data[4], f_data[5]);
    // get result from _black_vol (this is a macro)
    res = black_vol(&voa, (scl_opt_flag) mflag, guess, tol, rtol, maxiter);
    // if mask_eg is true, then return NAN if res.res < 0
    if (mask_neg && (res.res < 0)) {
      return PyFloat_FromDouble(NAN);
    }
    // else return Python float
    return PyFloat_FromDouble(res.res);
  }
  /**
   * todo: add logic for dealing with entire numpy arrays
   */

  // return -666 in the meantime
  return PyLong_FromLong(-666);
}

PyObject *_bachelier_vol_np(PyObject *self, PyObject *args) {
  // original objs for price, forward, strike, ttm, discount factor, call/put
  PyObject *data[6];
  // method flag and guess
  int mflag;
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
  if (!PyArg_ParseTuple(args, "OOOOOOidddip", data, data + 1, data + 2,
    data + 3, data + 4, data + 5, &mflag, &guess, &tol, &rtol, &maxiter,
    &mask_neg)) {
    // method will be converted to mflag (int)
    PyErr_SetString(PyExc_RuntimeError, "Argument parsing failed. The first "
      "6 arguments must be broadcastable to ndarray, method must be a string, "
      "guess, tol, and rtol must be floats, and maxiter must be an int.");
    Py_INCREF(Py_None);
    return Py_None;
  }
  // loop index, volatility result, argument struct for _black_vol
  int i;
  vol_obj_args voa;
  scl_rf_res res;
  /**
   * if all elements whose pointers are in data are scalars, call _black_vol
   * once and return the value. mask_neg determines if negative values are kept
   * or instead replaced with NAN upon return.
   */
  if (PyObject_is_numeric(data[0]) && PyObject_is_numeric(data[1]) &&
    PyObject_is_numeric(data[2]) && PyObject_is_numeric(data[3]) &&
    PyObject_is_numeric(data[4]) && PyObject_is_numeric(data[5])) {
    // convert into C doubles. PyErr_Occurred used to check for errors.
    double f_data[6];
    PyObject *exc;
    int i;
    for (i = 0; i < 6; i++) {
      f_data[i] = PyFloat_AsDouble(data[i]);
      exc = PyErr_Occurred();
      if (exc != NULL) {
        PyErr_Format(exc, "Failed to convert positional arg %d to double\n", i);
        Py_INCREF(Py_None);
        return Py_None;
      }
    }
    // populate vol_obj_args struct
    vol_obj_args_afill(voa, f_data[0], f_data[1], f_data[2], f_data[3],
      f_data[4], f_data[5]);
    // get result from _black_vol (this is a macro)
    res = black_vol(&voa, (scl_opt_flag) mflag, guess, tol, rtol, maxiter);
    // if mask_eg is true, then return NAN if res.res < 0
    if (mask_neg && (res.res < 0)) {
      return PyFloat_FromDouble(NAN);
    }
    // else return Python float
    return PyFloat_FromDouble(res.res);
  }
  /**
   * todo: add logic for dealing with entire numpy arrays
   */

  // return -666 in the meantime
  return PyLong_FromLong(-666);
}
PyObject *_black_price_np(PyObject *self, PyObject *args) {
  Py_INCREF(Py_None);
  return Py_None;
}

PyObject *_bachelier_price_np(PyObject *self, PyObject *args) {
  Py_INCREF(Py_None);
  return Py_None;
}
/**
 * @file np_broadcast.h
 * @brief Declaration for function in np_broadcast.c.
 */

#ifndef NP_BROADCAST_H
#define NP_BROADCAST_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif

#include "Python.h"

// convenience method for checking if something is a numeric Python type
#define PyObject_is_numeric(x) (PyLong_CheckExact(x) || PyFloat_CheckExact(x))

// row/column flags
#define NP_BROADCAST_ROWS 0
#define NP_BROADCAST_COLS 1

PyObject **np_float64_bcast_1d(PyObject **args, Py_ssize_t nargs,
  Py_ssize_t axis);
PyObject *np_float64_bcast_1d_ext(PyObject *self, PyObject *args);

#endif /* NP_BROADCAST_H */
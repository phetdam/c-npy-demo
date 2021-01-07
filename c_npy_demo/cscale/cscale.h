/**
 * @file cscale.h
 * @brief Declaration for function in `_cscale.c`.
 */

#ifndef CSCALE_H
#define CSCALE_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include "Python.h"

// convenience method for checking if something is a numeric Python type
#define PyObject_is_numeric(x) (PyLong_CheckExact(x) || PyFloat_CheckExact(x))

PyObject *stdscale(PyObject *self, PyObject *args, PyObject *kwargs);

#endif /* CSCALE_H */
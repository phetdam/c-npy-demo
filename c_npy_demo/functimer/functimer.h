/**
 * @file functimer.h
 * @brief header file for `c_npy_demo.functimer` extension module.
 */

#ifndef FUNCTIMER_H
#define FUNCTIMER_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#endif /* PY_SSIZE_T_CLEAN */

// argnames are self, args, and kwargs, as is typical for Python
PyObject *functimer_timeit_once(PyObject *, PyObject *, PyObject *);
PyObject *functimer_autorange(PyObject *, PyObject *, PyObject *);
PyObject *functimer_repeat(PyObject *, PyObject *, PyObject *);
PyObject *functimer_timeit(PyObject *, PyObject *, PyObject *);

#endif /* FUNCTIMER_H */
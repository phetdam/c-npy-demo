/**
 * @file timeitresult.c
 * @brief Slot function and method implementations for the `TimeitResult`.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stdbool.h>
#include <string.h>

#include "timeitresult.h"

// available array of units that unit is allowed to be equal to.
char const * const TimeitResult_units[] = {TimeitResult_UNITS, NULL};

/**
 * Return 1 if `unit` matches a value in `TimeitResult_units` else `0.
 * 
 * @param unit `char const *`, must be `NULL`-terminated
 * @returns 1 if valid unit in `TimeitResult_units, 0 otherwise.
 */
int TimeitResult_validate_unit(char const *unit) {
  int i = 0;
  // until the end of the array
  while (TimeitResult_units[i] != NULL) {
    // if identical, there's a match, so unit is valid
    if (strcmp(TimeitResult_units[i], unit) == 0) {
      return true;
    }
    i++;
  }
  // else unit is not valid
  return false;
}

/**
 * Custom destructor for the `TimeitResult` class.
 * 
 * Checks if `self` is `NULL` so that it can be safely used from C.
 * 
 * @param self `TimeitResult *` current instance
 */
void TimeitResult_dealloc(TimeitResult *self) {
  // if NULL, raise exception
  if (self == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "pointer to self is NULL");
    return;
  }
  // Py_DECREF times (guaranteed to not be NULL)
  Py_DECREF(self->times);
  // loop_times, brief might be NULL if never accessed as attribute
  Py_XDECREF(self->loop_times);
  Py_XDECREF(self->brief);
  // free the struct using the default function set to tp_free
  Py_TYPE(self)->tp_free((void *) self);
}

/**
 * Custom `__new__` implementation for `TimeitResult` class.
 * 
 * Includes `NULL` pointer checking for safer use from C. Since the
 * `TimeitResult` class is intended to be immutable, it does not implement a
 * custom initialization function (C analogue to `__init__`), so all necessary
 * initialization is performed here (C analogue to `__new__`).
 * 
 * @param type `PyTypeObject *` type object for the `TimeitResult` class
 * @param args `PyObject *` positional args
 * @param kwargs `PyObject *` keyword args
 * @returns `PyObject *` new instance of the `TimeitResult` struct or `NULL`
 *     if an error occurred. Sets error indicator.
 */
PyObject *TimeitResult_new(
  PyTypeObject *type, PyObject *args, PyObject *kwargs
) {
  // if type is NULL, raise exception and return NULL
  if (type == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "missing PyTypeObject *type");
    return NULL;
  }
  // new instance of the TimeitResult allocated by tp_alloc slot
  TimeitResult *self = (TimeitResult *) type->tp_alloc(type, 0);
  // if NULL, return NULL (error indicator set)
  if (self == NULL) {
    return NULL;
  }
  // loop_times and brief are initially set to NULL since they won't have any
  // value until the value is explicitly requested through attribute access
  self->loop_times = self->brief = NULL;
  // argument names
  char argnames[] = {"best", "unit", "number", "repeat", "times"};
  // parse args and kwargs. pass field addresses to PyArg_ParseTupleAndKeywords.
  // on error, need to Py_DECREF self, which is a new reference.
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "dsnnO", argnames, &(self->best), &(self->unit),
      &(self->number), &(self->repeat), &(self->times)
    )
  ) {
    Py_DECREF(self);
    return NULL;
  }
  // check that unit is one of several accepted values recorded in
  // TimeitResult_units. if not, set error indicator, Py_DECREF self
  if (!TimeitResult_validate_unit(self->unit)) {
    PyErr_SetString(
      PyExc_ValueError, "unit must be one of [" TimeitResult_UNITS_STR "]"
    );
    Py_DECREF(self);
    return NULL;
  }
  /**
   * check that number and repeat are positive. note that we don't check if
   * best is positive; maybe a weird "negative timer" was passed. on error, we
   * have to Py_DECREF self, which is a new reference.
   */
  if (self->number < 1) {
    PyErr_SetString(PyExc_ValueError, "number must be positive");
    Py_DECREF(self);
    return NULL;
  }
  if (self->repeat < 1) {
    PyErr_SetString(PyExc_ValueError, "repeat must be positive");
    Py_DECREF(self);
    return NULL;
  }
  // times must be tuple. on error, Py_DECREF self and set error indicator
  if (!PyTuple_CheckExact(self->times)) {
    PyErr_SetString(PyExc_TypeError, "times must be a tuple");
    Py_DECREF(self);
    return NULL;
  }
  // len(times) must equal repeat. if not, set error and Py_DECREF self. we use
  // PyTuple_GET_SIZE since self->times is already known to be a tuple
  if (PyTuple_GET_SIZE(self->times) != self->repeat) {
    PyErr_SetString(PyExc_ValueError, "len(times) must equal repeat");
    Py_DECREF(self);
    return NULL;
  }
  // all checks are complete, so return self
  return (PyObject *) self;
}

/**
 * Custom getter for `TimeitResult.loop_times`. Acts like cached `@property`.
 * 
 * @param self `TimeitResult *` current instance
 * @param closure `void *` (ignored)
 * @returns `PyObject *` tuple of trial times divided by number of loops per
 *     trial, the value given by `self->number`.
 */
PyObject *TimeitResult_getloop_times(TimeitResult *self, void *closure) {
  // dummy, returns (0.1, 0.2)
  PyObject *res = PyTuple_New(2);
  if (res == NULL) {
    return NULL;
  }
  // make some floats
  PyObject *f1, *f2;
  f1 = PyFloat_FromDouble(0.1);
  if (f1 == NULL) {
    Py_DECREF(res);
    return NULL;
  }
  f2 = PyFloat_FromDouble(0.2);
  if (f2 == NULL) {
    Py_DECREF(res);
    return NULL;
  }
  PyTuple_SET_ITEM(res, 0, f1);
  PyTuple_SET_ITEM(res, 1, f2);
  return res;
}

/**
 * Custom getter for `TimeitResult.brief`. Acts like cached `@property`.
 * 
 * @param self `TimeitResult *` current instance
 * @param closure `void *` (ignored)
 * @returns `PyObject *` Python Unicode object summary similar to the output
 *     from `timeit.main` printed when `timeit` is run using `python3 -m`.
 */
PyObject *TimeitResult_getbrief(TimeitResult *self, void *closure) {
  // dummy; returns "oowee"
  return PyUnicode_FromString("oowee");
}

/**
 * Custom `__repr__` implementation for `TimeitResult`.
 * 
 * @param self `TimeitResult *` current instance
 * @returns `PyObject *` Python unicode object representation for `self`
 */
PyObject *TimeitResult_repr(TimeitResult *self) {
  // dummy, returns "TimeitResult(bogus)"
  return PyUnicode_FromString("TimeitResult(bogus)");
}
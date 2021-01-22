/**
 * @file timeitresult.c
 * @brief Slot function and method implementations for the `TimeitResult`.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "timeitresult.h"

// available array of units that unit is allowed to be equal to.
char const * const TimeitResult_units[] = {TimeitResult_UNITS, NULL};

/**
 * Return 1 if `unit` matches a value in `TimeitResult_units` else 0.
 * 
 * @param unit `char const *`, must be `NULL`-terminated
 * @returns 1 if valid unit in `TimeitResult_units, 0 otherwise.
 */
int TimeitResult_validate_unit(char const *unit) {
  // return false if NULL and raise warning
  if (unit == NULL) {
    fprintf(stderr, "warning: %s: unit is NULL\n", __func__);
    return false;
  }
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
 * @note This is called when `Py_[X]DECREF` is called on a `TimeitResult *`.
 * 
 * @param self `TimeitResult *` current instance
 */
void TimeitResult_dealloc(TimeitResult *self) {
  // if NULL, raise exception
  if (self == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "pointer to self is NULL");
    return;
  }
  /**
   * times, loop_times, brief might be NULL, so we need Py_XDECREF. times can
   * be NULL if TimeitResult_new fails while loop_times and brief may be NULL
   * if they are never accessed by the user as attributes.
   */
  Py_XDECREF(self->times);
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
 * @note On error, note that we only `Py_DECREF` the `TimeitResult *` as the
 *     macro will call `TimeitResult_dealloc`, which will call `Py_[X]DECREF`
 *     as needed on the appropriate struct members.
 * 
 * @param type `PyTypeObject *` type object for the `TimeitResult` class
 * @param args `PyObject *` positional args
 * @param kwargs `PyObject *` keyword args
 * @returns `PyObject *` new instance (new reference) of the `TimeitResult`
 *     struct or `NULL` if an error occurred + sets error indicator.
 */
PyObject *TimeitResult_new(
  PyTypeObject *type, PyObject *args, PyObject *kwargs
) {
  // if type is NULL, raise exception and return NULL
  if (type == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "missing PyTypeObject *type");
    return NULL;
  }
  // if args is NULL, raise exception and return NULL
  if (args == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "missing PyObject *args");
    return NULL;
  }
  // new instance of the TimeitResult allocated by tp_alloc slot
  TimeitResult *self = (TimeitResult *) type->tp_alloc(type, 0);
  // if NULL, return NULL (error indicator set)
  if (self == NULL) {
    return NULL;
  }
  // set initial values to be overwritten later with args, kwargs
  self->best = self->number = self->repeat = 0;
  self->unit = NULL;
  self->times = self->loop_times = self->brief = NULL;
  // argument names (must be NULL-terminated)
  char *argnames[] = {"best", "unit", "number", "repeat", "times", NULL};
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
  // increase reference count to times since it is a Python object
  Py_INCREF(self->times);
  /**
   * check that unit is one of several accepted values recorded in
   * TimeitResult_units. if not, set error indicator, Py_DECREF self. note that
   * we doon't Py_DECREF self->tmies: tp_dealloc does that for us.
   */
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
   * have to Py_DECREF times, self, which is a new reference.
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
  // len(times) must equal repeat. if not, set error and Py_DECREF self
  if (PyTuple_Size(self->times) != self->repeat) {
    PyErr_SetString(PyExc_ValueError, "len(times) must equal repeat");
    Py_DECREF(self);
    return NULL;
  }
  // all checks are complete so return self
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
 * @returns `PyObject *` Python Unicode object representation for `self`. This
 *     is a new reference and is the C parallel to how `__repr__` would be
 *     implemented in pure Python.
 */
PyObject *TimeitResult_repr(TimeitResult *self) {
  /**
   * since PyUnicode_FromFormat doesn't accept any float-format strings we need
   * to create a Python float from self->best. we then pass the %R specifier
   * to PyUnicode_FromFormat to automatically call PyObject_Repr on the object.
   */
  PyObject *py_best = PyFloat_FromDouble(self->best);
  if (py_best == NULL) {
    return NULL;
  }
  // create Python string representation. Py_DECREF py_best on error
  PyObject *repr_str = PyUnicode_FromFormat(
    TIMEITRESULT_NAME "(best=%R, unit='%s', number=%zd, repeat=%zd, times=%R)",
    py_best, self->unit, self->number, self->repeat, self->times
  );
  if (repr_str == NULL) {
    Py_DECREF(py_best);
    return NULL;
  }
  // Py_DECREF py_best and return result
  Py_DECREF(py_best);
  return repr_str;
}
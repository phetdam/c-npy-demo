/**
 * @file timeitresult.c
 * @brief Slot function and method implementations for the `TimeitResult`.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "timeitresult.h"

// available array of units that unit is allowed to be equal to.
char const * const TimeitResult_units[] = {TimeitResult_UNITS, NULL};
/**
 * bases corresponding to TimeitResult_units. each ith value corresponds to
 * the number a time in seconds should be multipled by to get the times in the
 * units given by the ith entry in TimeitResult_units.
 */
double const TimeitResult_unit_bases[] = {TimeitResult_UNIT_BASES, 0};

/**
 * Check that `NULL`-terminated `ar` and `0`-terminated `br` have same length.
 * 
 * Usually `ar` will be `TimeitResult_units` and `br` will be
 * `TimeitResult_unit_bases` in production but this internal function allows
 * unit testing. The macro `TimeitResult_validate_units_bases` calls this
 * function, which is run during module initialization. Returns `1` if
 * lengths are matching else `0` is returned.
 * 
 * @note If one of the array has multiple trailing `NULL` values of `0` values,
 *     then the result of this function is inaccurate.
 * 
 * @param ar `NULL`-terminated `char const * const` array
 * @param br `0`-terminated `double const` array
 * @returns `1` if lengths are matching, `0` otherwise
 */
int _TimeitResult_validate_units_bases(
  char const * const * const ar, double const * const br
) {
  int i = 0;
  // loop until we reach the end of one of the arrays
  while ((ar[i] != NULL) && (br[i] != 0)) {
    i++;
  }
  // if last element of ar NULL and last element of br 0, then return true
  if ((ar[i] == NULL) && (br[i] == 0)) {
    return true;
  }
  return false;
}

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
 * Automatically determine units a time should be displayed in.
 * 
 * This is when `unit` is not passed to `functimer_timeit_enh` and will choose
 * the largest unit of time such that `fabs(best) >= 1`.
 * 
 * @param best Time in units of seconds
 * @param conv_p Memory location to write `best` converted to units returned
 *     by the function. If `NULL`, no writing is done.
 * @returns `char const *` to a value in `TimeitResult_units`
 */
char const *TimeitResult_autounit(double const best, double * const conv_p) {
  int i = 0;
  /**
   * loop through TimeitResult_unit_bases until we reach NULL or a unit such
   * that multiplying by the corresponding base b_i results in fabs(b_i * best)
   * to be less than 1. the final index is then decremented.
   */
  while (TimeitResult_unit_bases[i] != 0) {
    if (fabs(TimeitResult_unit_bases[i] * best) < 1) {
      break;
    }
    i++;
  }
  i--;
  // if conv_p is not NULL, write the converted time to that location
  if (conv_p != NULL) {
    *conv_p = TimeitResult_unit_bases[i] * best;
  }
  // return appropriate char const * from TimeitResult_units
  return TimeitResult_units[i];
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
   * times, loop_times, brief might be NULL, so we need Py_XDECREF. times can be
   * NULL if TimeitResult_new fails while loop_times, brief may be NULL if they
   * are never accessed by the user as attributes.
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
  // if type is NULL, set error indicator and return NULL
  if (type == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "missing PyTypeObject *type");
    return NULL;
  }
  // if args is NULL, set error indicator and return NULL
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
  // set pointers to NULL to be overwritten later with args, kwargs
  self->unit = NULL;
  self->times = self->loop_times = self->brief = NULL;
  // set default value for self->precision
  self->precision = 1;
  // argument names (must be NULL-terminated)
  char *argnames[] = {
    "best", "unit", "number", "repeat", "times", "precision", NULL
  };
  // parse args and kwargs. pass field addresses to PyArg_ParseTupleAndKeywords.
  // on error, need to Py_DECREF self, which is a new reference.
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "dsnnO|i", argnames, &(self->best), &(self->unit),
      &(self->number), &(self->repeat), &(self->times), &(self->precision)
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
   * check that number, precision, repeat are positive. we don't check if best
   * is positive; maybe a weird "negative timer" was passed. on error, we have
   * to Py_DECREF times, self, which is a new reference.
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
  if (self->precision < 1) {
    PyErr_SetString(PyExc_ValueError, "precision must be positive");
    Py_DECREF(self);
    return NULL;
  }
  // cap precison at TimeitResult_MAX_PRECISION. no human needs more precision
  // than the value given by TimeitResult_MAX_PRECISION.
  if (self->precision > TimeitResult_MAX_PRECISION) {
    PyErr_Format(
      PyExc_ValueError, "precision is capped at %d", TimeitResult_MAX_PRECISION
    );
    Py_DECREF(self);
    return NULL;
  }
  // times must be tuple. on error, Py_DECREF self and set error indicator
  if (!PyTuple_CheckExact(self->times)) {
    PyErr_SetString(PyExc_TypeError, "times must be a tuple");
    Py_DECREF(self);
    return NULL;
  }
  /**
   * len(times) must equal repeat. if not, set error and Py_DECREF self. we can
   * use PyTuple_GET_SIZE instead of PyTuple_Size since we already have
   * guaranteed that self->times is a tuple at this point.
   */
  if (PyTuple_GET_SIZE(self->times) != self->repeat) {
    PyErr_SetString(PyExc_ValueError, "len(times) must equal repeat");
    Py_DECREF(self);
    return NULL;
  }
  // check that all the elements of self->times are int or float
  for (Py_ssize_t i = 0; i < self->repeat; i++) {
    // get borrowed reference to ith element of self->times. note that we use
    // PyTuple_GET_ITEM since we are guaranteed not to go out of bounds.
    PyObject *time_i = PyTuple_GET_ITEM(self->times, i);
    // if neither int nor float, set error indicator, Py_DECREF self
    if (!PyLong_CheckExact(time_i) && !PyFloat_CheckExact(time_i)) {
      PyErr_SetString(PyExc_TypeError, "times must contain only int or float");
      Py_DECREF(self);
      return NULL;
    }
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
  // if self->loop_times is NULL, it has not been accessed before, so we have
  // to create a new Python tuple holding the per-loop times.
  if (self->loop_times == NULL) {
    // create new empty tuple. on error, return NULL. use PyTuple_GET_SIZE
    // since we already know that self->times is a tuple.
    self->loop_times = PyTuple_New(PyTuple_GET_SIZE(self->times));
    if (self->loop_times == NULL) {
      return NULL;
    }
    // for each trial, compute self->times[i] / self->number
    for (Py_ssize_t i = 0; i < self->repeat; i++) {
      /**
       * get value of self->times[i] as double. even though we already checked
       * that the contents of self->times are int or float, we still need to
       * use PyFloat_AsDouble since we allow elements of self->times to be int.
       * we also already checked that self->times is tuple and we won't go out
       * of bounds so we can use PyTuple_GET_ITEM instead of PyTuple_GetItem.
       */
      double time_i = PyFloat_AsDouble(PyTuple_GET_ITEM(self->times, i));
      PyObject *loop_time_i = PyFloat_FromDouble(
        time_i / ((double) self->number)
      );
      if (loop_time_i == NULL) {
        Py_DECREF(self->times);
        return NULL;
      }
      // add loop_time_i to self->loop_times. no error checking needed here.
      // note that the loop_time_i reference is stolen by self->loop_times.
      PyTuple_SET_ITEM(self->loop_times, i, loop_time_i);
    }
  }
  /**
   * Py_INCREF self->loop_times and then return. we have to Py_INCREF since
   * there is one reference in the instance and we need to give a reference to
   * the caller back in Python. same logic applies to self->brief or else the
   * Py_XDECREF can result in no references (created new reference in the
   * getter, given to Python caller, when tp_dealloc called may result in this
   * single new reference being set to zero even though caller holds a ref).
   */
  Py_INCREF(self->loop_times);
  return self->loop_times;
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
  // if self->brief is NULL, it has not been accessed before, so we have to
  // create a new Python string holding the brief. return NULL on error.
  if (self->brief == NULL) {
    /**
     * since PyUnicode_FromFormat doesn't format floats, we need to create a
     * rounded Python float from self->best (to nearest). we use
     * pow(10, self->precision) to give us the correct rounding precision.
     */
    double round_factor = pow(10, self->precision);
    PyObject *best_round = PyFloat_FromDouble(
      round(self->best * round_factor) / round_factor
    );
    if (best_round == NULL) {
      return NULL;
    }
    // get new reference to formatted string. use %R to use result of
    // PyObject_Repr on best_round in the formatted string.
    self->brief = PyUnicode_FromFormat(
      "%zd loops, best of %zd: %R %s per loop", self->number, self->repeat,
      best_round, self->unit
    );
    // don't need best_round anymore so Py_DECREF it
    Py_DECREF(best_round);
    // error. we already used Py_DECREF on best_round
    if (self->brief == NULL) {
      return NULL;
    }
  }
  // Py_INCREF self->brief + return. see TimeitResult_getloop_times comment.
  Py_INCREF(self->brief);
  return self->brief;
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
    TIMEITRESULT_NAME "(best=%R, unit='%s', number=%zd, repeat=%zd, times=%R, "
    "precision=%d)",
    py_best, self->unit, self->number, self->repeat, self->times,
    self->precision
  );
  if (repr_str == NULL) {
    Py_DECREF(py_best);
    return NULL;
  }
  // Py_DECREF py_best and return result
  Py_DECREF(py_best);
  return repr_str;
}
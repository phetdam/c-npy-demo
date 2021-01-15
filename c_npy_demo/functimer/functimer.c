/**
 * @file functimer.c
 * @brief Implementations for declarations in `functimer.h`.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "functimer.h"

/**
 * Operates in a similar manner to `timeit.timeit`.
 * 
 * Docstring in `_modinit.c`.
 * 
 * @param args PyObject * tuple of positional arguments
 * @param kwargs PyObject * dict of named arguments
 * @returns PyObject * numeric value
 */
PyObject *functimer_timeit_once(
  PyObject *self, PyObject *args, PyObject *kwargs
) {
  // callable, args, kwargs, timer function
  PyObject *func, *func_args, *func_kwargs, *timer;
  // if timer NULL after arg parsing, set to time.perf_counter
  func_args = func_kwargs = timer = NULL;
  // number of times to execute the callable with args and kwargs
  Py_ssize_t number = 1000000;
  // names of arguments
  char *argnames[] = {"func", "args", "kwargs", "timer", "number", NULL};
  // parse args and kwargs; sets appropriate exception so no need to check
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|OOOn", argnames, &func, &func_args, &func_kwargs,
      &timer, &number
    )
  ) { return NULL; }
  /**
   * Py_XINCREF func_args. we do this because if func_args is NULL (not given
   * by user), then we get new PyObject * reference for it. thus, at the end of
   * the function, we will have to Py_DECREF func_args. doing Py_XINCREF
   * increments borrowed reference count if func_args is provided by user so
   * Py_DECREF doesn't decrement BORROWED refs, which is of course bad.
   * 
   * note we don't need to do this for func_kwargs since PyObject_Call lets us
   * pass NULL if there are no keyword arguments to pass.
   */
  Py_XINCREF(func_args);
  // if func_args is NULL, no args specified, so set it to be empty tuple
  if (func_args == NULL) {
    func_args = PyTuple_New(0);
    if (func_args == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "couldn't allocate new empty tuple");
      return NULL;
    }
  }
  /**
   * if timer is NULL, then import time module and then attempt to import
   * perf_counter. time_module, perf_counter are initially set to NULL so that
   * Py_XDECREF can be used on them it even if if they are never referenced.
   */
  PyObject *time_module, *time_perf_counter;
  time_module = time_perf_counter = NULL;
  // if timer NULL, then timer was not provided so we use time.perf_counter
  if (timer == NULL) {
    time_module = PyImport_ImportModule("time");
    // if module failed to import, exception is set. Py_DECREF func_args
    if (time_module == NULL) {
      Py_DECREF(func_args);
      return NULL;
    }
    // try to get perf_counter from time
    time_perf_counter = PyObject_GetAttrString(time_module, "perf_counter");
    // if NULL, exception set. Py_DECREF time_module, func_args
    if (time_perf_counter == NULL) {
      Py_DECREF(time_module);
      Py_DECREF(func_args);
      return NULL;
    }
    // set timer to time.perf_counter
    timer = time_perf_counter;
  }
  /**
   * check that func_args is tuple and that func_kwargs is a dict (or is NULL).
   * need to also Py_XDECREF time_module, time_perf_counter, and Py_DECREF
   * func_args so that we clean up our garbage correctly.
   */
  if (!PyTuple_CheckExact(func_args)) {
    PyErr_SetString(PyExc_TypeError, "args must be a tuple");
    Py_XDECREF(time_module);
    Py_XDECREF(time_perf_counter);
    Py_DECREF(func_args);
    return NULL;
  }
  if ((func_kwargs != NULL) && !PyDict_CheckExact(func_kwargs)) {
    PyErr_SetString(PyExc_TypeError, "kwargs must be a dict");
    Py_XDECREF(time_module);
    Py_XDECREF(time_perf_counter);
    Py_DECREF(func_args);
    return NULL;
  }
  // starting, ending times recorded by timer function
  PyObject *start_time, *end_time;
  // get starting time from timer function. 
  start_time = PyObject_CallObject(timer, NULL);
  // if NULL, exception was raised. Py_DECREF and Py_XDECREF as needed
  if (start_time == NULL) {
    Py_XDECREF(time_module);
    Py_XDECREF(time_perf_counter);
    Py_DECREF(func_args);
    return NULL;
  }
  // if not numeric, raised exception. Py_DECREF and Py_XDECREF as needed. note
  // we also need to Py_DECREF start_time since it's a new reference
  if (!PyNumber_Check(start_time)) {
    PyErr_SetString(PyExc_TypeError, "timer must return a numeric value");
    Py_XDECREF(time_module);
    Py_XDECREF(time_perf_counter);
    Py_DECREF(func_args);
    Py_DECREF(start_time);
    return NULL;
  }
  // call function number times with func_args and func_kwargs
  for (Py_ssize_t i = 0; i < number; i++) {
    // if NULL is returned, an exception has been raised. Py_DECREF, Py_XDECREF
    if (PyObject_Call(func, func_args, NULL) == NULL) {
      Py_XDECREF(time_module);
      Py_XDECREF(time_perf_counter);
      Py_DECREF(func_args);
      Py_DECREF(start_time);
      return NULL;
    }
  }
  // get ending time from timer function
  end_time = PyObject_CallObject(timer, NULL);
  // if NULL, exception raised; Py_DECREF and Py_XDECREF as needed
  if (end_time == NULL) {
    Py_XDECREF(time_module);
    Py_XDECREF(time_perf_counter);
    Py_DECREF(func_args);
    Py_DECREF(start_time);
    return NULL;
  }
  // if not numeric, raised exception. Py_DECREF and Py_XDECREF as needed; also
  // need to Py_DECREF end_time since we got a new reference for it
  if (!PyNumber_Check(end_time)) {
    PyErr_SetString(PyExc_TypeError, "timer must return a numeric value");
    Py_XDECREF(time_module);
    Py_XDECREF(time_perf_counter);
    Py_DECREF(func_args);
    Py_DECREF(start_time);
    Py_DECREF(end_time);
    return NULL;
  }
  // compute time difference
  PyObject *timedelta = PyNumber_Subtract(end_time, start_time);
  // if NULL, failure. set message for exception, Py_DECREF and Py_XDECREF
  if (timedelta == NULL) {
    PyErr_SetString(PyExc_ArithmeticError, "unable to compute time delta");
    Py_XDECREF(time_module);
    Py_XDECREF(time_perf_counter);
    Py_DECREF(func_args);
    Py_DECREF(start_time);
    Py_DECREF(end_time);
    return NULL;
  }
  // decrement refcounts for time_module, time_perf_counter (may be NULL)
  Py_XDECREF(time_module);
  Py_XDECREF(time_perf_counter);
  // decrement refcounts for func_args, start_time, end_time
  Py_DECREF(func_args);
  Py_DECREF(start_time);
  Py_DECREF(end_time);
  // return the time delta
  return timedelta;
}

/**
 * Operates in a similar manner to `timeit.Timer.autorange`.
 * 
 * Docstring in `_modinit.c`.
 */
PyObject *functimer_autorange(
  PyObject *self, PyObject *args, PyObject *kwargs
) {
  // callable, args, kwargs
  PyObject *func, *func_args, *func_kwargs;
  func_args = func_kwargs = NULL;
  // names of arguments
  char *argnames[] = {"func", "args", "kwargs", NULL};
  // parse args and kwargs; sets appropriate exception so no need to check
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|OO", argnames, &func, &func_args, &func_kwargs
    )
  ) { return NULL; }
  // dummy return
  Py_INCREF(Py_None);
  return Py_None;
}
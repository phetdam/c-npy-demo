/**
 * @file functimer.c
 * @brief Implementations for declarations in `functimer.h`.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stdbool.h>

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
  // check that number is greater than 0. if not, set exception and exit
  if (number < 1) {
    PyErr_SetString(PyExc_ValueError, "number must be positive");
    return NULL;
  }
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
      return NULL;
    }
  }
  /**
   * check that func_args is tuple and that func_kwargs is a dict (or is NULL).
   * need to also Py_DECREF func_args to clean up the garbage
   */
  if (!PyTuple_CheckExact(func_args)) {
    PyErr_SetString(PyExc_TypeError, "args must be a tuple");
    Py_DECREF(func_args);
    return NULL;
  }
  if ((func_kwargs != NULL) && !PyDict_CheckExact(func_kwargs)) {
    PyErr_SetString(PyExc_TypeError, "kwargs must be a dict");
    Py_DECREF(func_args);
    return NULL;
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
    if (PyObject_Call(func, func_args, func_kwargs) == NULL) {
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
 * Docstring in `_modinit.c`. No callback allowed.
 * 
 * @note `kwargs` is `NULL` if no named args are passed.
 */
PyObject *functimer_autorange(
  PyObject *self, PyObject *args, PyObject *kwargs
) {
  /**
   * callable, args, kwargs, timer function. we don't actually need to use
   * these directly in autorange; these will just be used with
   * PyArg_ParseTupleAndKeywords so we can do some argument checking. since all
   * references are borrowed we don't need to Py_[X]DECREF any of them.
   */
  PyObject *func, *func_args, *func_kwargs, *timer;
  func_args = func_kwargs = timer = NULL;
  // names of arguments
  char *argnames[] = {"func", "args", "kwargs", "timer", NULL};
  // parse args and kwargs; sets appropriate exception so no need to check
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|OOO", argnames, &func, &func_args, &func_kwargs, &timer
    )
  ) { return NULL; }
  // number of times to run the function func (default 1)
  Py_ssize_t number = 1;
  // current number multipler
  Py_ssize_t multipler = 1;
  // bases to scale number of times to run so number = bases[i] * multipler
  int bases[] = {1, 2, 5};
  // total of the time reported by functimer_timeit_once
  double time_total;
  /**
   * Py_XINCREF kwargs. it may be NULL, in which case we will create a new dict
   * with the number = number mapping which will be Py_XDECREF'd at the end of
   * the function. otherwise, the borrowed ref gets its refcount incremented
   * now and then decremented at the end of the function.
   */
  Py_XINCREF(kwargs);
  // if NULL, create new dict
  if (kwargs == NULL) {
    kwargs = PyDict_New();
    // return NULL on failure
    if (kwargs == NULL) {
      return NULL;
    }
  }
  // keep going as long as number < PY_SSIZE_T_MAX / 10
  while (true) {
    // for each of the bases
    for (int i = 0; i < 3; i++) {
      // set number = bases[i] * multipler
      number = bases[i] * multipler;
      // create new PyLongObject from number
      PyObject *number_ = PyLong_FromSsize_t(number);
      // if NULL, return NULL, but also remember to Py_DECREF kwargs
      if (number_ == NULL) {
        Py_DECREF(kwargs);
        return NULL;
      }
      // set time_total to 0 to initialize
      time_total = 0;
      // try running the function number times
      for (Py_ssize_t j = 0; j < number; j++) {
        // add number_ to kwargs. Py_DECREF kwargs, number_
        if (PyDict_SetItemString(kwargs, "number", number_) < 0) {
          Py_DECREF(kwargs);
          Py_DECREF(number_);
          return NULL;
        }
        // save the returned time from functimer_timeit_once. the self, args,
        // kwargs refs are all borrowed so no need to Py_INCREF them.
        PyObject *timeit_time = functimer_timeit_once(self, args, kwargs);
        // if NULL, return NULL. let functimer_timeit set the error indicator.
        // Py_DECREF kwargs and number_ (they are new references)
        if (timeit_time == NULL) {
          Py_DECREF(kwargs);
          Py_DECREF(number_);
          return NULL;
        }
        // convert timeit_time to Python float and Py_DECREF timeit_time_temp
        PyObject *timeit_time_temp = timeit_time;
        timeit_time = PyNumber_Float(timeit_time);
        Py_DECREF(timeit_time_temp);
        // on error, exit. Py_DECREF kwargs and number_
        if (timeit_time == NULL) {
          Py_DECREF(kwargs);
          Py_DECREF(number_);
          return NULL;
        }
        // attempt to get double from timeit_time and Py_DECREF it when done
        double timeit_time_ = PyFloat_AsDouble(timeit_time);
        Py_DECREF(timeit_time);
        // check if error occurred (borrowed ref)
        PyObject *err_type = PyErr_Occurred();
        // if not NULL, then exit. error indicator already set. do Py_DECREFs
        if (err_type != NULL) {
          Py_DECREF(kwargs);
          Py_DECREF(number_);
          return NULL;
        }
        // add timeit_time_ to time_total
        time_total = time_total + timeit_time_;
      }
      // computation of time_total complete. if time_total >= 0.2 s, break
      if (time_total >= 0.2) {
        break;
      }
      // done with number_ so Py_DECREF it
      Py_DECREF(number_);
    }
    // if number > PY_SSIZE_T_MAX / 10, then break the while loop. emit warning
    if (number > (PY_SSIZE_T_MAX / 10.)) {
      PyErr_WarnEx(
        PyExc_RuntimeWarning, "return value will exceed PY_SSIZE_T_MAX / 10", 1
      );
      break;
    }
    // multiply multiplier by 10. we want 1, 2, 5, 10, 20, 50, ...
    multipler = multipler * 10;
  }
  // done with kwargs; Py_DECREF it
  Py_DECREF(kwargs);
  // return Python int from number. NULL returned on failure
  return PyLong_FromSsize_t(number);
}
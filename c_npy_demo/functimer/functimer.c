/**
 * @file functimer.c
 * @brief Implementations for declarations in `functimer.h`.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <float.h>
#include <stdbool.h>

#include "functimer.h"
#include "timeitresult.h"

/**
 * Operates in a similar manner to `timeit.timeit`.
 * 
 * Docstring in `_modinit.c`.
 * 
 * @param args tuple of positional arguments
 * @param kwargs dict of named arguments
 * @returns `PyObject *` numeric value
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
      args, kwargs, "O|OO$On", argnames, &func, &func_args, &func_kwargs,
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
  // get new start time; time was lost checking if start_time is valid
  Py_DECREF(start_time);
  start_time = PyObject_CallObject(timer, NULL);
  // if NULL, exception was raised. Py_DECREF and Py_XDECREF as needed
  if (start_time == NULL) {
    Py_XDECREF(time_module);
    Py_XDECREF(time_perf_counter);
    Py_DECREF(func_args);
    return NULL;
  }
  PyObject *func_res;
  // call function number times with func_args and func_kwargs
  for (Py_ssize_t i = 0; i < number; i++) {
    // call function and Py_XDECREF its result
    func_res = PyObject_Call(func, func_args, func_kwargs);
    Py_XDECREF(func_res);
    // if NULL is returned, an exception has been raised. Py_DECREF, Py_XDECREF
    if (func_res == NULL) {
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
 * 
 * @param args tuple of positional arguments
 * @param kwargs dict of named arguments
 * @returns `PyLongObject *` cast to `PyObject *` of number of loops in a trial
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
  // names of arguments
  char *argnames[] = {"func", "args", "kwargs", "timer", NULL};
  // parse args and kwargs; sets appropriate exception so no need to check
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|OO$O", argnames, &func, &func_args, &func_kwargs, &timer
    )
  ) { return NULL; }
  // number of times to run the function func (starts at 1)
  Py_ssize_t number;
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
        // add number_ to kwargs. Py_DECREF kwargs, number_ on failure
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
      // done with number_ so Py_DECREF it
      Py_DECREF(number_);
      // computation of time_total complete. if time_total >= 0.2 s, break the
      // outermost for loop. no more changes to number
      if (time_total >= 0.2) {
        break;
      }
    }
    // if time_total >= 0.2, break from the while loop
    if (time_total >= 0.2) {
      break;
    }
    // if number > PY_SSIZE_T_MAX / 10, then break the while loop. emit warning
    // and if an exception is raised (return == -1), Py_DECREF kwargs
    if (number > (PY_SSIZE_T_MAX / 10.)) {
      if(
        PyErr_WarnEx(
          PyExc_RuntimeWarning,
          "return value will exceed PY_SSIZE_T_MAX / 10", 1
        ) < 0
      ) {
        Py_DECREF(kwargs);
        return NULL;
      }
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

/**
 * Operates in a similar manner to `timeit.Timer.repeat`.
 * 
 * Docstring in `_modinit.c`.
 * 
 * @param args tuple of positional arguments
 * @param kwargs dict of named arguments
 * @returns `PyListObject *` list of trial items cast to `PyObject *`
 */
PyObject *functimer_repeat(PyObject *self, PyObject *args, PyObject *kwargs) {
  /**
   * callable, args, kwargs, timer function. we don't actually need to use
   * these directly; these will just be used with PyArg_ParseTupleAndKeywords
   * so we can do some argument checking. since all references are borrowed we
   * don't need to Py_[X]DECREF any of them.
   */
  PyObject *func, *func_args, *func_kwargs, *timer;
  // number of times to execute callable with arguments; not used
  Py_ssize_t number;
  // number of times to repeat the call to functimer_timeit_once
  Py_ssize_t repeat = 5;
  // names of arguments
  char *argnames[] = {
    "func", "args", "kwargs", "timer", "number", "repeat", NULL
  };
  // parse args and kwargs; sets appropriate exception so no need to check
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|OO$Onn", argnames, &func, &func_args, &func_kwargs,
      &timer, &number, &repeat
    )
  ) { return NULL; }
  // check that repeat is greater than 0. if not, set exception and exit. we
  // don't need to check number since functimer_timeit_once will do the check.
  if (repeat < 1) {
    PyErr_SetString(PyExc_ValueError, "repeat must be positive");
    return NULL;
  }
  // Python string for repeat
  PyObject *repeat_obj = PyUnicode_FromString("repeat");
  // check if repeat was passed as a named arg. if so, we remove it from kwargs
  // so we can directly pass kwargs into functimer_timeit_once
  if (kwargs != NULL) {
    // get return value from PyDict_Contains
    int has_repeat = PyDict_Contains(kwargs, repeat_obj);
    // if error, Py_DECREF repeat_obj and return NULL
    if (has_repeat < 0) {
      Py_DECREF(repeat_obj);
      return NULL;
    }
    // if repeat is in kwargs, then remove it from kwargs
    if (has_repeat) {
      // if failed, Py_DECREF repeat_obj and then return NULL
      if (PyDict_DelItemString(kwargs, "repeat") < 0) {
        Py_DECREF(repeat_obj);
        return NULL;
      }
    }
    // else do nothing
  }
  // don't need repeat_obj anymore so Py_DECREF it
  Py_DECREF(repeat_obj);
  // allocate new list to return
  PyObject *func_times = PyList_New(repeat);
  // for each trial
  for (Py_ssize_t i = 0; i < repeat; i++) {
    // get time result from functimer_timeit_once
    PyObject *func_time = functimer_timeit_once(self, args, kwargs);
    // if NULL then there was exception. Py_DECREF func_times and return NULL
    if (func_time == NULL) {
      Py_DECREF(func_times);
      return NULL;
    }
    // else set index i of func_times to func_time (reference stolen). no need
    // to use PyList_SetItem since we won't be out of bounds/leak old refs.
    PyList_SET_ITEM(func_times, i, func_time);
  }
  // return the list of times returned from functimer_timeit_once
  return func_times;
}

/**
 * Operates in a similar manner to `timeit.main` but returns a `TimeitResult`.
 * 
 * Docstring in `_modinit.c`.
 * 
 * @param args tuple of positional arguments
 * @param kwargs dict of named arguments
 * @returns `TimeitResult *` of timing results cast to `PyObject *`
 */
PyObject *functimer_timeit_enh(
  PyObject *self, PyObject *args, PyObject *kwargs
) {
  // callable, args, kwargs, timer function
  PyObject *func, *func_args, *func_kwargs, *timer;
  func = func_args = func_kwargs = timer = NULL;
  // number of times to execute func in a trial, number of trials. if number is
  // PY_SSIZE_T_MIN, then functimer_autorange is used to set number
  Py_ssize_t number = PY_SSIZE_T_MIN;
  Py_ssize_t repeat = 5;
  // display unit to use. if NULL then it will be automatically selected
  char const *unit = NULL;
  // precision to display brief output with
  int precision = 1;
  // argument names
  char *argnames[] = {
    "func", "args", "kwargs", "timer", "number", "repeat", "unit", "precision",
    NULL
  };
  // parse args and kwargs; sets appropriate exception automatically
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|OO$Onnsi", argnames, &func, &func_args, &func_kwargs,
      &timer, &number, &repeat, &unit, &precision
    )
  ) { return NULL; }
  /**
   * we defer checking of func, func_args, func_kwargs to functimer_timeit_once
   * and so must check number, repeat, unit, precision
   */
  // number must be positive (unless PY_SSIZE_T_MIN)
  if ((number != PY_SSIZE_T_MIN) && (number < 1)) {
    PyErr_SetString(PyExc_ValueError, "number must be positive");
    return NULL;
  }
  // repeat must be positive
  if (repeat < 1) {
    PyErr_SetString(PyExc_ValueError, "repeat must be positive");
    return NULL;
  }
  // unit must be valid. if NULL, it will be decided later
  if ((unit != NULL) && !TimeitResult_validate_unit(unit)) {
    PyErr_SetString(
      PyExc_ValueError, "unit must be one of [" TimeitResult_UNITS_STR "]"
    );
    return NULL;
  }
  // precision must be positive
  if (precision < 1) {
    PyErr_SetString(PyExc_ValueError, "precision must be positive");
    return NULL;
  }
  // precision must be <= TimeitResult_MAX_PRECISION
  if (precision > TimeitResult_MAX_PRECISION) {
    PyErr_Format(
      PyExc_ValueError, "precision is capped at %d", TimeitResult_MAX_PRECISION
    );
    return NULL;
  }
  // if precision >= floor(TimeitResult_MAX_PRECISION / 2), print warning.
  if (precision >= (TimeitResult_MAX_PRECISION / 2)) {
    if (
      PyErr_WarnFormat(
        PyExc_UserWarning, 1, "value of precision is rather high (>= %d). "
        "consider passing a lower value for better brief readability.",
        TimeitResult_MAX_PRECISION / 2
      ) < 0
    ) { return NULL; }
  }
  /**
   * now that all the parameters have been checked, we need to delegate
   * arguments to the right methods. func, func_args, and func_kwargs need to
   * be put together into a tuple, while timer needs to be put in a new dict
   * (if not NULL). if number == PY_SSIZE_T_MIN, then we need to call
   * functimer_autorange to give a value for number.
   */
  // number of positional arguments that will go in the new args tuple. if
  // func_args, func_kwargs are not NULL, n_new_args incremented by 1 each.
  int n_new_args = 1;
  if (func_args != NULL) {
    n_new_args++;
  }
  if(func_kwargs != NULL) {
    n_new_args++;
  }
  // new args tuple (for functimer_autorange, functimer_repeat)
  PyObject *new_args = PyTuple_New(n_new_args);
  if (new_args == NULL) {
    return NULL;
  }
  // pass func reference to new_args. Py_INCREF obviously needed.
  Py_INCREF(func);
  PyTuple_SET_ITEM(new_args, 0, func);
  // if func_args is not NULL, then Py_INCREF and add to position 1
  if (func_args != NULL) {
    Py_INCREF(func_args);
    PyTuple_SET_ITEM(new_args, 1, func_args);
  }
  // if func_kwargs is not NULL, Py_INCREF and add to last position
  if (func_kwargs != NULL) {
    Py_INCREF(func_kwargs);
    PyTuple_SET_ITEM(new_args, n_new_args - 1, func_args);
  }
  // new kwargs dict (for functimer_autorange, functimer_repeat)
  PyObject *new_kwargs = PyDict_New();
  // if NULL, only need to Py_DECREF new_args, as func, func_args, func_kwargs
  // had references stolen by new_args
  if (new_kwargs == NULL) {
    Py_DECREF(new_args);
    return NULL;
  }
  // if timer is not NULL, add timer to new_kwargs (borrow ref)
  if (timer != NULL) {
    PyDict_SetItemString(new_kwargs, "timer", timer);
  }
  // number as a PyLongObject * to be passed to new_kwargs when ready
  PyObject *number_ = NULL;
  // if number == PY_SSIZE_T_MIN, then we need to use functimer_autorange to
  // determine the number of loops to run in a trial
  if (number == PY_SSIZE_T_MIN) {
    // get result from functimer_autorange
    number_ = functimer_autorange(self, new_args, new_kwargs);
    // on error, need to Py_DECREF new_args, new_kwargs (new refs)
    if (number_ == NULL) {
      Py_DECREF(new_args);
      Py_DECREF(new_kwargs);
      return NULL;
    }
    // attempt to convert number_ into Py_ssize_t
    number = PyLong_AsSsize_t(number_);
    // if number == -1, error (don't even need to check PyErr_Occurred). then
    // Py_DECREF new_args, new_kwargs, number_ (also new ref)
    if (number == -1) {
      Py_DECREF(new_args);
      Py_DECREF(new_kwargs);
      Py_DECREF(number_);
      return NULL;
    }
  }
  // if number_ is NULL, it wasn't initialized in if statement, so initialize
  if (number_ == NULL) {
    number_ = PyLong_FromSsize_t(number);
    // Py_DECREF new_args, new_kwargs on error
    if (number_ == NULL) {
      Py_DECREF(new_args);
      Py_DECREF(new_kwargs);
      return NULL;
    }
  }
  // add number_ to new_kwargs. Py_DECREF new_args, new_kwargs, number_ if err
  if (PyDict_SetItemString(new_kwargs, "number", number_) < 0) {
    Py_DECREF(new_args);
    Py_DECREF(new_kwargs);
    Py_DECREF(number_);
    return NULL;
  }
  // repeat as a PyLongObject * to be passed to new_kwargs
  PyObject *repeat_ = PyLong_FromSsize_t(repeat);
  // on error, Py_DECREF new_args, new_kwargs, number_
  if (repeat_ == NULL) {
    Py_DECREF(new_args);
    Py_DECREF(new_kwargs);
    Py_DECREF(number_);
    return NULL;
  }
  // add repeat_ to new_kwargs, Py_DECREF on error (include repeat_)
  if (PyDict_SetItemString(new_kwargs, "repeat", repeat_) < 0) {
    Py_DECREF(new_args);
    Py_DECREF(new_kwargs);
    Py_DECREF(number_);
    Py_DECREF(repeat_);
    return NULL;
  }
  // call functimer_repeat with new_args, new_kwargs (borrowed refs) and get
  // times_list, the list of times in seconds for each of the repeat trials
  PyObject *times_list = functimer_repeat(self, new_args, new_kwargs);
  // if NULL, exception was set, so Py_DECREF as needed
  if (times_list == NULL) {
    Py_DECREF(new_args);
    Py_DECREF(new_kwargs);
    Py_DECREF(number_);
    Py_DECREF(repeat_);
    return NULL;
  }
  // create tuple out of times_list and Py_DECREF times_list; no longer needed
  PyObject *times_tuple = PySequence_Tuple(times_list);
  Py_DECREF(times_list);
  // Py_DECREF as needed if error
  if (times_tuple == NULL) {
    Py_DECREF(new_args);
    Py_DECREF(new_kwargs);
    Py_DECREF(number_);
    Py_DECREF(repeat_);
    return NULL;
  }
  // best time (for now, in seconds, as double)
  double best = DBL_MAX;
  // loop through times in times_tuple
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(times_tuple); i++) {
    // try to convert item from times_tuple to PyFloatObject. won't be out of
    // bounds, so no need to use PyTuple_GetItem. returned ref is borrowed.
    PyObject *time_i_ = PyNumber_Float(PyTuple_GET_ITEM(times_tuple, i));
    // if conversion to float fails, Py_DECREF as needed (include times_tuple)
    if (time_i_ == NULL) {
      Py_DECREF(new_args);
      Py_DECREF(new_kwargs);
      Py_DECREF(number_);
      Py_DECREF(repeat_);
      Py_DECREF(times_tuple);
      return NULL;
    }
    // convert time_i_ to double and Py_DECREF (no longer needed)
    double time_i = PyFloat_AsDouble(time_i_);
    Py_DECREF(time_i_);
    // if error occurred, Py_DECREF
    if (PyErr_Occurred()) {
      Py_DECREF(new_args);
      Py_DECREF(new_kwargs);
      Py_DECREF(number_);
      Py_DECREF(repeat_);
      Py_DECREF(times_tuple);
      return NULL;
    }
    // update best based on the value of time_i
    best = (time_i < best) ? time_i : best;
  }
  // divide best by number to get per-loop time
  best = best / ((double) number);
  /**
   * if unit is NULL, then we call TimeitResult_autounit to return char const *
   * pointer to TimeitResult_units (no need to free of course). the new value
   * of best is written to the address best is stored in. unit is never NULL.
   */
  if (unit == NULL) {
    unit = TimeitResult_autounit(best, &best);
  }
  /**
   * now we start creating Python objects from C values to pass to the
   * TimeitResult constructor. create new Python float from best time, in units
   * of unit. Py_DECREF on error as needed.
   */
  PyObject *best_ = PyFloat_FromDouble(best);
  if (best_ == NULL) {
    Py_DECREF(new_args);
    Py_DECREF(new_kwargs);
    Py_DECREF(number_);
    Py_DECREF(repeat_);
    Py_DECREF(times_tuple);
    return NULL;
  }
  // create Python string from unit, Py_DECREF on error (include best_)
  PyObject *unit_ = PyUnicode_FromString(unit);
  if (unit_ == NULL) {
    Py_DECREF(new_args);
    Py_DECREF(new_kwargs);
    Py_DECREF(number_);
    Py_DECREF(repeat_);
    Py_DECREF(times_tuple);
    Py_DECREF(best_);
    return NULL;
  }
  // create Python int from precision, Py_DECREF on error (include unit_)
  PyObject *precision_ = PyLong_FromLong(precision);
  if (precision_ == NULL) {
    Py_DECREF(new_args);
    Py_DECREF(new_kwargs);
    Py_DECREF(number_);
    Py_DECREF(repeat_);
    Py_DECREF(times_tuple);
    Py_DECREF(best_);
    Py_DECREF(unit_);
    return NULL;
  }
  // create new tuple of arguments to be passed to TimeitResult.__new__. note
  // that since we use PyTuple_Pack, we still need to Py_DECREF on error.
  //"best", "unit", "number", "repeat", "times", "precision", NULL
  PyObject *res_args = PyTuple_Pack(
    6, best_, unit_, number_, repeat_, times_tuple, precision_
  );
  if (res_args == NULL) {
    Py_DECREF(new_args);
    Py_DECREF(new_kwargs);
    Py_DECREF(number_);
    Py_DECREF(repeat_);
    Py_DECREF(times_tuple);
    Py_DECREF(best_);
    Py_DECREF(unit_);
    Py_DECREF(precision_);
    return NULL;
  }
  /**
   * no Py_[X]DECREF of func, func_args, func_kwargs since refs were stolen.
   * however, we Py_DECREF everything else: new_args, new_kwargs, number_,
   * repeat_, times_tuple, best_, unit_, precision_, res_args
   */
  Py_DECREF(new_args);
  Py_DECREF(new_kwargs);
  Py_DECREF(number_);
  Py_DECREF(repeat_);
  Py_DECREF(times_tuple);
  Py_DECREF(best_);
  Py_DECREF(unit_);
  Py_DECREF(precision_);
  // create new TimeitResult instance using res_args and Py_DECREF res_args
  PyObject *tir = TimeitResult_new(&TimeitResult_type, res_args, NULL);
  Py_DECREF(res_args);
  // TimeitResult_new will set error indicator on error
  if (tir == NULL) {
    return NULL;
  }
  // return new reference
  return tir;
}
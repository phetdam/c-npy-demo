/**
 * @file _timeapi.c
 * @brief Provides the callable API for the timing of Python functions in
 *     `functimer`.  The implementation in C means that there is less
 *     measurement error introduced by the slow execution speed of Python
 *     loops (see the implementation of the `timeit` module).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <float.h>
#include <stdbool.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// include C APIs of _timeresult, _timeunit extension modules
#include "timeresult.h"
#include "timeunit.h"

// timeit_once docstring
PyDoc_STRVAR(
  timeit_once_doc,
  "timeit_once(func, args=None, kwargs=None, *, timer=None, number=1000000)"
  "\n--\n\n"
  "Operates in the same way as ``timeit.timeit```, i.e. the same way as\n"
  "``timeit.Timer.timeit``. ``func`` will be executed with positional\n"
  "args ``args`` and keyword args ``kwargs`` ``number`` times and the total\n"
  "execution time will be returned. ``timer`` is the timing function and must\n"
  "return time in units of fractional seconds."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "func : callable\n"
  "    Callable to time\n"
  "args : tuple, default=None\n"
  "    Tuple of positional args to pass to ``func``\n"
  "kwargs : dict, default=None\n"
  "    Dict of named arguments to pass to ``func``\n"
  "timer : function, default=None\n"
  "    Timer function, defaults to ``time.perf_counter`` which returns time\n"
  "    in [fractional] seconds. If specified, must return time in fractional\n"
  "    seconds as a float and not take any positional arguments.\n"
  "number : int, default=1000000\n"
  "    Number of times to call ``func``"
  "\n\n"
  "Returns\n"
  "-------\n"
  "float\n"
  "    The time required for ``func`` to be executed ``number`` times with\n"
  "    args ``args`` and kwargs ``kwargs``, in units of seconds."
);
// argument names for timeit_once
static const char *timeit_once_argnames[] = {
  "func", "args", "kwargs", "timer", "number", NULL
};
/**
 * Operates in a similar manner to `timeit.timeit`.
 * 
 * @param args tuple of positional arguments
 * @param kwargs dict of named arguments, may be `NULL`
 * @returns New reference to `PyFloatObject *`, `NULL` with exception on error
 */
static PyObject *
timeit_once(PyObject *self, PyObject *args, PyObject *kwargs)
{
  // callable, args, kwargs, timer function
  PyObject *func, *func_args, *func_kwargs, *timer;
  // if timer NULL after arg parsing, set to time.perf_counter
  func_args = func_kwargs = timer = NULL;
  // number of times to execute the callable with args and kwargs
  Py_ssize_t number = 1000000;
  // parse args and kwargs; sets appropriate exception so no need to check
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|O!O!$On", (char **) timeit_once_argnames, &func,
      &PyTuple_Type, &func_args, &PyDict_Type, &func_kwargs, &timer, &number
    )
  ) {
    return NULL;
  }
  // check that func is callable and that timer, if not NULL, is callable
  if (!PyCallable_Check(func)) {
    PyErr_SetString(PyExc_ValueError, "func must be callable");
    return NULL;
  }
  if (timer != NULL && !PyCallable_Check(timer)) {
    PyErr_SetString(PyExc_ValueError, "timer must be callable");
    return NULL;
  }
  // check that number is greater than 0. if not, set exception and exit
  if (number < 1) {
    PyErr_SetString(PyExc_ValueError, "number must be positive");
    return NULL;
  }
  // if func_args is NULL, no args specified, so set it to be empty tuple. this
  // is required by PyObject_Call; kwargs (func_kwargs) however can be NULL.
  if (func_args == NULL) {
    func_args = PyTuple_New(0);
    if (func_args == NULL) {
      return NULL;
    }
  }
  // else Py_INCREF func_args since it will by Py_DECREF'd at finalization
  else {
    Py_INCREF(func_args);
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
      goto except_func_args;
    }
    // try to get perf_counter from time
    time_perf_counter = PyObject_GetAttrString(time_module, "perf_counter");
    // if NULL, exception set. Py_DECREF time_module, func_args
    if (time_perf_counter == NULL) {
      goto except_time_module;
    }
    // set timer to time.perf_counter
    timer = time_perf_counter;
  }
  // starting, ending times recorded by timer function, function result
  PyObject *start_time, *end_time, *func_res;
  // get starting time from timer function, NULL on error
  start_time = PyObject_CallObject(timer, NULL);
  if (start_time == NULL) {
    goto except_time_perf_counter;
  }
  // if not numeric, raised exception. Py_DECREF and Py_XDECREF as needed. note
  // we also need to Py_DECREF start_time since it's a new reference
  if (!PyFloat_Check(start_time)) {
    PyErr_SetString(
      PyExc_ValueError, "timer must return a float starting value"
    );
    goto except_start_time;
  }
  // call function number times with func_args and func_kwargs
  for (Py_ssize_t i = 0; i < number; i++) {
    // call function and Py_XDECREF its result (we never need it)
    func_res = PyObject_Call(func, func_args, func_kwargs);
    Py_XDECREF(func_res);
    // if NULL is returned, an exception has been raised. Py_DECREF, Py_XDECREF
    if (func_res == NULL) {
      goto except_start_time;
    }
  }
  // get ending time from timer function, NULL on error
  end_time = PyObject_CallObject(timer, NULL);
  if (end_time == NULL) {
    goto except_start_time;
  }
  // if not float, raise exception. Py_DECREF and Py_XDECREF as needed; also
  // need to Py_DECREF end_time since we got a new reference for it
  if (!PyFloat_Check(end_time)) {
    PyErr_SetString(PyExc_ValueError, "timer must return a float ending value");
    goto except_end_time;
  }
  // compute time difference
  PyObject *timedelta = PyNumber_Subtract(end_time, start_time);
  // if NULL, failure. set message for exception, Py_DECREF and Py_XDECREF
  if (timedelta == NULL) {
    goto except_end_time;
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
// clean up end_time reference on exception
except_end_time:
  Py_DECREF(end_time);
// clean up start_time reference on exception
except_start_time:
  Py_DECREF(start_time);
// clean up time perf_counter reference on exception
except_time_perf_counter:
  Py_XDECREF(time_perf_counter);
// clean up time module on exception
except_time_module:
  Py_XDECREF(time_module);
// clean up func_args on exception
except_func_args:
  Py_DECREF(func_args);
  return NULL;
}

// autorange docstring
PyDoc_STRVAR(
  autorange_doc,
  "autorange(func, args=None, kwargs=None, *, timer=None)\n"
  "\n--\n\n"
  "Automatically determine number of times to call ``functimer.timeit_once``."
  "\n\n"
  "Operates in the same way as ``timeit.Timer.autorange``. ``autorange``\n"
  "calls ``timeit_once`` 1, 2, 5, 10, 20, 50, etc. times until the total\n"
  "execution time is >= 0.2 seconds. The number of times ``timeit_once`` is\n"
  "to be called as determined by this formula is then returned."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "func : callable\n"
  "    Callable to time\n"
  "args : tuple, default=None\n"
  "    Tuple of positional args to pass to ``func``\n"
  "kwargs : dict, default=None\n"
  "    Dict of named arguments to pass to ``func``\n"
  "timer : function, default=None\n"
  "    Timer function, defaults to ``time.perf_counter`` which returns time\n"
  "    in [fractional] seconds. If specified, must return time in fractional\n"
  "    seconds as a float and not take any positional arguments."
  "\n\n"
  "Returns\n"
  "-------\n"
  "int"
);
// argument names for autorange
static const char *autorange_argnames[] = {
  "func", "args", "kwargs", "timer", NULL
};
/**
 * Operates in a similar manner to `timeit.Timer.autorange` but no callback.
 * 
 * @note `kwargs` is `NULL` if no named args are passed.
 * 
 * @param args tuple of positional arguments
 * @param kwargs dict of named arguments, may be `NULL`
 * @returns New reference to `PyLongObject *`, `NULL` with exception on error
 */
static PyObject *
autorange(PyObject *self, PyObject *args, PyObject *kwargs)
{
  /**
   * callable, args, kwargs, timer function. we don't actually need to use
   * these directly in autorange; these will just be used with
   * PyArg_ParseTupleAndKeywords so we can do some argument checking. since all
   * references are borrowed we don't need to Py_[X]DECREF any of them.
   */
  PyObject *func, *func_args, *func_kwargs, *timer;
  // parse args and kwargs; sets appropriate exception so no need to check
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|O!O!$O", (char **) autorange_argnames, &func,
      &PyTuple_Type, &func_args, &PyDict_Type, &func_kwargs, &timer
    )
  ) {
    return NULL;
  }
  // number of times to run the function func (starts at 1)
  Py_ssize_t number;
  // current number multipler
  Py_ssize_t multipler = 1;
  // bases to scale number of times to run so number = bases[i] * multipler
  int bases[] = {1, 2, 5};
  // total of the time reported by timeit_once
  double time_total;
  // if kwargs NULL, create new dict to hold number=number mapping that will be
  // used in call to timeit_once along with self, args.
  if (kwargs == NULL) {
    kwargs = PyDict_New();
    // return NULL on failure
    if (kwargs == NULL) {
      return NULL;
    }
  }
  // else set kwargs to a copy of the original dict it points to. this is ok
  // since kwargs is a borrowed reference to a dict.
  else {
    kwargs = PyDict_Copy(kwargs);
    if (kwargs == NULL) {
      return NULL;
    }
  }
  // PyLongObject * wrapper for number, the returned time from timeit_once.
  // timeit_time is either PyFloatObject * or a subtype.
  PyObject *number_, *timeit_time;
  // keep going as long as number < PY_SSIZE_T_MAX / 10
  while (true) {
    // for each of the bases
    for (int i = 0; i < 3; i++) {
      // set number = bases[i] * multipler
      number = bases[i] * multipler;
      // create new PyLongObject from number. NULL on error
      number_ = PyLong_FromSsize_t(number);
      if (number_ == NULL) {
        goto except_kwargs;
      }
      // set time_total to 0 to start + add number_ to kwargs. NULL on error
      time_total = 0;
      if (PyDict_SetItemString(kwargs, "number", number_) < 0) {
        goto except_number_;
      }
      // save the returned time from timeit_once. the self, args, kwargs refs
      // are all borrowed so no need to Py_INCREF them. NULL on error.
      timeit_time = timeit_once(self, args, kwargs);
      if (timeit_time == NULL) {
        goto except_number_;
      }
      // attempt to get time_total from timeit_time, which we know is float or
      // some subclass, i.e. PyFloat_Check returns true. Py_DECREF when done.
      time_total = PyFloat_AsDouble(timeit_time);
      Py_DECREF(timeit_time);
      // if not NULL, then exit. error indicator already set. do Py_DECREFs
      if (PyErr_Occurred()) {
        goto except_number_;
      }
      // computation of time_total complete. if time_total >= 0.2 s, Py_DECREF
      // kwargs and return number_, which wraps number.
      if (time_total >= 0.2) {
        Py_DECREF(kwargs);
        return number_;
      }
      // done with number_ in this loop iteration so Py_DECREF it
      Py_DECREF(number_);
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
        goto except_kwargs;
      }
      break;
    }
    // multiply multiplier by 10. we want 1, 2, 5, 10, 20, 50, ...
    multipler *= 10;
  }
  // done with kwargs so Py_DECREF it. number_, timeit_time already Py_DECREF'd
  Py_DECREF(kwargs);
  // return Python int from number. NULL returned on failure
  return PyLong_FromSsize_t(number);
// clean up on exception
except_number_:
  Py_DECREF(number_);
except_kwargs:
  Py_DECREF(kwargs);
  return NULL;
}

// timeit_repeat docstring
PyDoc_STRVAR(
  timeit_repeat_doc,
  "timeit_repeat(func, args=None, kwargs=None, *, timer=None, number=1000000, "
  "repeat=5)"
  "\n--\n\n"
  "Operates in the same way as ``timeit.repeat``, i.e. the same way as\n"
  "``timeit.Timer.repeat``. ``repeat`` calls to ``functimer.timeit_once`` are\n"
  "executed, where ``number`` gives the number of calls to ``func`` made in\n"
  "each call to ``functimer.timeit_once`` made by this function."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "func : callable\n"
  "    Callable to time\n"
  "args : tuple, default=None\n"
  "    Tuple of positional args to pass to ``func``\n"
  "kwargs : dict, default=None\n"
  "    Dict of named arguments to pass to ``func``\n"
  "timer : function, default=None\n"
  "    Timer function, defaults to ``time.perf_counter`` which returns time\n"
  "    in [fractional] seconds. If specified, must return time in fractional\n"
  "    seconds as a float and not take any positional arguments."
  "number : int, default=1000000\n"
  "    Number of times ``func`` is called by ``functimer.timeit_once``\n"
  "repeat : int, default=5\n"
  "    Number of times to call ``functimer.timeit_once``"
  "\n\n"
  "Returns\n"
  "-------\n"
  "numpy.ndarray\n"
  "    Times in fractional seconds taken for each call to ``timeit_once``,\n"
  "    the time taken to call ``func`` ``number`` times, shape ``(repeat,)``."
);
// argument names for timeit_repeat
static const char *timeit_repeat_argnames[] = {
    "func", "args", "kwargs", "timer", "number", "repeat", NULL
};
/**
 * Operates in a similar manner to `timeit.Timer.repeat`.
 * 
 * @param args tuple of positional arguments
 * @param kwargs dict of named arguments, may be `NULL`
 * @returns New reference to `PyArrayObject *`, `NULL` with exception on error
 */
static PyObject *
timeit_repeat(PyObject *self, PyObject *args, PyObject *kwargs) {
  /**
   * callable, args, kwargs, timer function. we don't actually need to use
   * these directly; these will just be used with PyArg_ParseTupleAndKeywords
   * so we can do some argument checking. since all references are borrowed we
   * don't need to Py_[X]DECREF any of them.
   */
  PyObject *func, *func_args, *func_kwargs, *timer;
  // number of times to execute callable with arguments; not actually used here
  Py_ssize_t number;
  // number of times to repeat the call to timeit_once. this is checked.
  Py_ssize_t repeat = 5;
  // parse args and kwargs, NULL on error
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|O!O!$Onn", (char **) timeit_repeat_argnames,
      &func, &PyTuple_Type, &func_args,
      &PyDict_Type, &func_kwargs, &timer, &number, &repeat
    )
  ) {
    return NULL;
  }
  // check that repeat is greater than 0. if not, set exception and exit. we
  // don't need to check number since timeit_once will do the check.
  if (repeat < 1) {
    PyErr_SetString(PyExc_ValueError, "repeat must be positive");
    return NULL;
  }
  // if kwargs is not NULL, we copy it. we must check that kwargs contains
  // repeat, which we remove so kwargs can be sent with args to timeit_once.
  if (kwargs != NULL) {
    kwargs = PyDict_Copy(kwargs);
    if (kwargs == NULL) {
      return NULL;
    }
    // Python string for "repeat", NULL on error
    PyObject *repeat_obj = PyUnicode_FromString("repeat");
    if (repeat_obj == NULL) {
      goto except_kwargs;
    }
    // get return value from PyDict_Contains and Py_DECREF unneeded repeat_obj
    int has_repeat = PyDict_Contains(kwargs, repeat_obj);
    Py_DECREF(repeat_obj);
    // if error, has_repeat < 0. return NULL
    if (has_repeat < 0) {
      goto except_kwargs;
    }
    // if repeat is in kwargs, then remove it from kwargs
    if (has_repeat) {
      // if failed, PyDict_DelItemString returns -1. return NULL
      if (PyDict_DelItemString(kwargs, "repeat") < 0) {
        goto except_kwargs;
      }
    }
    // else do nothing
  }
  // allocate new ndarray to return, type NPY_DOUBLE, flags NPY_ARRAY_CARRAY
  npy_intp dims[] = {(npy_intp) repeat};
  PyArrayObject *func_times;
  func_times = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  // NULL on error. on success, get data pointer
  if (func_times == NULL) {
    goto except_kwargs;
  }
  double *func_times_data = (double *) PyArray_DATA(func_times);
  // write the time result for each trial into func_times
  for (npy_intp i = 0; i < (npy_intp) repeat; i++) {
    // get time result from timeit_once, a PyFloatObject *. NULL on error
    PyObject *func_time = timeit_once(self, args, kwargs);
    if (func_time == NULL) {
      goto except_func_times;
    }
    // else write double value from func_time to func_times. use PyErr_Occurred
    // to check if PyFloat_AsDouble returned error. Py_DECREF in all cases.
    func_times_data[i] = PyFloat_AsDouble(func_time);
    if (PyErr_Occurred()) {
      Py_DECREF(func_time);
      goto except_func_times;
    }
    Py_DECREF(func_time);
  }
  // return the ndarray of times returned from timeit_once
  return (PyObject *) func_times;
// clean up on error
except_func_times:
  Py_DECREF(func_times);
except_kwargs:
  Py_XDECREF(kwargs);
  return NULL;
}

// timeit_plus docstring
PyDoc_STRVAR(
  timeit_plus_doc,
  "timeit_plus(func, args=None, kwargs=None, *, timer=None, number=None, "
  "repeat=5, unit=None, precision=1)"
  "\n--\n\n"
  "A callable, approximate C implementation of ``timeit.main``. Returns a\n"
  "``TimeResult`` instance with timing statistics whose attribute ``brief``\n"
  "provides the same exact string output as ``timeit.main``."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "func : callable\n"
  "    Callable to time\n"
  "args : tuple, default=None\n"
  "    Tuple of positional args to pass to ``func``\n"
  "kwargs : dict, default=None\n"
  "    Dict of named arguments to pass to ``func``\n"
  "timer : function, default=None\n"
  "    Timer function, defaults to ``time.perf_counter`` which returns time\n"
  "    in [fractional] seconds. If specified, must return time in fractional\n"
  "    seconds as a float and not take any positional arguments.\n"
  "number : int, default=None\n"
  "    Number of times to call ``func`` in a single timing trial. If not\n"
  "    specified, this is determined by a call to ``functimer.autorange``.\n"
  "repeat : int, default=5\n"
  "    Number of total timing trials. This value is directly passed to\n"
  "    ``functimer.timeit_repeat``, which is called in this function.\n"
  "unit : {" Py__timeunit_UNITS_STR "}, default=None\n"
  "    Units to display the per-loop time stored in the ``brief`` attribute\n"
  "    of the returned ``TimeResult`` with. If not specified, determined\n"
  "    by an internal function. Accepts the same values as ``timeit.main``.\n"
  "precision : int, default=1\n"
  "    Number of decimal places to display the per-loop time stored in the\n"
  "    ``brief`` attribute of the returned ``TimeResult`` with. The\n"
  "    maximum allowed precision is "
  xstringify(Py__timeunit_MAX_PRECISION) ", which should be plenty."
  "\n\n"
  "Returns\n"
  "-------\n"
  "TimeResult\n"
);
// argument names for timeit_plus
static const char *timeit_plus_argnames[] = {
  "func", "args", "kwargs", "timer", "number",
  "repeat", "unit", "precision", NULL
};
/**
 * Operates in a similar manner to `timeit.main` but returns a `TimeResult`.
 * 
 * @param args tuple of positional arguments
 * @param kwargs dict of named arguments, may be `NULL`
 * @returns New reference to `TimeResult *`, `NULL` with exception on error
 */
PyObject *
timeit_plus(PyObject *self, PyObject *args, PyObject *kwargs)
{
  // callable, args, kwargs, timer function
  PyObject *func, *func_args, *func_kwargs, *timer;
  func = func_args = func_kwargs = timer = NULL;
  // number of times to execute func in a trial, number of trials. if number is
  // PY_SSIZE_T_MIN, then autorange is used to set number
  Py_ssize_t number = PY_SSIZE_T_MIN;
  Py_ssize_t repeat = 5;
  // display unit to use. if NULL then it will be automatically selected
  char const *unit = NULL;
  // precision to display brief output with
  int precision = 1;
  // parse args and kwargs. we defer checking of func, timer, number, repeat
  // to timeit_repeat and so must check unit, precision.
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|O!O!$Onnsi", (char **) timeit_plus_argnames,
      &func, &PyTuple_Type, &func_args, &PyDict_Type, &func_kwargs,
      &timer, &number, &repeat, &unit, &precision
    )
  ) {
    return NULL;
  }
  // unit must be valid. if NULL, chosen by Py__timeunit_autoselect_unit
  if ((unit != NULL) && !Py__timeunit_validate_unit(unit)) {
    PyErr_SetString(
      PyExc_ValueError, "unit must be one of [" Py__timeunit_UNITS_STR "]"
    );
    return NULL;
  }
  // precision must be positive and <= Py__timeunit_MAX_PRECISION
  if (precision < 1) {
    PyErr_SetString(PyExc_ValueError, "precision must be positive");
    return NULL;
  }
  if (precision > Py__timeunit_MAX_PRECISION) {
    PyErr_Format(
      PyExc_ValueError, "precision is capped at %d", Py__timeunit_MAX_PRECISION
    );
    return NULL;
  }
  // if precision >= floor(Py__timeunit_MAX_PRECISION / 2), print warning.
  // warning can be turned into exception so check PyErr_WarnFormat return.
  if (precision >= (Py__timeunit_MAX_PRECISION / 2)) {
    if (
      PyErr_WarnFormat(
        PyExc_UserWarning, 1, "value of precision is rather high (>= %d). "
        "consider passing a lower value for better brief readability.",
        Py__timeunit_MAX_PRECISION / 2
      ) < 0
    ) {
      return NULL;
    }
  }
  /**
   * now that all the parameters have been checked, we need to delegate the
   * right arguments to the right functions. to simplify the case where we
   * might have one of func_args, func_kwargs NULL, we put func into a tuple
   * and func_args, func_kwargs, timer in a new dict (if not NULL). if
   * number == PY_SSIZE_T_MIN, then we need to call autorange to give a value
   * for number; a PyFloatObject * wrapper for number will be added to the new
   * kwargs dict. A PyLongObject * wrapper for repeat is then added to the new
   * kwargs dict before the timeit_repeat call.
   */
  // new positional tuple (for autorange, repeat). on success, add func to it.
  // must Py_INCREF since else a borrowed ref will be stolen.
  PyObject *new_args = PyTuple_New(1);
  if (new_args == NULL) {
    return NULL;
  }
  Py_INCREF(func);
  PyTuple_SET_ITEM(new_args, 0, func);
  // new kwargs dict (for autorange, repeat). NULL on error.
  PyObject *new_kwargs = PyDict_New();
  if (new_kwargs == NULL) {
    goto except_new_args;
  }
  // if func_args is not NULL, add to new_kwargs (ref borrowed). -1 on error.
  // repeat for func_kwargs and for timer.
  if (func_args != NULL) {
    if (PyDict_SetItemString(new_kwargs, "args", func_args) < 0) {
      goto except_new_kwargs;
    }
  }
  if (func_kwargs != NULL) {
    if (PyDict_SetItemString(new_kwargs, "kwargs", func_kwargs) < 0) {
      goto except_new_kwargs;
    }
  }
  if (timer != NULL) {
    if (PyDict_SetItemString(new_kwargs, "timer", timer) < 0) {
      goto except_new_kwargs;
    }
  }
  // number as a PyLongObject * to be passed to new_kwargs when ready
  PyObject *number_ = NULL;
  // if number == PY_SSIZE_T_MIN, then we need to use autorange to
  // determine the number of loops to run in a trial
  if (number == PY_SSIZE_T_MIN) {
    // get result from autorange; we pass new_args, new_kwargs. NULL on error
    number_ = autorange(self, new_args, new_kwargs);
    if (number_ == NULL) {
      goto except_new_kwargs;
    }
    // attempt to convert number_ into Py_ssize_t. -1 on error, which is an
    // invalid value for number, so we don't have to check PyErr_Occurred.
    number = PyLong_AsSsize_t(number_);
    if (number == -1) {
      goto except_number_;
    }
  }
  // if number_ is NULL, not initialized in if block, so initialize from number
  if (number_ == NULL) {
    number_ = PyLong_FromSsize_t(number);
    if (number_ == NULL) {
      goto except_new_kwargs;
    }
  }
  // add number_ to new_kwargs. PyDict_SetItemString returns -1 on error
  if (PyDict_SetItemString(new_kwargs, "number", number_) < 0) {
    goto except_number_;
  }
  // add repeat as a PyLongObject * to be passed to new_kwargs, NULL on error
  PyObject *repeat_ = PyLong_FromSsize_t(repeat);
  if (repeat_ == NULL) {
    goto except_number_;
  }
  // add repeat_ to new_kwargs, PyDict_SetItemString returns -1 on error
  if (PyDict_SetItemString(new_kwargs, "repeat", repeat_) < 0) {
    goto except_repeat_;
  }
  // call timeit_repeat with new_args, new_kwargs (borrowed refs) and get
  // func_times, the ndarray of times for each trial. NULL on error
  PyObject *func_times = timeit_repeat(self, new_args, new_kwargs);
  if (func_times == NULL) {
    goto except_repeat_;
  }
  // best time (for now, in seconds, as double) + get func_times data pointer
  double best, *func_times_data;
  best = DBL_MAX;
  func_times_data = (double *) PyArray_DATA((PyArrayObject *) func_times);
  // loop through times in func_times to find the shortest time. note that
  // func_times is guaranteed to have shape (repeat,), type NPY_DOUBLE
  for (npy_intp i = 0; i < (npy_intp) repeat; i++) {
    // update best based on the value of func_times_data[i]
    best = (func_times_data[i] < best) ? func_times_data[i] : best;
  }
  // divide best by number to get per-loop time
  best = best / (double) number;
  /**
   * if unit is NULL, call Py__timeunit_autoselect_unit to return char const *
   * pointer to a unit string in TimeResult_units (no need to free) + overwrite
   * best with best in units of the returned unit, which is never NULL.
   */
  if (unit == NULL) {
    unit = Py__timeunit_autoselect_unit(best, &best);
  }
  // now we start creating Python objects from C values to pass to the
  // TimeResult constructor. create new Python float from best.
  PyObject *best_ = PyFloat_FromDouble(best);
  if (best_ == NULL) {
    goto except_func_times;
  }
  // create Python string from unit, NULL on error
  PyObject *unit_ = PyUnicode_FromString(unit);
  if (unit_ == NULL) {
    goto except_best_;
  }
  // create Python int from precision, NULL on error
  PyObject *precision_ = PyLong_FromLong(precision);
  if (precision_ == NULL) {
    goto except_unit_;
  }
  // create new tuple of arguments to be passed to TimeResult.__new__
  PyObject *res_args = PyTuple_Pack(
    6, best_, unit_, number_, repeat_, func_times, precision_
  );
  if (res_args == NULL) {
    goto except_precision_;
  }
  /**
   * no Py_[X]DECREF of func, func_args, func_kwargs since refs were stolen.
   * we Py_DECREF everything else except res_args, unit_. res_args we are not
   * done with. note unit_ must be kept alive or else when res_args is freed,
   * unit_ is also freed, and the TimeResult that is returned will have its
   * unit member point to garbage. leaving unit_ alive does leak a little bit
   * of memory however; the only other option is for the TimeResult object to
   * manage its own buffer with PyMem_Raw[Malloc|Free].
   */
  Py_DECREF(new_args);
  Py_DECREF(new_kwargs);
  Py_DECREF(number_);
  Py_DECREF(repeat_);
  Py_DECREF(func_times);
  Py_DECREF(best_);
  Py_DECREF(precision_);
  // create new TimeResult instance using res_args and Py_DECREF res_args
  PyObject *tir = PyTimeResult_New(res_args, NULL);
  Py_DECREF(res_args);
  // NULL on error with exception set, else return new ref on success
  if (tir == NULL) {
    return NULL;
  }
  return tir;
// clean up on error
except_precision_:
  Py_DECREF(precision_);
except_unit_:
  Py_DECREF(unit_);
except_best_:
  Py_DECREF(best_);
except_func_times:
  Py_DECREF(func_times);
except_repeat_:
  Py_DECREF(repeat_);
except_number_:
  Py_DECREF(number_);
except_new_kwargs:
  Py_DECREF(new_kwargs);
except_new_args:
  Py_DECREF(new_args);
  return NULL;
}

// _timeapi methods
static PyMethodDef _timeapi_methods[] = {
  {
    "timeit_once", (PyCFunction) timeit_once,
    METH_VARARGS | METH_KEYWORDS, timeit_once_doc
  },
  {
    "timeit_repeat", (PyCFunction) timeit_repeat,
    METH_VARARGS | METH_KEYWORDS, timeit_repeat_doc
  },
  {
    "autorange", (PyCFunction) autorange,
    METH_VARARGS | METH_KEYWORDS, autorange_doc
  },
  {
    "timeit_plus", (PyCFunction) timeit_plus,
    METH_VARARGS | METH_KEYWORDS, timeit_plus_doc
  },
  // sentinel required; needs to have at least one NULL in it
  {NULL, NULL, 0, NULL}
};

// _timeapi docstring
PyDoc_STRVAR(
  _timeapi_doc,
  "Provides the functional API for timing callables with arguments."
  "\n\n"
  "Inspired by ``timeit`` but times callables with arguments without using\n"
  "executable string statements, allowing more efficient timing of different\n"
  "callables when they share the same, possibly expensive, setup."
  "\n\n"
  ".. codeauthor:: Derek Huang <djh458@stern.nyu.edu>"
);
// _timeapi module definition
static struct PyModuleDef _timeapi_module = {
  PyModuleDef_HEAD_INIT,
  .m_name = "_timeapi",
  .m_doc = _timeapi_doc,
  .m_size = -1,
  .m_methods = _timeapi_methods
};

// module init function
PyMODINIT_FUNC
PyInit__timeapi(void) {
  // try to import NumPy array API. on error, returns NULL
  import_array();
  // try to import _timeresult, _timeunit C APIS. on error, returns NULL
  import__timeresult();
  import__timeunit();
  // on successful array import, we just create module. NULL on error.
  return PyModule_Create(&_timeapi_module);
}
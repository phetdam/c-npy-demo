/**
 * @file functimer.c
 * @brief C extension module that provides a callable API for the timing of
 *     Python functions. The implementation in C means that there is less
 *     measurement error introduced by the slow execution speed of Python
 *     loops (see the implementation of the `timeit` module).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <float.h>
#include <stdbool.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// module name and docstring
#define MODULE_NAME "functimer"
PyDoc_STRVAR(
  module_doc,
  "An internal C extension function timing module."
  "\n\n"
  "Inspired by timeit but times callables with arguments without using\n"
  "executable string statements. This way, timings of several different\n"
  "callables avoid multiple setup statement calls, which is wasteful."
  "\n\n"
  "Times obtained by default using time.perf_counter. Functional API only."
  "\n\n"
  ".. codeauthor:: Derek Huang <djh458@stern.nyu.edu>"
);

// definition for the TimeitResult struct
typedef struct {
  PyObject_HEAD
  // best per-loop runtime of the tested callable in units of unit
  double best;
  // string unit to use in the brief (report similar to timeit.main output).
  // either nsec, usec, msec, or sec, like in timeit.
  char const *unit;
  // number of loops the callable was run during each trial
  Py_ssize_t number;
  // number of trials that were run
  Py_ssize_t repeat;
  // read-only ndarrays of per-loop runtimes, total runtimes for each trial.
  // loop_times is a cached property and will be created only upon access.
  PyObject *loop_times;
  PyObject *times;
  // precision to use when displaying best in brief. defaults to 1. in the
  // __new__ method precision is capped at 20.
  int precision;
  // cached property. Python string with output similar to timeit.main output
  PyObject *brief;
} TimeitResult;

// TimeitResult class name
#define TimeitResult_name "TimeitResult"
// maximum precision value that may be passed to the TimeitResult constructor
#define TimeitResult_MAX_PRECISION 20
// list of valid values that unit can take. used to initialize
// TimeitResult_units and TimeitResult_UNITS_STR
#define TimeitResult_UNITS "nsec", "usec", "msec", "sec"
// valid values used to initialize TimeitResult_unit_bases. all values are
// interpreted as doubles (TimeitResult_unit_bases is double const array)
#define TimeitResult_UNIT_BASES 1e9, 1e6, 1e3, 1
/**
 * stringify combines varargs into a string, i.e. a, b, c -> "a, b, c", while
 * xstringify allows varargs to be macro expanded before stringification.
 * TimeitResult_UNITS_STR therefore simply escapes the quotes in
 * TimeitResult_UNITS and double quotes the whole thing.
 */
#define stringify(...) #__VA_ARGS__
#define xstringify(...) stringify(__VA_ARGS__)
#define TimeitResult_UNITS_STR xstringify(TimeitResult_UNITS)
// NULL-terminated array of strings indicating values unit can take
static char const * const TimeitResult_units[] = {TimeitResult_UNITS, NULL};
/**
 * bases corresponding to TimeitResult_units. each ith value corresponds to
 * the number a time in seconds should be multipled by to get the times in the
 * units given by the ith entry in TimeitResult_units.
 */
static double const TimeitResult_unit_bases[] = {TimeitResult_UNIT_BASES, 0};

/**
 * Return 1 if `unit` matches a value in `TimeitResult_units` else 0.
 * 
 * @param unit `char const *`, must be `NULL`-terminated
 * @returns 1 if valid unit in `TimeitResult_units, 0 otherwise.
 */
static int
validate_time_unit(const char *unit)
{
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
 * This is when `unit` is not passed to `timeit_enh` and will choose
 * the largest unit of time such that `fabs(best) >= 1`.
 * 
 * @param best Time in units of seconds
 * @param conv_p Memory location to write `best` converted to units returned
 *     by the function. If `NULL`, no writing is done.
 * @returns `char const *` to a value in `TimeitResult_units`.
 */
static const char *
autoselect_time_unit(const double best, double * const conv_p)
{
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
 * @note This is called when `Py_[X]DECREF` is called on a `TimeitResult *`.
 * 
 * @param self `TimeitResult *` current instance
 */
static void
TimeitResult_dealloc(TimeitResult *self)
{
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

// TimeitResult_new argument names (must be NULL-terminated)
static const char *TimeitResult_new_argnames[] = {
  "best", "unit", "number", "repeat", "times", "precision", NULL
};
/**
 * Custom `__new__` implementation for `TimeitResult` class.
 * 
 * Since the `TimeitResult` class is intended to be immutable, there is no
 * custom initialization function (C analogue to `__init__`), so all necessary
 * initialization is performed here (C analogue to `__new__`).
 * 
 * @note On error, note that we only `Py_DECREF` the `TimeitResult *` as the
 *     macro will call `TimeitResult_dealloc`, which will call `Py_[X]DECREF`
 *     as needed on the appropriate struct members.
 * 
 * @param type `PyTypeObject *` type object for the `TimeitResult` class
 * @param args `PyObject *` positional args tuple
 * @param kwargs `PyObject *` keyword args dict, may be NULL
 * @returns `PyObject *` new instance (new reference) of the `TimeitResult`
 *     struct or `NULL` if an error occurred + sets error indicator.
 */
static PyObject *
TimeitResult_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  // new instance of the TimeitResult allocated by tp_alloc slot. NULL on error
  TimeitResult *self = (TimeitResult *) type->tp_alloc(type, 0);
  if (self == NULL) {
    return NULL;
  }
  // set pointers to NULL to be overwritten later. self->loop_times and
  // self->brief also need to be NULL so that dealloc works correctly.
  self->unit = NULL;
  self->times = self->loop_times = self->brief = NULL;
  // set default value for self->precision
  self->precision = 1;
  /**
   * parse args and kwargs. pass field addresses to PyArg_ParseTupleAndKeywords.
   * on error, need to Py_DECREF self, which is a new reference. we also first
   * need to set self->times to NULL in the case that self->times is correctly
   * parsed but self->precision is not, as then the dealloc function will
   * Py_DECREF a borrowed reference, which of course is dangerous.
   */
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "dsnnO|i", (char **) TimeitResult_new_argnames,
      &(self->best), &(self->unit), &(self->number), &(self->repeat),
      &PyTuple_Type, &(self->times), &(self->precision)
    )
  ) {
    self->times = NULL;
    goto except;
  }
  /**
   * check that unit is one of several accepted values recorded in
   * TimeitResult_units. if not, set error indicator, Py_DECREF self. note that
   * we don't Py_DECREF self->times: tp_dealloc does that for us.
   */
  if (!validate_time_unit(self->unit)) {
    PyErr_SetString(
      PyExc_ValueError, "unit must be one of [" TimeitResult_UNITS_STR "]"
    );
    goto except;
  }
  /**
   * check that number, precision, repeat are positive. we don't check if best
   * is positive; maybe a weird "negative timer" was passed. on error, we have
   * to Py_DECREF times, self, which is a new reference.
   */
  if (self->number < 1) {
    PyErr_SetString(PyExc_ValueError, "number must be positive");
    goto except;
  }
  if (self->repeat < 1) {
    PyErr_SetString(PyExc_ValueError, "repeat must be positive");
    goto except;
  }
  if (self->precision < 1) {
    PyErr_SetString(PyExc_ValueError, "precision must be positive");
    goto except;
  }
  // cap precision at TimeitResult_MAX_PRECISION. no human needs more precision
  // than the value given by TimeitResult_MAX_PRECISION.
  if (self->precision > TimeitResult_MAX_PRECISION) {
    PyErr_Format(
      PyExc_ValueError, "precision is capped at %d", TimeitResult_MAX_PRECISION
    );
    goto except;
  }
  // now we handle self->times, which holds a borrowed ref. convert to read-only
  // ndarray and force a copy (original is probably writable). NULL on error.
  self->times = PyArray_FROM_OTF(
    self->times, NPY_DOUBLE, NPY_ARRAY_CARRAY_RO | NPY_ARRAY_ENSURECOPY
  );
  if (self->times == NULL) {
    goto except;
  }
  // make sure that self->times is 1D and has size equal to self->repeat. no
  // need to manually Py_DECREF self->times since dealloc takes care of that.
  if (PyArray_NDIM((PyArrayObject *) self->times) != 1) {
    PyErr_SetString(PyExc_ValueError, "times must be 1D");
    goto except;
  }
  // typically npy_intp is the same size as Py_ssize_t and is long int
  if (PyArray_SIZE((PyArrayObject *) self->times) != (npy_intp) self->repeat) {
    PyErr_SetString(PyExc_ValueError, "times.size must equal repeat");
    goto except;
  }
  // all checks are complete so return self
  return (PyObject *) self;
// clean up self and return NULL on exception
except:
  Py_DECREF(self);
  return NULL;
}

/**
 * Custom getter for `TimeitResult.loop_times`. Acts like cached `@property`.
 * 
 * @param self `TimeitResult *` current instance
 * @param closure `void *` (ignored)
 * @returns New reference to read-only `PyArrayObject *` tuple of trial times
 *     divided by number of loops per trial, the value given by `self->number`.
 */
static PyObject *
TimeitResult_getloop_times(TimeitResult *self, void *closure)
{
  // if self->loop_times is NULL, it has not been accessed before, so we have
  // to create a new read-only ndarray holding the per-loop times.
  if (self->loop_times == NULL) {
    // create new ndarray, type NPY_DOUBLE, C major layout. use dims of
    // self->times so we don't have to make new dims array. NULL on error.
    self->loop_times = PyArray_SimpleNew(
      1, PyArray_DIMS((PyArrayObject *) self->times), NPY_DOUBLE
    );
    if (self->loop_times == NULL) {
      return NULL;
    }
    // get size of times and pointers to data of loop_times, times
    npy_intp n_times = PyArray_SIZE((PyArrayObject *) self->times);
    double *times_data = (double *) PyArray_DATA(
      (PyArrayObject *) self->times
    );
    double *loop_times_data = (double *) PyArray_DATA(
      (PyArrayObject *) self->loop_times
    );
    // compute times_data[i] / self->number and write to loop_times_data[i]
    for (npy_intp i = 0; i < n_times; i++) {
      loop_times_data[i] = times_data[i] / (double) self->number;
    }
    // make loop_times read-only by disabling the NPY_ARRAY_WRITEABLE flags
    PyArray_CLEARFLAGS((PyArrayObject *) self->loop_times, NPY_ARRAY_WRITEABLE);
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
 * @returns New `PyObject *` Python Unicode object summary similar to output
 *     from `timeit.main` printed when `timeit` is run using `python3 -m`.
 */
static PyObject *
TimeitResult_getbrief(TimeitResult *self, void *closure)
{
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
    /**
     * get new reference to formatted string. use %R to use result of
     * PyObject_Repr on best_round in the formatted string. note that if
     * number == 1, then we write "loop" instead of "loops"
     */
    self->brief = PyUnicode_FromFormat(
      "%zd loop%s, best of %zd: %R %s per loop", self->number,
      (self->number == 1) ? "" : "s", self->repeat, best_round, self->unit
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
static PyObject *
TimeitResult_repr(TimeitResult *self)
{
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
    TimeitResult_name "(best=%R, unit='%s', number=%zd, repeat=%zd, times=%R, "
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

// standard members for TimeitResult, all read-only. doc in TimeitResult_doc.
static PyMemberDef TimeitResult_members[] = {
  {"best", T_DOUBLE, offsetof(TimeitResult, best), READONLY, NULL},
  {"unit", T_STRING, offsetof(TimeitResult, unit), READONLY, NULL},
  {"number", T_PYSSIZET, offsetof(TimeitResult, number), READONLY, NULL},
  {"repeat", T_PYSSIZET, offsetof(TimeitResult, repeat), READONLY, NULL},
  {"times", T_OBJECT_EX, offsetof(TimeitResult, times), READONLY, NULL},
  {"precision", T_INT, offsetof(TimeitResult, precision), READONLY, NULL},
  // required sentinel, at least name must be NULL
  {NULL, 0, 0, 0, NULL}
};

// TimeitResult docstrings for cached properties
PyDoc_STRVAR(
  TimeitResult_brief_doc,
  "A short string formatted similarly to that of timeit.main."
  "\n\n"
  "We can better describe this cached property by example. Suppose that\n"
  "calling ``repr`` on a ``TimeitResult`` instance yields"
  "\n\n"
  ".. code:: python3"
  "\n\n"
  "   TimeitResult(best=88.0, unit='usec', number=10000, repeat=5,\n"
  "       times=array([0.88, 1.02, 1.04, 1.024, 1]), precision=1)"
  "\n\n"
  "Note we manually wrapped the output here. Accessing ``brief`` yields"
  "\n\n"
  ".. code:: text"
  "\n\n"
  "   10000 loops, best of 5: 88.0 usec per loop"
  "\n\n"
  "Note that ``brief`` is a cached property computed upon first access,\n"
  "yielding new references on subsequent accesses."
  "\n\n"
  "Returns\n"
  "-------\n"
  "str"
);
PyDoc_STRVAR(
  TimeitResult_loop_times_doc,
  "The unweighted average times taken per loop, per trial, in seconds."
  "\n\n"
  "Like ``brief``, this is a cached property computed upon first access,\n"
  "yielding new references on subsequent accesses. The returned numpy.ndarray\n"
  "will be aligned, read-only, with type ``NPY_DOUBLE``."
  "\n\n"
  "Returns\n"
  "-------\n"
  "numpy.ndarray"
);
// getters for TimeitResult.brief and TimeitResult.loop_times. documentation
// for these cached properties are in TimeitResult_doc.
static PyGetSetDef TimeitResult_getters[] = {
  {
    "brief", (getter) TimeitResult_getbrief, NULL,
    TimeitResult_brief_doc, NULL
  },
  {
    "loop_times", (getter) TimeitResult_getloop_times, NULL,
    TimeitResult_loop_times_doc, NULL
  },
  // sentinel required; name must be NULL
  {NULL, NULL, NULL, NULL, NULL}
};

PyDoc_STRVAR(
  TimeitResult_doc,
  "An immutable type for holding timing results from functimer.timeit_enh."
  "\n\n"
  "All attributes are read-only. The ``loop_times`` and ``brief`` attributes\n"
  "are cached properties computed on demand when they are first accessed."
  "\n\n"
  "Parameters\n"
  "----------\n"
  "\n\n"
  "Attributes\n"
  "----------\n"
  "best : float\n"
  "    The best average function execution time in units of ``unit``.\n"
  "unit : {" TimeitResult_UNITS_STR "}\n"
  "    The unit of time that ``best`` is displayed in.\n"
  "number : int\n"
  "    The number of times the function is called in a single timing trial.\n"
  "repeat : int\n"
  "    The total number of timing trials, i.e. number of repeated trials.\n"
  "times : numpy.ndarray\n"
  "    The total execution times in seconds for each timing trial, shape\n"
  "    (repeat,). ``times`` is read-only and has type ``NPY_DOUBLE``.\n"
  "precision : int\n"
  "    The number of decimal places used to display ``best`` in ``brief``.\n"
  "brief : str\n"
  "loop_times : numpy.ndarray"
);
// type object for the TimeitResult type
static PyTypeObject TimeitResult_type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  // full type name is npapibench.MODULE_NAME.TimeitResult
  .tp_name = "npapibench." MODULE_NAME "." TimeitResult_name,
  // docstring and size for the TimeitResult type
  .tp_doc = TimeitResult_doc,
  .tp_basicsize = sizeof(TimeitResult),
  // not a variable-size object, so set to 0
  .tp_itemsize = 0,
  // omit Py_TPFLAGS_BASETYPE as this class is final
  .tp_flags = Py_TPFLAGS_DEFAULT,
  // custom __new__ function; no __init__ implementation for reinitialization
  .tp_new = TimeitResult_new,
  // custom destructor
  .tp_dealloc = (destructor) TimeitResult_dealloc,
  // standard class members; all are read-only
  .tp_members = TimeitResult_members,
  // getters for the brief and loop_times cached properties
  .tp_getset = TimeitResult_getters,
  // TimeitResult __repr__ method
  .tp_repr = (reprfunc) TimeitResult_repr
};

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
      args, kwargs, "O|OO$On", (char **) timeit_once_argnames,
      &func, &func_args, &func_kwargs, &timer, &number
    )
  ) {
    return NULL;
  }
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
  // check that func_args is tuple and that func_kwargs is a dict (or is NULL).
  // need to also Py_DECREF func_args to clean up the garbage
  if (!PyTuple_CheckExact(func_args)) {
    PyErr_SetString(PyExc_TypeError, "args must be a tuple");
    goto except_func_args;
  }
  if ((func_kwargs != NULL) && !PyDict_CheckExact(func_kwargs)) {
    PyErr_SetString(PyExc_TypeError, "kwargs must be a dict");
    goto except_func_args;
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
      PyExc_TypeError, "timer must return a float starting value"
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
    PyErr_SetString(PyExc_TypeError, "timer must return a float ending value");
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
      args, kwargs, "O|OO$O", (char **) autorange_argnames,
      &func, &func_args, &func_kwargs, &timer
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
  // PyLongObject * that will be used to hold the loop count in Python, the
  // returned time from timeit_once, timeit_time temp variable, error ref
  PyObject *number_, *timeit_time, *timeit_time_temp, *err_type;
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
      // set time_total to 0 to initialize
      time_total = 0;
      // add number_ to kwargs. Py_DECREF kwargs, number_ on failure
      if (PyDict_SetItemString(kwargs, "number", number_) < 0) {
        goto except_number_;
      }
      // save the returned time from timeit_once. the self, args, kwargs refs
      // are all borrowed so no need to Py_INCREF them. NULL on error.
      timeit_time = timeit_once(self, args, kwargs);
      if (timeit_time == NULL) {
        goto except_number_;
      }
      // convert timeit_time to Python float and Py_DECREF timeit_time_temp
      timeit_time_temp = timeit_time;
      timeit_time = PyNumber_Float(timeit_time);
      Py_DECREF(timeit_time_temp);
      // on error, exit. Py_DECREF kwargs and number_
      if (timeit_time == NULL) {
        goto except_number_;
      }
      // attempt to get time_total from timeit_time (Py_DECREF'd when done)
      time_total = PyFloat_AsDouble(timeit_time);
      Py_DECREF(timeit_time);
      // check if error occurred (borrowed ref)
      err_type = PyErr_Occurred();
      // if not NULL, then exit. error indicator already set. do Py_DECREFs
      if (err_type != NULL) {
        goto except_number_;
      }
      // done with number_ so Py_DECREF it
      Py_DECREF(number_);
      // computation of time_total complete. if time_total >= 0.2 s, Py_DECREF
      // kwargs and return Python int from number (NULL on failure)
      if (time_total >= 0.2) {
        Py_DECREF(kwargs);
        return PyLong_FromSsize_t(number);
      }
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
  // done with kwargs; Py_DECREF it number_, timeit_time alread Py_DECREF'd
  Py_DECREF(kwargs);
  // return Python int from number. NULL returned on failure
  return PyLong_FromSsize_t(number);
// clean up number_ on exception
except_number_:
  Py_DECREF(number_);
// clean up kwargs on exception
except_kwargs:
  Py_DECREF(kwargs);
  return NULL;
}

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
      args, kwargs, "O|OO$Onn", (char **) timeit_repeat_argnames,
      &func, &func_args, &func_kwargs, &timer, &number, &repeat
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
  // check if repeat was passed as a named arg. if so, we remove it from kwargs
  // so we can directly pass args, kwargs into timeit_once
  if (kwargs != NULL) {
    // Python string for "repeat", NULL on error
    PyObject *repeat_obj = PyUnicode_FromString("repeat");
    if (repeat_obj == NULL) {
      return NULL;
    }
    // get return value from PyDict_Contains and Py_DECREF unneeded repeat_obj
    int has_repeat = PyDict_Contains(kwargs, repeat_obj);
    Py_DECREF(repeat_obj);
    // if error, has_repeat < 0. return NULL
    if (has_repeat < 0) {
      return NULL;
    }
    // if repeat is in kwargs, then remove it from kwargs
    if (has_repeat) {
      // if failed, PyDict_DelItemString returns -1. return NULL
      if (PyDict_DelItemString(kwargs, "repeat") < 0) {
        return NULL;
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
    return NULL;
  }
  double *func_times_data = (double *) PyArray_DATA(func_times);
  // write the time result for each trial into func_times
  for (npy_intp i = 0; i < (npy_intp) repeat; i++) {
    // get time result from timeit_once, a PyFloatObject *. NULL on error
    PyObject *func_time = timeit_once(self, args, kwargs);
    if (func_time == NULL) {
      goto except;
    }
    // else write double value from func_time to func_times. use PyErr_Occurred
    // to check if PyFloat_AsDouble returned error. Py_DECREF in all cases.
    func_times_data[i] = PyFloat_AsDouble(func_time);
    if (PyErr_Occurred()) {
      Py_DECREF(func_time);
      goto except;
    }
    Py_DECREF(func_time);
  }
  // return the ndarray of times returned from timeit_once
  return (PyObject *) func_times;
// clean up on error
except:
  Py_DECREF(func_times);
  return NULL;
}

PyDoc_STRVAR(
  timeit_enh_doc,
  "timeit_enh(func, args=None, kwargs=None, *, timer=None, number=None, "
  "repeat=5, unit=None, precision=1)"
  "\n--\n\n"
  "A callable, approximate C implementation of ``timeit.main``. Returns a\n"
  "``TimeitResult`` instance with timing statistics whose attribute ``brief``\n"
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
  "unit : {" TimeitResult_UNITS_STR "}, default=None\n"
  "    Units to display the per-loop time stored in the ``brief`` attribute\n"
  "    of the returned ``TimeitResult`` with. If not specified, determined\n"
  "    by an internal function. Accepts the same values as ``timeit.main``.\n"
  "precision : int, default=1\n"
  "    Number of decimal places to display the per-loop time stored in the\n"
  "    ``brief`` attribute of the returned ``TimeitResult`` with. The\n"
  "    maximum allowed precision is "
  xstringify(TimeitResult_MAX_PRECISION) ", which should be plenty."
  "\n\n"
  "Returns\n"
  "-------\n"
  "TimeitResult\n"
);
// argument names for timeit_enh
static const char *timeit_enh_argnames[] = {
  "func", "args", "kwargs", "timer", "number", "repeat", "unit", "precision",
  NULL
};
/**
 * Operates in a similar manner to `timeit.main` but returns a `TimeitResult`.
 * 
 * @param args tuple of positional arguments
 * @param kwargs dict of named arguments, may be `NULL`
 * @returns New reference to `TimeitResult *`, `NULL` with exception on error
 */
PyObject *
timeit_enh(PyObject *self, PyObject *args, PyObject *kwargs)
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
  // parse args and kwargs. we defer checking of func, func_args, func_kwargs
  // to timeit_once and so must check number, repeat, unit, precision.
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|OO$Onnsi", (char **) timeit_enh_argnames, &func,
      &func_args, &func_kwargs, &timer, &number, &repeat, &unit, &precision
    )
  ) {
    return NULL;
  }
  // number must be positive (unless PY_SSIZE_T_MIN), repeat must be positive
  if ((number != PY_SSIZE_T_MIN) && (number < 1)) {
    PyErr_SetString(PyExc_ValueError, "number must be positive");
    return NULL;
  }
  if (repeat < 1) {
    PyErr_SetString(PyExc_ValueError, "repeat must be positive");
    return NULL;
  }
  // unit must be valid. if NULL, it will be decided with autoselect_time_unit
  if ((unit != NULL) && !validate_time_unit(unit)) {
    PyErr_SetString(
      PyExc_ValueError, "unit must be one of [" TimeitResult_UNITS_STR "]"
    );
    return NULL;
  }
  // precision must be positive and <= TimeitResult_MAX_PRECISION
  if (precision < 1) {
    PyErr_SetString(PyExc_ValueError, "precision must be positive");
    return NULL;
  }
  if (precision > TimeitResult_MAX_PRECISION) {
    PyErr_Format(
      PyExc_ValueError, "precision is capped at %d", TimeitResult_MAX_PRECISION
    );
    return NULL;
  }
  // if precision >= floor(TimeitResult_MAX_PRECISION / 2), print warning.
  // warning can be turned into exception so check PyErr_WarnFormat return.
  if (precision >= (TimeitResult_MAX_PRECISION / 2)) {
    if (
      PyErr_WarnFormat(
        PyExc_UserWarning, 1, "value of precision is rather high (>= %d). "
        "consider passing a lower value for better brief readability.",
        TimeitResult_MAX_PRECISION / 2
      ) < 0
    ) {
      return NULL;
    }
  }
  /**
   * now that all the parameters have been checked, we need to delegate the
   * right arguments to the right functions. func, func_args, and func_kwargs
   * need to be put together into a tuple, while timer needs to be put in a new
   * dict (if not NULL). if number == PY_SSIZE_T_MIN, then we need to call
   * autorange to give a value for number; then a PyFloatObject * wrapper for
   * number will be added to the new kwargs dict. A PyLongObject * wrapper for
   * repeat is then added to the new kwargs dict before the timeit_repeat call.
   * 
   * first, count number of positional arguments going into new args tuple.
   * n_new_args incremented for each of non-NULL func_args, func_kwargs.
   */
  int n_new_args = 1;
  if (func_args != NULL) {
    n_new_args++;
  }
  if(func_kwargs != NULL) {
    n_new_args++;
  }
  // new args tuple (for autorange, repeat)
  PyObject *new_args = PyTuple_New(n_new_args);
  if (new_args == NULL) {
    return NULL;
  }
  // pass func reference to new_args. Py_INCREF obviously needed (ref stolen)
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
  // new kwargs dict (for autorange, repeat). NULL on error.
  PyObject *new_kwargs = PyDict_New();
  if (new_kwargs == NULL) {
    goto except_new_args;
  }
  // if timer is not NULL, add timer to new_kwargs (borrow ref), -1 on failure
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
   * if unit is NULL, call autoselect_time_unit to return char const * pointer
   * to a unit string in TimeitResult_units (no need to free) + overwrite best
   * with best in units of the returned unit, which is never NULL.
   */
  if (unit == NULL) {
    unit = autoselect_time_unit(best, &best);
  }
  // now we start creating Python objects from C values to pass to the
  // TimeitResult constructor. create new Python float from best.
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
  // create new tuple of arguments to be passed to TimeitResult.__new__
  PyObject *res_args = PyTuple_Pack(
    6, best_, unit_, number_, repeat_, func_times, precision_
  );
  if (res_args == NULL) {
    goto except_precision_;
  }
  // no Py_[X]DECREF of func, func_args, func_kwargs since refs were stolen.
  // we Py_DECREF everything else except res_args which holds final refs.
  Py_DECREF(new_args);
  Py_DECREF(new_kwargs);
  Py_DECREF(number_);
  Py_DECREF(repeat_);
  Py_DECREF(func_times);
  Py_DECREF(best_);
  Py_DECREF(unit_);
  Py_DECREF(precision_);
  // create new TimeitResult instance using res_args and Py_DECREF res_args
  PyObject *tir = TimeitResult_new(&TimeitResult_type, res_args, NULL);
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

// static array of module methods
static PyMethodDef functimer_methods[] = {
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
    "timeit_enh", (PyCFunction) timeit_enh,
    METH_VARARGS | METH_KEYWORDS, timeit_enh_doc
  },
  // sentinel required; needs to have at least one NULL in it
  {NULL, NULL, 0, NULL}
};

// static module definition struct
static struct PyModuleDef functimer_def = {
  PyModuleDef_HEAD_INIT,
  .m_name = MODULE_NAME,
  .m_doc = module_doc,
  .m_size = -1,
  .m_methods = functimer_methods
};

// module initialization function
PyMODINIT_FUNC
PyInit_functimer(void) {
  // try to import NumPy array API. on error, automatically returns NULL
  import_array();
  // check if type is ready. if error (return < 0), exception is set
  if (PyType_Ready(&TimeitResult_type) < 0) {
    return NULL;
  }
  // create the module. if NULL, clean up and return NULL
  PyObject *module = PyModule_Create(&functimer_def);
  if (module == NULL) {
    return NULL;
  }
  // add TimeitResult_MAX_PRECISION to module. -1 on error
  if (
    PyModule_AddIntConstant(
      module, "MAX_PRECISION", TimeitResult_MAX_PRECISION
    ) < 0
  ) {
    return NULL;
  }
  // add PyTypeObject * to module. PyModule_AddObject only steals a reference
  // on success, so on error (returns -1), must Py_DECREF &Timeitresult_type.
  Py_INCREF(&TimeitResult_type);
  if (
    PyModule_AddObject(
      module, TimeitResult_name, (PyObject *) &TimeitResult_type
    ) < 0
  ) {
    Py_DECREF(&TimeitResult_type);
    Py_DECREF(module);
    return NULL;
  }
  return module;
}
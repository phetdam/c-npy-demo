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

// module name and docstring
#define MODULE_NAME "functimer"
PyDoc_STRVAR(
  module_doc,
  "An internal C extension function timing module."
  "\n\n"
  "Inspired by :mod:`timeit` but times callables with arguments without using\n"
  "executable string statements. This way, timings of several different\n"
  "callables avoid multiple setup statement calls, which is wasteful."
  "\n\n"
  "Times obtained by default using time.perf_counter. Functional API only."
  "\n\n"
  ".. codeauthor:: Derek Huang <djh458@stern.nyu.edu>"
);

// definition for the TimeitResult struct. not subclassable.
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
  // tuples of per-loop runtimes, total runtimes for each trial. loop_times is
  // a cached property and will be created only upon access.
  PyObject *loop_times;
  PyObject *times;
  // precision to use when displaying best in brief. defaults to 1. in the
  // __new__ method precision will capped at 70.
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
/**
 * NULL-terminated array of strings indicating values unit can take. note the
 * conditional compilation: this is non-static when C_NPY_DEMO_DEBUG is defined
 * by passing -DC_NPY_DEMO_DEBUG to gcc during test runner compilation.
 */
#ifdef C_NPY_DEMO_DEBUG
#else
static
#endif /* C_NPY_DEMO_DEBUG */
char const * const TimeitResult_units[] = {TimeitResult_UNITS, NULL};
/**
 * bases corresponding to TimeitResult_units. each ith value corresponds to
 * the number a time in seconds should be multipled by to get the times in the
 * units given by the ith entry in TimeitResult_units.
 */
#ifdef C_NPY_DEMO_DEBUG
#else
static
#endif /* C_NPY_DEMO_DEBUG */
double const TimeitResult_unit_bases[] = {TimeitResult_UNIT_BASES, 0};

/**
 * Return 1 if `unit` matches a value in `TimeitResult_units` else 0.
 * 
 * @param unit `char const *`, must be `NULL`-terminated
 * @returns 1 if valid unit in `TimeitResult_units, 0 otherwise.
 */
#ifdef C_NPY_DEMO_DEBUG
int
#else
static int
#endif /* C_NPY_DEMO_DEBUG */
TimeitResult_validate_unit(char const *unit) {
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
#ifdef C_NPY_DEMO_DEBUG
char const *
#else
static char const *
#endif /* C_NPY_DEMO_DEBUG */
TimeitResult_autounit(double const best, double * const conv_p) {
  int i = 0;
  /**
   * loop through TimeitResult_unit_bases until we reach NULL or a unit such
   * that multiplying by the corresponding base b_i results in fabs(b_i * best)
   * to be less than 1. the final index is then decremented if i > 0.
   */
  while (TimeitResult_unit_bases[i] != 0) {
    if (fabs(TimeitResult_unit_bases[i] * best) < 1) {
      break;
    }
    i++;
  }
  if (i > 0) {
    i--;
  }
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
static void
TimeitResult_dealloc(TimeitResult *self) {
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

// need to decalre TimeitResult_type since it is used in TimeitResult_new. note
// TimeitResult_type is non-static if C_NPY_DEMO_DEBUG defined,
#ifdef C_NPY_DEMO_DEBUG
PyTypeObject
#else
static PyTypeObject
#endif /* C_NPY_DEMO_DEBUG S*/
TimeitResult_type;
// TimeitResult_new argument names (must be NULL-terminated)
static char *TimeitResult_new_argnames[] = {
  "best", "unit", "number", "repeat", "times", "precision", NULL
};
/**
 * Custom `__new__` implementation for `TimeitResult` class.
 * 
 * Includes `NULL` pointer checking for safer use from C. Since the
 * `TimeitResult` class is intended to be immutable, it does not implement a
 * custom initialization function (C analogue to `__init__`), so all necessary
 * initialization is performed here (C analogue to `__new__`).
 * 
 * @note This is also conditionally `static` since we use it directly in the
 *     extension module to create a new `TimeitResult` and so is unit tested.
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
#ifdef C_NPY_DEMO_DEBUG
PyObject *
#else
static PyObject *
#endif /* C_NPY_DEMO_DEBUG */
TimeitResult_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  // if type is NULL, set error indicator and return NULL
  if (type == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "missing PyTypeObject *type");
    return NULL;
  }
  // type must be the address of the TimeitResult type object
  if (type != &TimeitResult_type) {
    PyErr_SetString(
      PyExc_TypeError,
      "type must be the address of the TimeitResult PyTypeObject"
    );
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
      args, kwargs, "dsnnO!|i", TimeitResult_new_argnames,
      &(self->best), &(self->unit), &(self->number), &(self->repeat),
      &PyTuple_Type, &(self->times), &(self->precision)
    )
  ) {
    self->times = NULL;
    goto except;
  }
  // increase reference count to times since it is borrowed
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
  // cap precison at TimeitResult_MAX_PRECISION. no human needs more precision
  // than the value given by TimeitResult_MAX_PRECISION.
  if (self->precision > TimeitResult_MAX_PRECISION) {
    PyErr_Format(
      PyExc_ValueError, "precision is capped at %d", TimeitResult_MAX_PRECISION
    );
    goto except;
  }
  /**
   * len(times) must equal repeat. if not, set error and Py_DECREF self. we can
   * use PyTuple_GET_SIZE instead of PyTuple_Size since we already have
   * guaranteed that self->times is a tuple at this point.
   */
  if (PyTuple_GET_SIZE(self->times) != self->repeat) {
    PyErr_SetString(PyExc_ValueError, "len(times) must equal repeat");
    goto except;
  }
  // check that all the elements of self->times are int or float
  for (Py_ssize_t i = 0; i < self->repeat; i++) {
    // get borrowed reference to ith element of self->times. note that we use
    // PyTuple_GET_ITEM since we are guaranteed not to go out of bounds.
    PyObject *time_i = PyTuple_GET_ITEM(self->times, i);
    // if neither int nor float, set error indicator, Py_DECREF self
    if (!PyLong_CheckExact(time_i) && !PyFloat_CheckExact(time_i)) {
      PyErr_SetString(PyExc_TypeError, "times must contain only int or float");
      goto except;
    }
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
 * @returns `PyObject *` tuple of trial times divided by number of loops per
 *     trial, the value given by `self->number`.
 */
static PyObject *
TimeitResult_getloop_times(TimeitResult *self, void *closure) {
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
        Py_DECREF(self->loop_times);
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
static PyObject *
TimeitResult_getbrief(TimeitResult *self, void *closure) {
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
TimeitResult_repr(TimeitResult *self) {
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

// TimeitResult docstrings for read-only members
PyDoc_STRVAR(
  functimer_timeitresult_best_doc,
  "The best average execution time of the timed function in units specified\n"
  "by :attr:`TimeitResult.unit`."
);
PyDoc_STRVAR(
  functimer_timeitresult_unit_doc,
  "The time unit that :attr:`TimeitResult.best` is displayed in."
);
PyDoc_STRVAR(
  functimer_timeitresult_number_doc,
  "Number of times the function is called in a single timing trial."
);
PyDoc_STRVAR(
  functimer_timeitresult_repeat_doc,
  "The total number of timing trials."
);
PyDoc_STRVAR(
  functimer_timeitresult_times_doc,
  "A tuple of the total execution times in seconds for each timing trial."
);
PyDoc_STRVAR(
  functimer_timeitresult_precision_doc,
  "Number of decimal places used when displaying :attr:`TimeitResult.best`\n"
  "in :attr:`TimeitResult.brief`."
);
// standard members for TimeitResult, all read-only
static PyMemberDef TimeitResult_members[] = {
  {
    "best",T_DOUBLE, offsetof(TimeitResult, best), READONLY,
    functimer_timeitresult_best_doc
  },
  {
    "unit", T_STRING, offsetof(TimeitResult, unit), READONLY,
    functimer_timeitresult_unit_doc
  },
  {
    "number", T_PYSSIZET, offsetof(TimeitResult, number), READONLY,
    functimer_timeitresult_number_doc
  },
  {
    "repeat", T_PYSSIZET, offsetof(TimeitResult, repeat), READONLY,
    functimer_timeitresult_repeat_doc
  },
  {
    "times", T_OBJECT_EX, offsetof(TimeitResult, times), READONLY,
    functimer_timeitresult_times_doc
  },
  {
    "precision", T_INT, offsetof(TimeitResult, precision), READONLY,
    functimer_timeitresult_precision_doc
  },
  // required sentinel, at least name must be NULL
  {NULL, 0, 0, 0, NULL}
};

// TimeitResult docstrings for cached properties
PyDoc_STRVAR(
  functimer_timeitresult_getbrief_doc,
  "A short string formatted similarly to that of timeit.main."
  "\n\n"
  "For example, suppose that calling :func:`repr` on a :class:`TimeitResult`\n"
  "instance yields (manually wrapped for compactness)"
  "\n\n"
  ".. code:: python3"
  "\n\n"
  "   TimeitResult(best=88.0, unit='usec', number=10000, repeat=5,\n"
  "       times=(0.88, 1.02, 1.04, 1.024, 1), precision=1)"
  "\n\n"
  "Accessing the ``brief`` attribute [#]_ yields"
  "\n\n"
  ".. code:: text"
  "\n\n"
  "   10000 loops, best of 5: 88.0 usec per loop"
  "\n\n"
  ".. [#] It is more accurate to call ``brief`` a cached property, as it is\n"
  "   computed the first time it is accessed and simply yields new references\n"
  "   whenever it is repeatedly accessed."
);
PyDoc_STRVAR(
  functimer_timeitresult_getloop_times_doc,
  "The unweighted average time taken per loop, per trial, in units of seconds."
  "\n\n"
  "Like TimeitResult.brief, this is a cached property computed on the first\n"
  "access and yields new references on subsequent accesses."
);
// getters for TimeitResult.brief and TimeitResult.loop_times
static PyGetSetDef TimeitResult_getters[] = {
  {
    "brief", (getter) TimeitResult_getbrief, NULL,
    functimer_timeitresult_getbrief_doc, NULL
  },
  {
    "loop_times", (getter) TimeitResult_getloop_times, NULL,
    functimer_timeitresult_getloop_times_doc, NULL
  },
  // sentinel required; name must be NULL
  {NULL, NULL, NULL, NULL, NULL}
};

PyDoc_STRVAR(
  functimer_timeitresult_doc,
  "An immutable type for holding timing results from functimer.timeit_enh."
  "\n\n"
  "All attributes are read-only. The loop_times and brief attributes are\n"
  "cached properties computed on demand when they are first accessed."
);
// static PyTypeObject for the TimeitResult type. also conditionally compiled
// since we need it to test TimeitResult_new in the test runner.
#ifdef C_NPY_DEMO_DEBUG
PyTypeObject
#else
static PyTypeObject
#endif /* C_NPY_DEMO_DEBUG S*/
TimeitResult_type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  // full type name is c_npy_demo.MODULE_NAME.TimeitResult
  .tp_name = "c_npy_demo." MODULE_NAME "." TimeitResult_name,
  // docstring and size for the TimeitResult struct
  .tp_doc = functimer_timeitresult_doc,
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
  functimer_timeit_once_doc,
  "timeit_once(func, args=None, kwargs=None, *, timer=None, number=1000000)\n"
  "--\n\n"
  "Operates in the same way as :func:`timeit.timeit`, i.e. the same way as\n"
  ":meth:`timeit.Timer.timeit`. ``func`` will be executed with positional\n"
  "args ``args`` and keyword args ``kwargs`` ``number`` times and the total\n"
  "execution time will be returned. ``timer`` is the timing function and must\n"
  "return time in units of fractional seconds."
  "\n\n"
  ":param func: Callable to time\n"
  ":type func: callable\n"
  ":param args: Tuple of positional args to pass to ``func``\n"
  ":type args: tuple, optional\n"
  ":param kwargs: Dict of named arguments to pass to ``func``\n"
  ":type kwargs: dict, optional\n"
  ":param timer: Timer function, defaults to :func:`time.perf_counter` which\n"
  "    returns time in seconds. If specified, must return time in fractional\n"
  "    seconds and not take any arguments.\n"
  ":type timer: function, optional\n"
  ":param number: Number of times to call ``func``\n"
  ":type number: int, optional\n"
  ":returns: Time for ``func`` to be executed ``number`` times with args\n"
  "    ``args`` and kwargs ``kwargs``, in units of ``timer``.\n"
  ":rtype: float"
);
// argument names for timeit_once
static char *functimer_timeit_once_argnames[] = {
  "func", "args", "kwargs", "timer", "number", NULL
};
/**
 * Operates in a similar manner to `timeit.timeit`.
 * 
 * Docstring in `_modinit.c`.
 * 
 * @param args tuple of positional arguments
 * @param kwargs dict of named arguments
 * @returns `PyObject *` numeric value
 */
static PyObject *
functimer_timeit_once(PyObject *self, PyObject *args, PyObject *kwargs) {
  // callable, args, kwargs, timer function
  PyObject *func, *func_args, *func_kwargs, *timer;
  // if timer NULL after arg parsing, set to time.perf_counter
  func_args = func_kwargs = timer = NULL;
  // number of times to execute the callable with args and kwargs
  Py_ssize_t number = 1000000;
  // parse args and kwargs; sets appropriate exception so no need to check
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|OO$On", functimer_timeit_once_argnames,
      &func, &func_args, &func_kwargs, &timer, &number
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
  // starting, ending times recorded by timer function
  PyObject *start_time, *end_time;
  // get starting time from timer function. 
  start_time = PyObject_CallObject(timer, NULL);
  // if NULL, exception was raised. Py_DECREF and Py_XDECREF as needed
  if (start_time == NULL) {
    goto except_time_perf_counter;
  }
  // if not numeric, raised exception. Py_DECREF and Py_XDECREF as needed. note
  // we also need to Py_DECREF start_time since it's a new reference
  if (!PyNumber_Check(start_time)) {
    PyErr_SetString(PyExc_TypeError, "timer must return a numeric value");
    goto except_start_time;
  }
  // get new start time; time was lost checking if start_time is valid
  Py_DECREF(start_time);
  start_time = PyObject_CallObject(timer, NULL);
  // if NULL, exception was raised. Py_DECREF and Py_XDECREF as needed
  if (start_time == NULL) {
    goto except_time_perf_counter;
  }
  PyObject *func_res;
  // call function number times with func_args and func_kwargs
  for (Py_ssize_t i = 0; i < number; i++) {
    // call function and Py_XDECREF its result
    func_res = PyObject_Call(func, func_args, func_kwargs);
    Py_XDECREF(func_res);
    // if NULL is returned, an exception has been raised. Py_DECREF, Py_XDECREF
    if (func_res == NULL) {
      goto except_start_time;
    }
  }
  // get ending time from timer function
  end_time = PyObject_CallObject(timer, NULL);
  // if NULL, exception raised; Py_DECREF and Py_XDECREF as needed
  if (end_time == NULL) {
    goto except_start_time;
  }
  // if not numeric, raised exception. Py_DECREF and Py_XDECREF as needed; also
  // need to Py_DECREF end_time since we got a new reference for it
  if (!PyNumber_Check(end_time)) {
    PyErr_SetString(PyExc_TypeError, "timer must return a numeric value");
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
  functimer_autorange_doc,
  "autorange(func, args=None, kwargs=None, *, timer=None)\n"
  "--\n\n"
  "Determine number of times to call :func:`c_npy_demo.functimer.timeit_once."
  "\n\n"
  "Operates in the same way as :meth:`timeit.Timer.autorange`.\n"
  ":func:`autorange` calls :func:`timeit_once` 1, 2, 5, 10, 20, 50, etc.\n"
  "times until the total execution time is >= 0.2 seconds. The number of\n"
  "times :func:`timeit_once` is to be called is then returned."
  "\n\n"
  ":param func: Callable to time\n"
  ":type func: callable\n"
  ":param args: Tuple of positional args to pass to ``func``\n"
  ":type args: tuple, optional\n"
  ":param kwargs: Dict of named arguments to pass to ``func``\n"
  ":type kwargs: dict, optional\n"
  ":param timer: Timer function, defaults to :func:`time.perf_counter` which\n"
  "    returns time in seconds. If specified, must return time in fractional\n"
  "    seconds and not take any arguments.\n"
  ":type timer: function, optional\n"
  ":rtype: int"
);
// argument names for functimer_autorange
char *functimer_autorange_argnames[] = {
  "func", "args", "kwargs", "timer", NULL
};
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
static PyObject *
functimer_autorange(PyObject *self, PyObject *args, PyObject *kwargs) {
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
      args, kwargs, "O|OO$O", functimer_autorange_argnames,
      &func, &func_args, &func_kwargs, &timer
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
  // PyLongObject * that will be used to hold the loop count in Python
  PyObject *number_;
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
      // save the returned time from functimer_timeit_once. the self, args,
      // kwargs refs are all borrowed so no need to Py_INCREF them.
      PyObject *timeit_time = functimer_timeit_once(self, args, kwargs);
      // if NULL, return NULL. let functimer_timeit set the error indicator.
      // Py_DECREF kwargs and number_ (they are new references)
      if (timeit_time == NULL) {
        goto except_number_;
      }
      // convert timeit_time to Python float and Py_DECREF timeit_time_temp
      PyObject *timeit_time_temp = timeit_time;
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
      PyObject *err_type = PyErr_Occurred();
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
    multipler = multipler * 10;
  }
  // done with kwargs; Py_DECREF it
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
  functimer_repeat_doc,
  "repeat(func, args=None, kwargs=None, *, timer=None, number=1000000, "
  "repeat=5)\n"
  "--\n\n"
  "Operates in the same way as :func:`timeit.repeat`, i.e. the same was as\n"
  ":meth:`timeit.Timer.repeat`. :func:`timeit_once` will be executed\n"
  "``repeat`` times, with ``number`` the number of loops to run in each\n"
  ":func:`timeit_once` call."
  "\n\n"
  ":param func: Callable to time\n"
  ":type func: callable\n"
  ":param args: Tuple of positional args to pass to ``func``\n"
  ":type args: tuple, optional\n"
  ":param kwargs: Dict of named arguments to pass to ``func``\n"
  ":type kwargs: dict, optional\n"
  ":param timer: Timer function, defaults to :func:`time.perf_counter` which\n"
  "    returns time in seconds. If specified, must return time in fractional\n"
  "    seconds and not take any arguments.\n"
  ":type timer: function, optional\n"
  ":param number: Number of times to call ``func``\n"
  ":type number: int, optional\n"
  ":param repeat: Number of times to repeat the call to :func:`timeit_once`\n"
  ":param repeat: int, optional\n"
  ":returns: List of ``repeat`` times in fractional seconds taken for each\n"
  "    call to :func:`timeit_once`."
  ":rtype: list"
);
// argument names for functimer_repeat
static char *functimer_repeat_argnames[] = {
    "func", "args", "kwargs", "timer", "number", "repeat", NULL
};
/**
 * Operates in a similar manner to `timeit.Timer.repeat`.
 * 
 * Docstring in `_modinit.c`.
 * 
 * @param args tuple of positional arguments
 * @param kwargs dict of named arguments
 * @returns `PyListObject *` list of trial items cast to `PyObject *`
 */
static PyObject *
functimer_repeat(PyObject *self, PyObject *args, PyObject *kwargs) {
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
  // parse args and kwargs; sets appropriate exception so no need to check
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|OO$Onn", functimer_repeat_argnames,
      &func, &func_args, &func_kwargs, &timer, &number, &repeat
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

PyDoc_STRVAR(
  functimer_timeit_enh_doc,
  "timeit_enh(func, args=None, kwargs=None, *, timer=None, number=None, "
  "repeat=5, unit=None, precision=1)\n"
  "--\n\n"
  "A callable, C-implemented emulation of :func:`timeit.main`. Returns a\n"
  ":class:`TimeitResult` object with timing statistics whose attribute\n"
  ":attr:`TimeitResult.brief` provides the same exact output as\n"
  ":func:`timeit.main`."
  "\n\n"
  ":param func: Callable to time\n"
  ":type func: callable\n"
  ":param args: Tuple of positional args to pass to ``func``\n"
  ":type args: tuple, optional\n"
  ":param kwargs: Dict of named arguments to pass to ``func``\n"
  ":type kwargs: dict, optional\n"
  ":param timer: Timer function, defaults to :func:`time.perf_counter` which\n"
  "    returns time in seconds. If specified, must return time in fractional\n"
  "    seconds and not take any arguments.\n"
  ":type timer: function, optional\n"
  ":param number: Number of times to call ``func``. If not specified, this\n"
  "    is automatically determined by :func:`autorange` internally.\n"
  ":type number: int, optional\n"
  ":param repeat: Number of times to repeat the call to :func:`timeit_once`\n"
  ":param repeat: int, optional\n"
  ":param unit: Units to display :attr:`TimeitResult.brief` results in. If\n"
  "    specified, this is automatically determined internally. Accepts the\n"
  "    same values that :func:`timeit.main` accepts.\n"
  ":type unit: str, optional\n"
  ":param precision: Number of decimal places to display\n"
  "    :attr:`TimeitResult.brief` results in.\n"
  ":type precision: int, optional\n"
  ":rtype: :class:`~c_npy_demo.functimer.TimeitResult`"
);
// argument names for functimer_timeit_enh
static char *functimer_timeit_enh_argnames[] = {
  "func", "args", "kwargs", "timer", "number", "repeat", "unit", "precision",
  NULL
};
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
  // parse args and kwargs; sets appropriate exception automatically
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|OO$Onnsi", functimer_timeit_enh_argnames, &func,
      &func_args, &func_kwargs, &timer, &number, &repeat, &unit, &precision
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
    goto except_new_args;
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
      goto except_new_kwargs;
    }
    // attempt to convert number_ into Py_ssize_t
    number = PyLong_AsSsize_t(number_);
    // if number == -1, error (don't even need to check PyErr_Occurred). then
    // Py_DECREF new_args, new_kwargs, number_ (also new ref)
    if (number == -1) {
      goto except_number_;
    }
  }
  // if number_ is NULL, it wasn't initialized in if statement, so initialize
  if (number_ == NULL) {
    number_ = PyLong_FromSsize_t(number);
    // Py_DECREF new_args, new_kwargs on error
    if (number_ == NULL) {
      goto except_new_args;
    }
  }
  // add number_ to new_kwargs. Py_DECREF new_args, new_kwargs, number_ if err
  if (PyDict_SetItemString(new_kwargs, "number", number_) < 0) {
    goto except_number_;
  }
  // repeat as a PyLongObject * to be passed to new_kwargs
  PyObject *repeat_ = PyLong_FromSsize_t(repeat);
  // on error, Py_DECREF new_args, new_kwargs, number_
  if (repeat_ == NULL) {
    goto except_number_;
  }
  // add repeat_ to new_kwargs, Py_DECREF on error (include repeat_)
  if (PyDict_SetItemString(new_kwargs, "repeat", repeat_) < 0) {
    goto except_repeat_;
  }
  // call functimer_repeat with new_args, new_kwargs (borrowed refs) and get
  // times_list, the list of times in seconds for each of the repeat trials
  PyObject *times_list = functimer_repeat(self, new_args, new_kwargs);
  // if NULL, exception was set, so Py_DECREF as needed
  if (times_list == NULL) {
    goto except_repeat_;
  }
  // create tuple out of times_list and Py_DECREF times_list; no longer needed
  PyObject *times_tuple = PySequence_Tuple(times_list);
  Py_DECREF(times_list);
  // Py_DECREF as needed if error
  if (times_tuple == NULL) {
    goto except_repeat_;
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
      goto except_times_tuple;
    }
    // convert time_i_ to double and Py_DECREF (no longer needed)
    double time_i = PyFloat_AsDouble(time_i_);
    Py_DECREF(time_i_);
    // if error occurred, Py_DECREF
    if (PyErr_Occurred()) {
      goto except_times_tuple;
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
    goto except_times_tuple;
  }
  // create Python string from unit, Py_DECREF on error (include best_)
  PyObject *unit_ = PyUnicode_FromString(unit);
  if (unit_ == NULL) {
    goto except_best_;
  }
  // create Python int from precision, Py_DECREF on error (include unit_)
  PyObject *precision_ = PyLong_FromLong(precision);
  if (precision_ == NULL) {
    goto except_unit_;
  }
  // create new tuple of arguments to be passed to TimeitResult.__new__. note
  // that since we use PyTuple_Pack, we still need to Py_DECREF on error.
  //"best", "unit", "number", "repeat", "times", "precision", NULL
  PyObject *res_args = PyTuple_Pack(
    6, best_, unit_, number_, repeat_, times_tuple, precision_
  );
  if (res_args == NULL) {
    goto except_precision_;
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
  /**
   * create new TimeitResult instance using res_args. note that since the
   * references parsed inside TimeitResult_new are borrowed, we can't safely
   * Py_DECREF res_args and must leave one reference alive.
   */
  PyObject *tir = TimeitResult_new(&TimeitResult_type, res_args, NULL);
  // TimeitResult_new will set error indicator on error
  if (tir == NULL) {
    return NULL;
  }
  // return new reference
  return tir;
// clean up in order from last to first allocated on exception
except_precision_:
  Py_DECREF(precision_);
except_unit_:
  Py_DECREF(unit_);
except_best_:
  Py_DECREF(best_);
except_times_tuple:
  Py_DECREF(times_tuple);
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
    "timeit_once",
    // cast PyCFunctionWithKeywords to PyCFunction to silence compiler warning
    (PyCFunction) functimer_timeit_once,
    METH_VARARGS | METH_KEYWORDS,
    functimer_timeit_once_doc
  },
  {
    "repeat",
    (PyCFunction) functimer_repeat,
    METH_VARARGS | METH_KEYWORDS,
    functimer_repeat_doc
  },
  {
    "autorange",
    (PyCFunction) functimer_autorange,
    METH_VARARGS | METH_KEYWORDS,
    functimer_autorange_doc
  },
  {
    "timeit_enh",
    (PyCFunction) functimer_timeit_enh,
    METH_VARARGS | METH_KEYWORDS,
    functimer_timeit_enh_doc
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
PyMODINIT_FUNC PyInit_functimer(void) {
  // check if type is ready. if error (return < 0), exception is set
  if (PyType_Ready(&TimeitResult_type) < 0) {
    return NULL;
  }
  /**
   * now that type has been initialized, we can add more attributes to the
   * dict at TimeitResult_type.tp_dict. we add TimeitResult_MAX_PRECISION as
   * the MAX_PRECISION class attribute.
   */
  PyObject *max_precision = PyLong_FromLong((long) TimeitResult_MAX_PRECISION);
  // no need to Py_DECREF &TimeitResult_type
  if (max_precision == NULL) {
    return NULL;
  }
  // add max_precision to Timeitresult_Type.tp_dict as MAX_PRECISION. if
  // assignment fails, Py_DECREF max_precision
  if (
    PyDict_SetItemString(
      TimeitResult_type.tp_dict, "MAX_PRECISION", max_precision
    ) < 0
  ) {
    Py_DECREF(max_precision);
    return NULL;
  }
  // create the module. if NULL, return
  PyObject *module = PyModule_Create(&functimer_def);
  if (module == NULL) {
    return NULL;
  }
  /**
   * add PyTypeObject * to module. need to Py_INCREF &TimeitResult_type since
   * it starts with zero references. PyModule_AddObject only steals a reference
   * on success, so on error (returns -1), must Py_DECREF &Timeitresult_type.
   * also need to Py_DECREF module, which is a new reference.
   */
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
  // return module pointer
  return module;
}
/**
 * @file _timeresult.c
 * @brief Provides the `TimeResult` returned by `functimer` timing functions.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <float.h>
#include <stdbool.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define TIMERESULT_MODULE
#include "timeresult.h"
#include "timeunit.h"

// definition for the TimeResult struct
typedef struct {
  PyObject_HEAD
  // best per-loop runtime of the tested callable in units of unit
  double best;
  // string unit to use in the brief (report similar to timeit.main output).
  // either nsec, usec, msec, or sec, like in timeit.
  const char *unit;
  // number of times the callable was run each trial, number of total trials
  Py_ssize_t number;
  Py_ssize_t repeat;
  // read-only ndarrays of per-loop runtimes, total runtimes for each trial.
  // loop_times is a cached property and will be created only upon access.
  PyObject *loop_times;
  PyObject *times;
  // precision to use when displaying best in brief
  int precision;
  // cached property. Python string with output similar to timeit.main output
  PyObject *brief;
} TimeResult;

/**
 * Custom destructor for the `TimeResult` class.
 * 
 * @note This is called when `Py_[X]DECREF` is called on a `TimeResult *`.
 * 
 * @param self `TimeResult *` current instance
 */
static void
TimeResult_dealloc(TimeResult *self)
{
  /**
   * times, loop_times, brief might be NULL, so we need Py_XDECREF. times can be
   * NULL if TimeResult_new fails while loop_times, brief may be NULL if they
   * are never accessed by the user as attributes.
   */
  Py_XDECREF(self->times);
  Py_XDECREF(self->loop_times);
  Py_XDECREF(self->brief);
  // free the struct using the default function set to tp_free
  Py_TYPE(self)->tp_free((void *) self);
}

// TimeResult_new argument names (must be NULL-terminated)
static const char *TimeResult_new_argnames[] = {
  "best", "unit", "number", "repeat", "times", "precision", NULL
};
/**
 * Custom `__new__` implementation for `TimeResult` class.
 * 
 * Since the `TimeResult` class is intended to be immutable, there is no
 * custom initialization function (C analogue to `__init__`), so all necessary
 * initialization is performed here (C analogue to `__new__`).
 * 
 * @note On error, note that we only `Py_DECREF` the `TimeResult *` as the
 *     macro will call `TimeResult_dealloc`, which will call `Py_[X]DECREF`
 *     as needed on the appropriate struct members.
 * 
 * @param type `PyTypeObject *` type object for the `TimeResult` class
 * @param args `PyObject *` positional args tuple
 * @param kwargs `PyObject *` keyword args dict, may be NULL
 * @returns `PyObject *` new instance (new reference) of the `TimeResult`
 *     struct or `NULL` if an error occurred + sets error indicator.
 */
static PyObject *
TimeResult_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  // new instance of the TimeResult allocated by tp_alloc slot. NULL on error
  TimeResult *self = (TimeResult *) type->tp_alloc(type, 0);
  if (self == NULL) {
    return NULL;
  }
  // set pointers to NULL to be overwritten later. self->loop_times and
  // self->brief also need to be NULL so that dealloc works correctly.
  self->unit = NULL;
  self->times = self->loop_times = self->brief = NULL;
  // set default value for self->precision
  self->precision = 1;
  // parse args and kwargs
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "dsnnO|i", (char **) TimeResult_new_argnames,
      &(self->best), &(self->unit), &(self->number),
      &(self->repeat), &(self->times), &(self->precision)
    )
  ) {
    // set self->times to NULL in case this is set but error occurs after. this
    // prevents Py_DECREF of NULL or borrowed ref during TimeResult_dealloc.
    self->times = NULL;
    goto except;
  }
  // handle self->times, which holds a borrowed ref. convert to read-only
  // ndarray and force a copy (original is probably writable). NULL on error.
  self->times = PyArray_FROM_OTF(
    self->times, NPY_DOUBLE, NPY_ARRAY_CARRAY_RO | NPY_ARRAY_ENSURECOPY
  );
  if (self->times == NULL) {
    // calls TimeResult_dealloc which will Py_DECREF self->times
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
  // check that unit is one of several valid units in Py__timeunit_UNITS
  if (!Py__timeunit_validate_unit(self->unit)) {
    PyErr_SetString(
      PyExc_ValueError, "unit must be one of [" Py__timeunit_UNITS_STR "]"
    );
    goto except;
  }
  // check that number, precision, repeat are positive. we don't check if best
  // is positive; maybe a weird "negative timer" was passed
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
  // cap precision at Py__timeunit_MAX_PRECISION. no human needs more precision
  // than the value given by Py__timeunit_MAX_PRECISION.
  if (self->precision > Py__timeunit_MAX_PRECISION) {
    PyErr_Format(
      PyExc_ValueError, "precision is capped at %d", Py__timeunit_MAX_PRECISION
    );
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
 * Custom getter for `TimeResult.loop_times`. Acts like cached `@property`.
 * 
 * @param self `TimeResult *` current instance
 * @param closure `void *` (ignored)
 * @returns New reference to read-only `PyArrayObject *` tuple of trial times
 *     divided by number of loops per trial, the value given by `self->number`.
 */
static PyObject *
TimeResult_getloop_times(TimeResult *self, void *closure)
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
 * Custom getter for `TimeResult.brief`. Acts like cached `@property`.
 * 
 * @param self `TimeResult *` current instance
 * @param closure `void *` (ignored)
 * @returns New `PyObject *` Python Unicode object summary similar to output
 *     from `timeit.main` printed when `timeit` is run using `python3 -m`.
 */
static PyObject *
TimeResult_getbrief(TimeResult *self, void *closure)
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
  // Py_INCREF self->brief + return. see TimeResult_getloop_times comment.
  Py_INCREF(self->brief);
  return self->brief;
}

/**
 * Custom `__repr__` implementation for `TimeResult`.
 * 
 * @param self `TimeResult *` current instance
 * @returns `PyObject *` Python Unicode object representation for `self`. This
 *     is a new reference and is the C parallel to how `__repr__` would be
 *     implemented in pure Python.
 */
static PyObject *
TimeResult_repr(TimeResult *self)
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
    "TimeResult(best=%R, unit='%s', number=%zd, repeat=%zd, times=%R, "
    "precision=%d)", py_best, self->unit, self->number,
    self->repeat, self->times, self->precision
  );
  if (repr_str == NULL) {
    Py_DECREF(py_best);
    return NULL;
  }
  // Py_DECREF py_best and return result
  Py_DECREF(py_best);
  return repr_str;
}

// standard members for TimeResult, all read-only. doc in TimeResult_doc.
static PyMemberDef TimeResult_members[] = {
  {"best", T_DOUBLE, offsetof(TimeResult, best), READONLY, NULL},
  {"unit", T_STRING, offsetof(TimeResult, unit), READONLY, NULL},
  {"number", T_PYSSIZET, offsetof(TimeResult, number), READONLY, NULL},
  {"repeat", T_PYSSIZET, offsetof(TimeResult, repeat), READONLY, NULL},
  {"times", T_OBJECT_EX, offsetof(TimeResult, times), READONLY, NULL},
  {"precision", T_INT, offsetof(TimeResult, precision), READONLY, NULL},
  // required sentinel, at least name must be NULL
  {NULL, 0, 0, 0, NULL}
};

// TimeResult docstrings for cached properties brief, loop_times
PyDoc_STRVAR(
  TimeResult_brief_doc,
  "A short string formatted similarly to that of timeit.main."
  "\n\n"
  "We can better describe this cached property by example. Suppose that\n"
  "calling ``repr`` on a ``TimeResult`` instance yields"
  "\n\n"
  ".. code:: python3"
  "\n\n"
  "   TimeResult(best=88.0, unit='usec', number=10000, repeat=5,\n"
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
  TimeResult_loop_times_doc,
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
// getters for TimeResult.brief and TimeResult.loop_times. documentation
// for these cached properties are in TimeResult_doc.
static PyGetSetDef TimeResult_getters[] = {
  {
    "brief", (getter) TimeResult_getbrief, NULL,
    TimeResult_brief_doc, NULL
  },
  {
    "loop_times", (getter) TimeResult_getloop_times, NULL,
    TimeResult_loop_times_doc, NULL
  },
  // sentinel required; name must be NULL
  {NULL, NULL, NULL, NULL, NULL}
};

PyDoc_STRVAR(
  TimeResult_doc,
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
  "unit : {" Py__timeunit_UNITS_STR "}\n"
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
// type object for the TimeResult type
static PyTypeObject TimeResult_type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  // full type name is package.subpackage[s].module.TimeResult
  .tp_name = "npapibench.functimer._timeresult.TimeResult",
  // docstring and size for the TimeResult type
  .tp_doc = TimeResult_doc,
  .tp_basicsize = sizeof(TimeResult),
  // not a variable-size object, so set to 0
  .tp_itemsize = 0,
  // omit Py_TPFLAGS_BASETYPE as this class is final
  .tp_flags = Py_TPFLAGS_DEFAULT,
  // custom __new__ function; no __init__ implementation for reinitialization
  .tp_new = TimeResult_new,
  // custom destructor
  .tp_dealloc = (destructor) TimeResult_dealloc,
  // standard class members; all are read-only
  .tp_members = TimeResult_members,
  // getters for the brief and loop_times cached properties
  .tp_getset = TimeResult_getters,
  // TimeResult __repr__ method
  .tp_repr = (reprfunc) TimeResult_repr
};

// _timeresult module docstring
PyDoc_STRVAR(
  _timeresult_doc,
  "Provides the TimeResult returned by functimer timing routines."
  "\n\n"
  ".. codeauthor:: Derek Huang <djh458@stern.nyu.edu>"
);
// _timeresult module definition
static struct PyModuleDef _timeresult_module = {
  PyModuleDef_HEAD_INIT,
  .m_name = "_timeresult",
  .m_doc = _timeresult_doc,
  .m_size = -1
};

// module init function
PyMODINIT_FUNC
PyInit__timeresult(void) {
  // PyObject * for module + API capsule, static void * array for C API
  PyObject *module, *capsule;
  static void *PyTimeResult_API[PyTimeResult_API_pointers];
  // try to import NumPy array API. on error, automatically returns NULL
  import_array();
  // try to import _timeunit C API. on error, automatically returns NULL
  import__timeunit();
  // check if type is ready. if error (return < 0), exception is set
  if (PyType_Ready(&TimeResult_type) < 0) {
    return NULL;
  }
  // create the module. if NULL, clean up and return NULL
  module = PyModule_Create(&_timeresult_module);
  if (module == NULL) {
    return NULL;
  }
  // initialize the pointers of the C API
  PyTimeResult_API[PyTimeResult_Type_NUM] = \
    (void *) &TimeResult_type;
  PyTimeResult_API[PyTimeResult_New_NUM] = \
    (void *) TimeResult_new;
  // create no-name capsule containing address to C array API.
  // PyModule_AddObject only steals ref on success, so Py_DECREF on error.
  capsule = PyCapsule_New((void *) PyTimeResult_API, NULL, NULL);
  // PyModule_AddObject returns NULL + sets exception if value arg is NULL
  if (PyModule_AddObject(module, "_C_API", capsule) < 0) {
    // need to Py_XDECREF since capsule may be NULL
    Py_XDECREF(capsule);
    goto except;
  }
  // add PyTypeObject * to module. PyModule_AddObject only steals a reference
  // on success, so on error (returns -1), must Py_DECREF &Timeitresult_type.
  Py_INCREF(&TimeResult_type);
  if (
    PyModule_AddObject(
      module, "TimeResult", (PyObject *) &TimeResult_type
    ) < 0
  ) {
    Py_DECREF(&TimeResult_type);
    // capsule had ref stolen and thus doesn't require Py_DECREF
    goto except;
  }
  // success, so return module. capsule, &TimeResult_type Py_DECREF'd
  return module;
// clean up on error
except:
  Py_DECREF(module);
  return NULL;
}
/**
 * @file _modinit.c
 * @brief Initialization file for the `c_npy_demo.functimer` module.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"
// need to include structmember.h to access the PyMemberDef struct definition
#include "structmember.h"

#include "functimer.h"
#include "timeitresult.h"

// module name and docstring
#define MODULE_NAME "functimer"
PyDoc_STRVAR(
  MODULE_DOC,
  "An internal C extension function timing module."
  "\n\n"
  "Inspired by :mod:`timeit` but times callables with arguments without using\n"
  "executable string statements. This way, timings of several different\n"
  "callables avoid multiple setup statement calls, which is wasteful."
  "\n\n"
  "Times obtained by default using :func:`time.perf_counter`."
  "\n\n"
  "Functional API only."
);

// method docstrings. note that for the signature to be correctly parsed, we
// need to place it in the docstring followed by "\n--\n\n"
PyDoc_STRVAR(
  FUNCTIMER_TIMEIT_ONCE_DOC,
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
PyDoc_STRVAR(
  FUNCTIMER_REPEAT_DOC,
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
PyDoc_STRVAR(
  FUNCTIMER_AUTORANGE_DOC,
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
/*
// update docstring later
PyDoc_STRVAR(
  FUNCTIMER_TIMEIT_ENH_DOC,
  "timeit_enh(func, args=None, kwargs=None, *, timer=None, number=None, "
  "repeat=None, unit=None, precision=1)\n"
  "--\n\n"
  ":rtype: :class:`~c_npy_demo.functimer.TimeitResult`"
);
*/
// TimeitResult class docstring
PyDoc_STRVAR(
  FUNCTIMER_TIMEITRESULT_DOC,
  "An immutable type for holding timing results from :func:`timeit_enh`."
  "\n\n"
  "All attributes are read-only. :attr:`loop_times` and :attr:`brief` are\n"
  "computed on demand when they are first accessed."
);
// TimeitResult docstrings for cached properties
PyDoc_STRVAR(
  FUNCTIMER_TIMEITRESULT_GETBRIEF_DOC,
  "A short string formatted similarly to that of :func:`timeit.main`."
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
  FUNCTIMER_TIMEITRESULT_GETLOOP_TIMES_DOC,
  "The unweighted average time taken per loop, per trial, in units of seconds."
  "\n\n"
  "Like :attr:`~TimeitResult.brief`, this is a cached property computed on\n"
  "the first access that yields new references on subsequent accesses."
);

// static array of module methods
static PyMethodDef functimer_methods[] = {
  {
    "timeit_once",
    // cast PyCFunctionWithKeywords to PyCFunction to silence compiler warning
    (PyCFunction) functimer_timeit_once,
    METH_VARARGS | METH_KEYWORDS,
    FUNCTIMER_TIMEIT_ONCE_DOC
  },
  {
    "repeat",
    (PyCFunction) functimer_repeat,
    METH_VARARGS | METH_KEYWORDS,
    FUNCTIMER_REPEAT_DOC
  },
  {
    "autorange",
    (PyCFunction) functimer_autorange,
    METH_VARARGS | METH_KEYWORDS,
    FUNCTIMER_AUTORANGE_DOC,
  },
  /*
  {
    "timeit_enh",
    (PyCFunction) functimer_timeit_enh,
    METH_VARARGS | METH_KEYWORDS,
    FUNCTIMER_TIMEIT_ENH_DOC
  },
  */
  // sentinel required; needs to have at least one NULL in it
  {NULL, NULL, 0, NULL}
};

// standard members for TimeitResult, all read-only
// todo: add docstrings for the members?
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

// getters for TimeitResult.brief and TimeitResult.loop_times
static PyGetSetDef TimeitResult_getters[] = {
  {
    "brief", (getter) TimeitResult_getbrief, NULL,
    FUNCTIMER_TIMEITRESULT_GETBRIEF_DOC, NULL
  },
  {
    "loop_times", (getter) TimeitResult_getloop_times, NULL,
    FUNCTIMER_TIMEITRESULT_GETLOOP_TIMES_DOC, NULL
  },
  // sentinel required; name must be NULL
  {NULL, NULL, NULL, NULL, NULL}
};

// static type definition struct
static PyTypeObject TimeitResult_type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  // full type name is c_npy_demo.MODULE_NAME.TimeitResult
  .tp_name = "c_npy_demo." MODULE_NAME "." TIMEITRESULT_NAME,
  // docstring and size for the TimeitResult struct
  .tp_doc = FUNCTIMER_TIMEITRESULT_DOC,
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

// static module definition struct
static struct PyModuleDef functimer_def = {
  PyModuleDef_HEAD_INIT,
  MODULE_NAME,
  MODULE_DOC,
  -1,
  functimer_methods
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
      module, TIMEITRESULT_NAME, (PyObject *) &TimeitResult_type
    ) < 0
  ) {
    Py_DECREF(&TimeitResult_type);
    Py_DECREF(module);
    return NULL;
  }
  // return module pointer
  return module;
}
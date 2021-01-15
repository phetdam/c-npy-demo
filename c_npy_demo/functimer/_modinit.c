/**
 * @file _modinit.c
 * @brief Initialization file for the `c_npy_demo.functimer` module.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "functimer.h"

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
  "timeit_once(func, args=None, kwargs=None, timer=None, number=1000000)\n"
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
  ":returns: Time for ``func`` to be executed ``number`` times with args\n"
  "    ``args`` and kwargs ``kwargs``, in units of ``timer``.\n"
  ":rtype: float"
);
/*
PyDoc_STRVAR(
  FUNCTIMER_REPEAT_DOC,
  "repeat(func, args=None, kwargs=None, number=1000000, repeat=5)\n"
  "--\n\n"
  "Operates in the same way as :func:`timeit.repeat`, i.e. the same was as\n"
  ":meth:`timeit.Timer.repeat`. TBA"
  "\n\n"
  ":rtype: list"
);
*/
PyDoc_STRVAR(
  FUNCTIMER_AUTORANGE_DOC,
  "autorange(func, args=None, kwargs=None, timer=None)\n"
  "--\n\n"
  "Determine number of time to call :func:`c_npy_demo.functimer.timeit_once."
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
PyDoc_STRVAR(
  FUNCTIMER_TIMEIT_DOC,
  "timeit(func, args=None, kwargs=None, number=None, repeat=None, "
  "unit=None)\n"
  "--\n\n"
  ":rtype: :class:`~c_npy_demo.functimer.TimeitResult`"
);
*/

// static array of module methods
static PyMethodDef functimer_methods[] = {
  {
    "timeit_once",
    // cast PyCFunctionWithKeywords to PyCFunction to silence compiler warning
    (PyCFunction) functimer_timeit_once,
    METH_VARARGS | METH_KEYWORDS,
    FUNCTIMER_TIMEIT_ONCE_DOC
  },
  /*
  {
    "repeat",
    (PyCFunction) functimer_repeat,
    METH_VARARGS | METH_KEYWORDS,
    FUNCTIMER_REPEAT_DOC
  },
  */
  {
    "autorange",
    (PyCFunction) functimer_autorange,
    METH_VARARGS | METH_KEYWORDS,
    FUNCTIMER_AUTORANGE_DOC,
  },
  /*
  {
    "timeit",
    (PyCFunction) functimer_timeit,
    METH_VARARGS | METH_KEYWORDS,
    FUNCTIMER_TIMEIT_DOC
  },
  */
  // sentinel required; needs to have at least one NULL in it
  {NULL, NULL, 0, NULL}
};

// module definition struct
static struct PyModuleDef functimer_def = {
  PyModuleDef_HEAD_INIT,
  MODULE_NAME,
  MODULE_DOC,
  -1,
  functimer_methods
};

// module initialization function
PyMODINIT_FUNC PyInit_functimer(void) {
  // create the module
  PyObject *module;
  module = PyModule_Create(&functimer_def);
  // return module pointer (could be NULL on failure)
  return module;
}
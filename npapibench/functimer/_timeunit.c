/**
 * @file _timeunit.c
 * @brief Defines time unit related globals and functions used by both the
 *     `_timeresult` and `_timeapi` C extension modules through a C API and
 *     also exposes these names in the Python layer through wrappers.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define TIMEUNIT_MODULE
#include "timeunit.h"

// NULL-terminated array of strings indicating valid unit values. these are
// used both by the TimeitResult_new function and by _timeapi functions.
static const char * const valid_units[] = {Py__timeunit_UNIT_LIST, NULL};
/**
 * bases corresponding to valid_units. each ith value corresponds to
 * the number a time in seconds should be multipled by to get the times in the
 * units given by the ith entry in valid_units.
 */
// 0-terminated array of doubles indicating valid unit bases. used by
// autoselect_time_unit when determining the "best" unit to display a time in.
static const double valid_unit_bases[] = {Py__timeunit_UNIT_BASE_LIST, 0};

/**
 * Get the corresponding base for a time unit `unit`.
 * 
 * If `unit` matches a value in `valid_units`, the corresponding unit base will
 * be returned (true), else 0 is returned. Returns 0 if `unit` is `NULL`.
 * 
 * @param unit `char const *`, must be `NULL`-terminated
 * @returns The corresponding base if valid unit in `valid_units`, 0 on error.
 */
static double
validate_unit(const char *unit)
{
  // if unit NULL, return 0 silently
  if (unit == NULL) {
    return 0;
  }
  int i = 0;
  // until the end of the valid_units array
  while (valid_units[i] != NULL) {
    // if identical, there's a match, so unit is valid. return the base.
    if (strcmp(valid_units[i], unit) == 0) {
      return valid_unit_bases[i];
    }
    i++;
  }
  // else unit is not valid
  return 0;
}

/**
 * Automatically determine units a time should be displayed in.
 * 
 * Used when `unit` is not passed to `_timeapi.timeit_enh` and will choose
 * the largest unit of time such that `fabs(best) >= 1`.
 * 
 * @param best Time in units of seconds
 * @param conv_p Memory location to write `best` converted to units returned
 *     by the function. If `NULL`, no writing is done.
 * @returns `char const *` to a value in `valid_units`.
 */
static const char *
autoselect_unit(const double best, double *conv_p)
{
  int i = 0;
  /**
   * loop through valid_unit_bases until we reach NULL or a unit such
   * that multiplying by the corresponding base b_i results in fabs(b_i * best)
   * to be less than 1. the final index is then decremented.
   */
  while (valid_unit_bases[i] != 0) {
    if (fabs(valid_unit_bases[i] * best) < 1) {
      break;
    }
    i++;
  }
  i--;
  // if conv_p is not NULL, write the converted time to that location
  if (conv_p != NULL) {
    *conv_p = valid_unit_bases[i] * best;
  }
  // return appropriate char const * from valid_units
  return valid_units[i];
}

// _timeunit docstring
PyDoc_STRVAR(
  _timeunit_doc,
  "Provides Python wrappers of time unit related globals from the C layer."
  "\n\n"
  "Exposes ``VALID_UNITS``, ``VALID_UNIT_BASES``, and ``MAX_PRECISION``."
  "\n\n"
  "``VALID_UNITS`` is a tuple of valid time units that may be passed to the\n"
  "``TimeitResult`` constructor or to ``_timeapi.timeit_enh``.\n"
  "``VALID_UNIT_BASES`` is a tuple of the float bases corresponding to the\n"
  "choices in ``VALID_UNITS`` indicating the value a time in seconds would\n"
  "have to be multiplied by to be converted to the respective unit in\n"
  "``VALID_UNITS``. ``MAX_PRECISION`` is just an int that gives the maximum\n"
  "display precision for the best per-loop time displayed in the ``brief``\n"
  "attribute of the ``TimeitResult``; see its docstring for more details."
  "\n\n"
  ".. codeauthor:: Derek Huang <djh459@stern.nyu.edu>"
);
// _timeunit module definition
static PyModuleDef _timeunit_module = {
  // name, docstring, size = -1 to disable subinterpreters. no methods.
  .m_name = "_timeunit",
  .m_doc = _timeunit_doc,
  .m_size = -1
};

// module init function
PyMODINIT_FUNC
PyInit__timeunit(void)
{
  // module object, valid_units as tuple of Python strings, valid_unit_bases
  // as tuple of Python strings, C API capsule, temp
  PyObject *module, *py_units, *py_unit_bases, *capsule, *temp_o;
  // static void * array for C API exposed by capsule
  static void *Py__timeunit_API[Py__timeunit_API_pointers];
  // create module, NULL on error
  module = PyModule_Create(&_timeunit_module);
  if (module == NULL) {
    return NULL;
  }
  // initialize the pointers for the C API
  Py__timeunit_API[Py__timeunit_UNITS_NUM] = \
    (void *) valid_units;
  Py__timeunit_API[Py__timeunit_UNIT_BASES_NUM] = \
    (void *) valid_unit_bases;
  Py__timeunit_API[Py__timeunit_validate_unit_NUM] = \
    (void *) validate_unit;
  Py__timeunit_API[Py__timeunit_autoselect_unit_NUM] = \
    (void *) autoselect_unit;
  // create no-name capsule containing address to C array API.
  // PyModule_AddObject only steals ref on success, so Py_DECREF on error.
  capsule = PyCapsule_New((void *) Py__timeunit_API, NULL, NULL);
  // PyModule_AddObject returns NULL + sets exception if value arg is NULL
  if (PyModule_AddObject(module, "_C_API", capsule) < 0) {
    // need to Py_XDECREF since capsule may be NULL
    Py_XDECREF(capsule);
    goto except_module;
  }
  // attempt to add the Py__timeunit_MAX_PRECISION constant to module
  if (
    PyModule_AddIntConstant(
      module, "MAX_PRECISION", Py__timeunit_MAX_PRECISION
    ) < 0
  ) {
    // note that capsule ref has been stolen so we do not Py_DECREF it
    goto except_module;
  }
  // create tuple for valid_units and populate with its units
  py_units = PyTuple_New(Py__timeunit_NUNITS);
  if (py_units == NULL) {
    goto except_module;
  }
  for (Py_ssize_t i = 0; i < Py__timeunit_NUNITS; i++) {
    // convert valid_units[i] to Python string and add to tuple on success. no
    // need to Py_DECREF temp_o since ref stolen by PyTuple_SET_ITEM.
    temp_o = PyUnicode_FromString(valid_units[i]);
    if (temp_o == NULL) {
      goto except_py_units;
    }
    PyTuple_SET_ITEM(py_units, i, temp_o);
  }
  // create tuple for valid_unit_bases and populate with its elements
  py_unit_bases = PyTuple_New(Py__timeunit_NUNITS);
  if (py_unit_bases == NULL) {
    goto except_py_units;
  }
  for (Py_ssize_t i = 0; i < Py__timeunit_NUNITS; i++) {
    temp_o = PyFloat_FromDouble(valid_unit_bases[i]);
    if (temp_o == NULL) {
      goto except_py_unit_bases;
    }
    PyTuple_SET_ITEM(py_unit_bases, i, temp_o);
  }
  // attempt to add py_units, py_unit_bases to module. add py_unit_bases first
  // so that on exception, we can use except_py_units label to clean up.
  if (PyModule_AddObject(module, "VALID_UNIT_BASES", py_unit_bases) < 0) {
    goto except_py_unit_bases;
  }
  if (PyModule_AddObject(module, "VALID_UNITS", py_units) < 0) {
    // py_unit_bases has had its reference stolen so we don't Py_DECREF it
    goto except_py_units;
  }
  // successfully add capsule, py_units, py_unit_bases (refs stolen), so return
  return module;
// clean up on exceptions
except_py_unit_bases:
  Py_DECREF(py_unit_bases);
except_py_units:
  Py_DECREF(py_units);
except_module:
  Py_DECREF(module);
  return NULL;
}
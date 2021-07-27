/**
 * @file timeunit.h
 * @brief Exposes a C API for static names in `_timeitresult.c`, in particular
 *     unit-related constants and a couple of utility functions.
 */

#ifndef TIMEUNIT_H
#define TIMEUNIT_H

// PY_SSIZE_T_CLEAN define must be guarded; may have been defined before
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif /* PY_SSIZE_T_CLEAN */

#include <Python.h>

#include "xstringify.h"

// maximum precision value that may be passed to the TimeitResult constructor
#define Py__timeunit_MAX_PRECISION 20
// number of valid units in Py__timeunit_UNITS, Py__timeunit_UNIT_BASES
#define Py__timeunit_NUNITS 4
// comma-separated list of valid values that unit can take. used to initialize
// Py__timeunit_UNITS and Py__timeunit_UNITS_STR.
#define Py__timeunit_UNIT_LIST "nsec", "usec", "msec", "sec"
// valid values used to initialize Py__timeunit_UNIT_BASES. all values are
// interpreted as doubles (Py__timeunit_UNIT_BASES is double const array)
#define Py__timeunit_UNIT_BASE_LIST 1e9, 1e6, 1e3, 1

// total number of function pointers stored in the void ** C API
#define Py__timeunit_API_pointers 4

// API indices for each of the exposed C names from _timeunit.c
#define Py__timeunit_UNITS_NUM 0
#define Py__timeunit_UNIT_BASES_NUM 1
#define Py__timeunit_validate_unit_NUM 2
#define Py__timeunit_autoselect_unit_NUM 3

// in client modules, define the void ** API and the import function.
// __INTELLISENSE__ always defined in VS Code; allows Intellisense to work here
#if defined(__INTELLISENSE__) || !defined(TIMEUNIT_MODULE)
static void **Py__timeunit_API;
// internal names from _timeunit
#define Py__timeunit_UNITS \
  ((const char * const *) \
  Py__timeunit_API[Py__timeunit_UNITS_NUM])
#define Py__timeunit_UNIT_BASES \
  ((const double *) \
  Py__timeunit_API[Py__timeunit_UNIT_BASES_NUM])
#define Py__timeunit_validate_unit \
  (*(double \
  (*)(const char *)) \
  Py__timeunit_API[Py__timeunit_validate_unit_NUM])
#define Py__timeunit_autoselect_unit \
  (*(const char *\
  (*)(const double, double *)) \
  Py__timeunit_API[Py__timeunit_autoselect_unit_NUM])
// macros to use Py__timeunit_UNITS, Py__timeunit_UNIT_BASES as strings
#define Py__timeunit_UNITS_STR xstringify(Py__timeunit_UNIT_LIST)
#define Py__timeunit_UNIT_BASES_STR xstringify(Py__timeuni_UNIT_BASE_LIST)

/**
 * Makes the `_timeunit.c` C API available in a client module.
 * 
 * @note `PyCapsule_Import` fails when trying to import the capsule using
 *     relative imports within the package since technically the subpackages
 *     have not yet been initialized, i.e. functimer not initialized yet as a
 *     member of the npapibench package. we make these work by doing a standard
 *     module import and then manually extracting the capsule. This is the
 *     process done by NumPy for its `_import_array` function.
 * 
 * @returns `-1` on failure, `0` on success.
 */
#include <stdio.h>
static int
_import__timeunit(void)
{
  PyObject *module, *capsule;
  // get module and if successful, try to get _C_API attribute
  module = PyImport_ImportModule("npapibench.functimer._timeunit");
  if (module == NULL) {
    return -1;
  }
  capsule = PyObject_GetAttrString(module, "_C_API");
  // don't need module anymore so we Py_DECREF it
  Py_DECREF(module);
  if (capsule == NULL) {
    return -1;
  }
  // check that capsule is actually a capsule
  if (!PyCapsule_CheckExact(capsule)) {
    PyErr_SetString(PyExc_RuntimeError, "_C_API is not a PyCapsule object");
    Py_DECREF(capsule);
    return -1;
  }
  // else if it is, try to get its pointer
  Py__timeunit_API = PyCapsule_GetPointer(capsule, NULL);
  if (Py__timeunit_API == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "_timeunit _C_API"); \
    return -1;
  }
  return 0;
}

#define import__timeunit() \
  { \
    if (_import__timeunit() < 0) { \
      PyErr_SetString(PyExc_ImportError, "could not import _timeunit C API"); \
      return NULL; \
    } \
  }
#endif /* defined(__INTELLISENSE__) || !defined(TIMEUNIT_MODULE) */

#endif /* TIMEUNIT_H */
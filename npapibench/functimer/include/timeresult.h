/**
 * @file timeresult.h
 * @brief Exposes a C API for static names in `_timeresult.c`. In particular,
 *     exposes the `PyTypeObject` for the `TimeResult` and a macro to its
 *     `__new__` method to allow creation of a `TimeResult` from C code.
 */

#ifndef TIMERESULT_H
#define TIMERESULT_H

// PY_SSIZE_T_CLEAN, NPY_NO_DEPRECATED_API defines must be guarded
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif /* PY_SSIZE_T_CLEAN */

#include <Python.h>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif /* NPY_NO_DEPRECATED_API */

#include <numpy/arrayobject.h>

// total number of function pointers stored in the void ** C API
#define PyTimeResult_API_pointers 2

// API indices for each of the exposed C names from _timeresult.c
#define PyTimeResult_Type_NUM 0
#define PyTimeResult_New_NUM 1

// in client modules, define the void ** API and the import function.
// __INTELLISENSE__ always defined in VS Code; allows Intellisense to work here
#if defined(__INTELLISENSE__) || !defined(TIMERESULT_MODULE)
// using NumPy naming convention
static void **PyTimeResult_API;
// internal names from _timeresult
#define PyTimeResult_Type \
  (*(PyTypeObject *) \
  PyTimeResult_API[PyTimeResult_Type_NUM])
#define PyTimeResult_New(args, kwargs) \
  (*(PyObject *\
  (*)(PyTypeObject *, PyObject *, PyObject *)) \
  PyTimeResult_API[PyTimeResult_New_NUM])(&PyTimeResult_Type, args, kwargs)

/**
 * Makes the `_timeresult.c` C API available in a client module.
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
static int
_import__timeresult(void)
{
  PyObject *module, *capsule;
  // get module and if successful, try to get _C_API attribute
  module = PyImport_ImportModule("npapibench.functimer._timeresult");
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
  PyTimeResult_API = PyCapsule_GetPointer(capsule, NULL);
  if (PyTimeResult_API == NULL) {
    return -1;
  }
  return 0;
}

#define import__timeresult() \
  { \
    if (_import__timeresult() < 0) { \
      PyErr_SetString( \
        PyExc_ImportError, "could not import _timeresult C API" \
      ); \
      return NULL; \
    } \
  }
#endif /* defined(__INTELLISENSE__) || !defined(TIMERESULT_MODULE) */

#endif /* TIMERESULT_H */
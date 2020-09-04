/**
 * @file _modinit.c
 * @brief Initializes the C extension module c_numpy_demo._ivmod.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL _IVMOD_ARRAY_API
#include "numpy/arrayobject.h"

#include "np_demo.h"
#include "np_euro_options.h"

// module name and docstring
#define MODULE_NAME "_ivmod"
#define MODULE_DOC \
"Demo module for interfacting with NumPy arrays in C.\n\n" \
"Also contains C implementations of functions to compute Bachelier and\n" \
"Black implied volatilities with either Halley's or Newton's method."

// method docstrings
#define PyObject_type_doc "Docstring for PyObject_type.\n\nTBD lorem ipsum"
#define PyArrayObject_sum_doc "Docstring for PyArrayObject_sum.\n\nTBD lorem" \
  " ipsum"
#define _black_vol_np_doc \
"Computes Black implied volatility for broadcastable arguments.\n\n" \
"    Do not call directly. Implied volatility is computed by a C\n" \
"    implementation of either Halley's or Newton's method, depending on the\n" \
"    method flag specified."

// static array of module methods
PyMethodDef mod_methods[] = {
  {"PyObject_type", PyObject_type, METH_VARARGS, PyObject_type_doc},
  {"PyArrayObject_sum", PyArrayObject_sum, METH_VARARGS, PyArrayObject_sum_doc},
  {"loop", loop, METH_VARARGS, "do a dumb loop"},
  {"_black_vol_np", _black_vol_np, METH_VARARGS, _black_vol_np_doc},
  {NULL, NULL, 0, NULL} // not sure why we need the sentinel here
};

// module definition struct
struct PyModuleDef mod_def = {
  PyModuleDef_HEAD_INIT,
  /**
   * module name, module docstring, per-interpreter module state (-1 if required
   * if state is maintained through variables), static pointer to methods
   */
  MODULE_NAME,
  MODULE_DOC,
  -1,
  mod_methods
};

// module initialization function
PyMODINIT_FUNC PyInit__ivmod(void) {
  // create the module
  PyObject *module;
  module = PyModule_Create(&mod_def);
  // import numpy api
  import_array();
  // error check?
  if (PyErr_Occurred()) {
    return NULL;
  }
  // return module pointer (could be NULL)
  return module;
}
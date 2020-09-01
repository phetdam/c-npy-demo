/**
 * @file _modinit.c
 * @brief Initializes the C extension module c_numpy_demo.cext.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL NP_TOUCH_ARRAY_API
#include "numpy/arrayobject.h"

#include "np_demo.h"

/* method docstring definitions */
#define PyObject_type_doc "Docstring for PyObject_type.\n\nTBD lorem ipsum"
#define PyArrayObject_sum_doc "Docstring for PyArrayObject_sum.\n\nTBD lorem" \
  " ipsum"

/* methods for np_touch */
PyMethodDef cext_methods[] = {
  {"PyObject_type", PyObject_type, METH_VARARGS, PyObject_type_doc},
  {"PyArrayObject_sum", PyArrayObject_sum, METH_VARARGS, PyArrayObject_sum_doc},
  {NULL, NULL, 0, NULL} /* sentinel, not sure why we need this */
};

/* definition of module cext */
struct PyModuleDef cext_module = {
  PyModuleDef_HEAD_INIT,
  "cext", /* module name */
  "Demo module for interfacing with NumPy arrays in C", /* module docstring */
  -1, /* something about the per-interpreter module state? -1 makes it global */
  cext_methods /* static pointer to methods */
};

/* module initialization function */
PyMODINIT_FUNC PyInit_cext(void) {
  /* create the module */
  PyObject *module;
  module = PyModule_Create(&cext_module);
  /* import numpy api */
  import_array();
  /* error check? */
  if (PyErr_Occurred()) {
    return NULL;
  }
  /* return module pointer (could be NULL) */
  return module;
}
/**
 * @file _modinit.c
 * @brief Initializes the C extension module c_numpy_demo._np_bcast.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define PY_ARRAY_UNIQUE_SYMBOL _IVMOD_ARRAY_API
#include "numpy/arrayobject.h"

#include "np_broadcast.h"

// module name and docstring
#define MODULE_NAME "_np_bcast"
#define MODULE_DOC \
"Contains C implementation of a NumPy 1D float64 argument broadcaster.\n\n" \
"    This function should but called from ``c_numpy_demo.utils``."

// method docstrings
#define np_float64_bcast_1d_ext_doc \
"Broadcasts arguments in an iterable into float64 :class:`numpy.ndarray`.\n\n" \
"    :param args: 1D iterable with argument broadcastable into column or\n" \
"        row vector float64 :class:`numpy.ndarray`.\n" \
"    :type args: tuple, list, dict\n" \
"    :param axis: ``0`` for flat vector, ``1`` for column vector.\n" \
"    :type axis: int" 

// static array of module methods
PyMethodDef mod_methods[] = {
  {"np_float64_bcast_1d_ext", np_float64_bcast_1d_ext, METH_VARARGS,
    np_float64_bcast_1d_ext_doc},
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
PyMODINIT_FUNC PyInit__np_bcast(void) {
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
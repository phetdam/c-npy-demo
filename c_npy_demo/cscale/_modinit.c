/**
 * @file _modinit.c
 * @brief Initializes the C extension module `c_npy_demo.cscale`.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

// don't include deprecated numpy C API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL CSCALE_ARRAY_API
#include "numpy/arrayobject.h"

#include "cscale.h"

// module name and docstring
#define MODULE_NAME "cscale"
PyDoc_STRVAR(
  MODULE_DOC, "The C implementation of :func:`c_npy_demo.pyscale.stdscale`."
);

// method docstrings. note that for the signature to be correctly parsed, we
// need to place it in the docstring followed by "\n--\n\n"
PyDoc_STRVAR(
  CSCALE_STDSCALE_DOC,
  "stdscale(ar, ddof = 1)\n--\n\n"
  "Centers and scales array to have zero mean and unit variance.\n\n"
  ":param args: Arbitrary :class:`numpy.ndarray`\n"
  ":type args: :class:`numpy.ndarray`\n"
  ":param ddof: Delta degrees of freedom, i.e. so that the divisor used\n"
  "    in standard deviation calculations is ``n_obs - ddof``.\n"
  ":type ddof: int\n"
  ":returns: Centered and scaled :class:`numpy.ndarray`\n"
  ":rtype: :class:`numpy.ndarray`"
);

// static array of module methods
static PyMethodDef cscale_methods[] = {
  {
    "stdscale",
    // cast PyCFunctionWithKeywords to PyCFunction (silences compiler warning)
    (PyCFunction) cscale_stdscale,
    METH_VARARGS | METH_KEYWORDS,
    CSCALE_STDSCALE_DOC
  },
  /**
   * see https://stackoverflow.com/questions/43371780/why-does-pymethoddef-
   * arrays-require-a-sentinel-element-containing-multiple-nulls. at least one
   * NULL should be present; defining a NULL method is more consistent.
   */
  {NULL, NULL, 0, NULL}
};

// module definition struct
static struct PyModuleDef cscale_def = {
  PyModuleDef_HEAD_INIT,
  /**
   * module name, module docstring, per-interpreter module state (-1 if required
   * if state is maintained through variables), static pointer to methods
   */
  MODULE_NAME,
  MODULE_DOC,
  -1,
  cscale_methods
};

// module initialization function
PyMODINIT_FUNC PyInit_cscale(void) {
  // create the module
  PyObject *module;
  module = PyModule_Create(&cscale_def);
  // import numpy api
  import_array();
  // error check?
  if (PyErr_Occurred()) {
    return NULL;
  }
  // return module pointer (NULL on failure)
  return module;
}
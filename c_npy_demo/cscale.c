/**
 * @file cscale.c
 * @brief C extension module containing core function to take a `numpy.ndarray`
 *     and return a transformed version with zero mean and unit variance. 
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <math.h>

// don't include deprecated numpy C API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// arrayobject.h gives access to the array API, npymath.h the core math library
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

/**
 * docstring for cscale.stdscale. for the function signature to be correctly
 * parse and show up in Python, we include it in the docstring and follow it
 * with "\n--\n\n"
 */
PyDoc_STRVAR(
  cscale_stdscale_doc,
  "stdscale(ar, ddof=1)\n--\n\n"
  "Centers and scales array to have zero mean and unit variance.\n\n"
  ":param args: Arbitrary :class:`numpy.ndarray`\n"
  ":type args: :class:`numpy.ndarray`\n"
  ":param ddof: Delta degrees of freedom, i.e. so that the divisor used\n"
  "    in standard deviation calculations is ``n_obs - ddof``.\n"
  ":type ddof: int\n"
  ":returns: Centered and scaled :class:`numpy.ndarray`\n"
  ":rtype: :class:`numpy.ndarray`"
);
// argument names for cscale_stdscale
static char *stdscale_argnames[] = {"ar", "ddof", NULL};
/**
 * Centers and scale a `numpy.ndarray` to zero mean, unit variance.
 * 
 * @param args Positional arguments
 * @param kwargs Keyword arguments
 * @returns `PyArrayObject *` cast to `PyObject *` 
 */
static PyObject *
cscale_stdscale(PyObject *self, PyObject *args, PyObject *kwargs) {
  // numpy ndarray, temp ndarray, delta degrees of freedom, size of ar
  PyArrayObject *ar, *ar_new;
  // on failed PyArg_ParseTupleAndKeyword, ar may not be modified
  ar = NULL;
  npy_intp ddof, ar_size;  
  ddof = 0;
  // check args and kwargs. | indicates that all args after it are optional.
  // exception is set on error automatically.
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O&|n", stdscale_argnames,
      &PyArray_Converter, (void *) &ar, &ddof
    )
  ) {
    goto except;
  }
  // check that ar is of the correct types
  if (!PyArray_ISINTEGER(ar) && !PyArray_ISFLOAT(ar)) {
    PyErr_SetString(PyExc_TypeError, "ar must have dtype int or float");
    goto except;
  }
  // check that ddof is nonnegative
  if (ddof < 0) {
    PyErr_SetString(PyExc_ValueError, "ddof must be a nonnegative int");
    goto except;
  }
  // get total number of elements in the array
  ar_size = PyArray_SIZE(ar);
  // if no elements, raise runtime warning and return ar (new ref)
  if (ar_size == 0) {
    PyErr_WarnEx(PyExc_RuntimeWarning, "mean of empty array", 1);
    PyErr_WarnEx(PyExc_RuntimeWarning, "division by 0", 1);
    return (PyObject *) ar;
  }
  // try to convert ar into double writeable C-contiguous array. NULL on error
  ar_new = (PyArrayObject *) PyArray_FROM_OTF(
    (PyObject *) ar, NPY_DOUBLE, NPY_ARRAY_CARRAY
  );
  if (ar_new == NULL) {
    goto except;
  }
  // else if successful, we can Py_DECREF original ar and swap pointers
  Py_DECREF(ar);
  ar = ar_new;
  // since array is type double, we can operate directly on the data.
  double *ar_data = (double *) PyArray_DATA(ar);
  // compute mean, standard deviation of data (naively)
  double ar_mean, ar_std;
  ar_mean = ar_std = 0;
  for (npy_intp i = 0; i < ar_size; i++) {
    // use ar_mean for sum of elements and ar_std for sum of squared elements
    ar_mean += ar_data[i];
    ar_std += ar_data[i] * ar_data[i];
  }
  // compute mean and standard deviation, with ddof built in
  ar_mean = ar_mean / ar_size;
  // more numerically stable than computing the MLE variance and then
  // multiplying  by ar / (ar_size - ddof), which is mathematically the same
  ar_std = sqrt(
    (ar_std - ar_size * ar_mean * ar_mean) / (ar_size - ddof)
  );
  // loop through elements of array again, centering and scaling them
  for (npy_intp i = 0; i < ar_size; i++) {
    ar_data[i] = (ar_data[i] - ar_mean) / ar_std;
  }
  return (PyObject *) ar;
// clean up and return NULL on exceptions
except:
  Py_XDECREF(ar);
  return NULL;
}

// static array of module methods
static PyMethodDef cscale_methods[] = {
  {
    "stdscale",
    // cast PyCFunctionWithKeywords to PyCFunction (silences compiler warning)
    (PyCFunction) cscale_stdscale,
    METH_VARARGS | METH_KEYWORDS,
    cscale_stdscale_doc
  },
  /**
   * see https://stackoverflow.com/questions/43371780/why-does-pymethoddef-
   * arrays-require-a-sentinel-element-containing-multiple-nulls. at least one
   * NULL should be present; defining a NULL method is more consistent.
   */
  {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(
  module_doc, "The C implementation of :func:`c_npy_demo.pyscale.stdscale`."
);
// module definition struct
static struct PyModuleDef cscale_def = {
  PyModuleDef_HEAD_INIT,
  // module name, module docstring, per-interpreter module state (-1 required
  // if state is maintained through globals), static pointer to methods
  .m_name = "cscale",
  .m_doc = module_doc,
  .m_size = -1,
  .m_methods = cscale_methods
};

// module initialization function
PyMODINIT_FUNC PyInit_cscale(void) {
  // import numpy api. on error, error indicator is set and NULL returned
  import_array();
  // create and return module pointer (NULL on failure)
  return PyModule_Create(&cscale_def);
}
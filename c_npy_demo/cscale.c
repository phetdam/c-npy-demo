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
#define PY_ARRAY_UNIQUE_SYMBOL CSCALE_ARRAY_API
// arrayobject.h gives access to the array API, npymath.h the core math library
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

// warns if NpyIter_Deallocate fails on PyArrayObject ar's NpyIter iter
#define NpyIter_DeallocAndWarn(iter, ar) if (NpyIter_Deallocate(iter) == \
  NPY_FAIL) { PyErr_WarnEx(PyExc_RuntimeWarning, "unable to deallocate " \
  "iterator of " #ar, 1); }

// module name and docstring
#define MODULE_NAME "cscale"
PyDoc_STRVAR(
  module_doc, "The C implementation of :func:`c_npy_demo.pyscale.stdscale`."
);

/**
 * docstring for cscale.stdscale. for the function signature to be correctly
 * parse and show up in Python, we include it in the docstring and follow it
 * with "\n--\n\n"
 */
PyDoc_STRVAR(
  cscale_stdscale_doc,
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
  // numpy ndarray and delta degrees of freedom
  PyArrayObject *ar;
  int ddof = 0;  
  // check args and kwargs. | indicates that all args after it are optional.
  // exception is set on error automatically.
  if (
    !PyArg_ParseTupleAndKeywords(
      args, kwargs, "O!|i", stdscale_argnames, &PyArray_Type, &ar, &ddof
    )
  ) {
    return NULL;
  }
  // check that ar is of the correct types
  if (!PyArray_ISINTEGER(ar) && !PyArray_ISFLOAT(ar)) {
    PyErr_SetString(PyExc_TypeError, "ar must have dtype int or float");
    return NULL;
  }
  // check that ddof is nonnegative
  if (ddof < 0) {
    PyErr_SetString(PyExc_ValueError, "ddof must be a nonnegative int");
    return NULL;
  }
  // get total number of elements in the array
  npy_int ar_size = PyArray_Size(ar);
  // if there aren't any elements, raise runtime warning and return NaN
  if (ar_size == 0) {
    PyErr_WarnEx(PyExc_RuntimeWarning, "mean of empty array", 1);
    return PyFloat_FromDouble(NPY_NAN);
  }
  /**
   * allocate output array onto ar. use C contiguous order, float64 (double)
   * type, aligned memory, and writeable array. has same elements as original.
   * 
   * NPY_ARRAY_CARRAY = NPY_ARRAY_C_CONTINUOUS | NPY_ARRAY_BEHAVED
   * NPY_ARRAY_BEHAVED = NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE
   * 
   * don't need to Py_INCREF the original ar.
   */
  ar = (PyArrayObject *) PyArray_FromArray(
    ar, PyArray_DescrFromType(NPY_DOUBLE), NPY_ARRAY_CARRAY
  );
  // if allocation fails, raise exception
  if (ar == NULL) {
    PyErr_SetString(
      PyExc_RuntimeError, "unable to allocate new result array with same shape"
    );
    return NULL;
  }
  // get typenum of ar
  int typenum = PyArray_TYPE(ar);
  // iterator, function to move to next inner loop, data pointer
  NpyIter *iter;
  NpyIter_IterNextFunc *iternext;
  char **data_ptr;
  /**
   * get new iterator.
   * 
   * NPY_ITER_READWRITE - read and write to array
   * NPY_ITER_C_INDEX - track raveled flat C index
   * NPY_CORDER - interpret array in C order (it was created in C order)
   * NPY_NO_CASTING - don't do any data type casts
   * 
   * technically specifying as a flag NPY_ITER_EXTERNAL_LOOP is supposed to be
   * more efficient but for simplicity purposes, we exclude it. that means that
   * *((double *) data_ptr[i]) is the current value for the ith array.
   */
  PyArray_Descr *dtype = PyArray_DescrFromType(typenum);
  iter = NpyIter_New(
    ar, NPY_ITER_READWRITE | NPY_ITER_C_INDEX, NPY_CORDER, NPY_NO_CASTING, dtype
  );
  // NpyIter_New doesn't steal a reference to dtype so we need to Py_DECREF
  Py_DECREF(dtype);
  // if iter is NULL, raise exception and Py_DECREF [new] ar
  if (iter == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "unable to get iterator for ar");
    Py_DECREF(ar);
    return NULL;
  }
  // get NpyIter_IterNextFunc pointer for checking if we can still iterate
  iternext = NpyIter_GetIterNext(iter, NULL);
  // if iternext is NULL, raise exception, deallocate iter, Py_DECREF ar
  if (iternext == NULL) {
    PyErr_SetString(
      PyExc_RuntimeError, "unable to get iternext from ar iterator"
    );
    // macro deallocating NpyIter iter of PyArrayObject ar with
    // NpyIter_Deallocate that raises a RuntimeWarning if deallocation fails
    NpyIter_DeallocAndWarn(iter, ar);    // don't actually need semicolon
    Py_DECREF(ar);
    return NULL;
  }
  // get pointer to data array. if we had multi-iterator, each ith element is
  // char * pointing to current element of the ith array.
  data_ptr = NpyIter_GetDataPtrArray(iter);
  // mean, standard deviation of flattened ar
  double ar_mean, ar_std;
  ar_mean = ar_std = 0;
  // iterate through the elements
  do {
    // since there is only one array (not multi-iterator), double-dereference.
    // need to cast to double for correct type interpretation.
    double cur_val = *((double *) *data_ptr);
    // use ar_mean as sum, use ar_std as squared sum. update
    ar_mean = ar_mean + cur_val;
    ar_std = ar_std + cur_val * cur_val;
  } while (iternext(iter));
  // compute mean and std dev. std dev computed as sqrt(E[X^2] - E[X]^2).
  ar_mean = ar_mean / ar_size;
  ar_std = sqrt(ar_std / ar_size - ar_mean * ar_mean);
  // reset the state of the iterator. if it fails, we need to raise exception,
  // deallocate iter with NpyIter_Deallocate (warns), and Py_DECREF ar
  if (NpyIter_Reset(iter, NULL) == NPY_FAIL) {
    PyErr_SetString(PyExc_RuntimeError, "failed to reset ar iterator state");
    NpyIter_DeallocAndWarn(iter, ar);
    Py_DECREF(ar);
  }
  // iterate through elements, centering and scaling
  do {
    double cur_val = *((double *) *data_ptr);
    *((double *) *data_ptr) = (cur_val - ar_mean) / ar_std;
  } while (iternext(iter));
  // done with iterator, so deallocate it. raise warning if failed
  NpyIter_DeallocAndWarn(iter, ar);
  // return centered and scaled out_ar
  return (PyObject *) ar;
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

// module definition struct
static struct PyModuleDef cscale_def = {
  PyModuleDef_HEAD_INIT,
  /**
   * module name, module docstring, per-interpreter module state (-1 required
   * if state is maintained through globals), static pointer to methods
   */
  MODULE_NAME,
  module_doc,
  -1,
  cscale_methods
};

// module initialization function
PyMODINIT_FUNC PyInit_cscale(void) {
  // import numpy api. on error, error indicator is set and NULL returned
  import_array();
  // create and return module pointer (NULL on failure)
  return PyModule_Create(&cscale_def);
}
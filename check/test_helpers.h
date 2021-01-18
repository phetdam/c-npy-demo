/**
 * @file test_helpers.h
 * @brief Header file containing useful declarations usable by all test suites.
 */

#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#endif /* PY_SSIZE_T_CLEAN */

// whether to exit the test runner immediately if Py_FinalizeEx returns an
// error. set to false by default so other tests can run.
extern int Py_Finalize_err_stop;
/**
 * calls Py_FinalizeEx with error handling controlled by Py_Finalize_err_stop.
 * optionally can return a value ret from the function it is called from if
 * Py_FinalizeEx errors. the typical way to use the macro is
 * 
 * Py_FinalizeEx_handle_err(return_this_on_error)
 * return the_normal_return_value;
 */
#define Py_FinalizeEx_handle_err(ret) if (Py_FinalizeEx() < 0) { \
  fprintf(stderr, "error: %s: Py_FinalizeEx error\n", __func__); \
  if (Py_Finalize_err_stop) { exit(120); } else { return ret; } }

#endif /* TEST_HELPERS_H */
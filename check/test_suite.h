/**
 * @file test_suite.h
 * @brief Header file for `test_suite.c`.
 */

#ifndef TEST_SUITE_H
#define TEST_SUITE_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#endif /* PY_SSIZE_T_CLEAN */

#include <check.h>

// whether to exit the test runner immediately the Py_FinalizeEx returns an
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
// returns main test suite run by the runner. takes timeout value
Suite *make_suite(double);

#endif /* TEST_SUITE_H */
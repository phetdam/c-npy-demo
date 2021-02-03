/**
 * @file test_helpers.h
 * @brief Header file containing useful declarations usable by all test suites.
 * @note must `#define NO_TEST_HELPERS_DEFINE` before all `#include`s of
 *     `test_helpers.h` except for one since the `py_setup`, `py_teardown`
 *     definitions are included in this header.
 */

#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#endif /* PY_SSIZE_T_CLEAN */

// empty macros indicating whether tests need the Python C API or not
#define PY_C_API_REQUIRED
#define NO_PY_C_API
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

// guard to only include definitions once. see note
#ifdef NO_TEST_HELPERS_DEFINE
void py_setup(void);
void py_teardown(void);
#else
/**
 * Python interpreter fixture setup to allow use of the Python C API
 */
void py_setup(void) {
  Py_Initialize();
}

/**
 * Python interpreter fixture teardown to finalize interpreter
 */
void py_teardown(void) {
  Py_FinalizeEx_handle_err()
}
#endif /* NO_TEST_HELPERS_DEFINE */

#endif /* TEST_HELPERS_H */
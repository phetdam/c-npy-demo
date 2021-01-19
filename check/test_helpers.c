/**
 * @file test_helpers.c
 * @brief Implementations for items declared in `test_helpers.h`.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "test_helpers.h"

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
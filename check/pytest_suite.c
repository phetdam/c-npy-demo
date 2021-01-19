/**
 * @file test_suite.c
 * @brief Embeds Python interpreter to run `pytest` tests and yield test suite.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stdio.h>

#include <check.h>

#include "pytest_suite.h"
#include "test_helpers.h"

/**
 * Runs `pytest` unit tests. `faulthandler` enabled.
 * 
 * See https://docs.python.org/3/library/faulthandler.html for details.
 */ 
START_TEST(test_pytest) {
  // required initialization function
  Py_Initialize();
  // import the faulthandler module. handle errors
  PyObject *faulth_mod = PyImport_ImportModule("faulthandler");
  if (faulth_mod == NULL) {
    
  }
  // import the pytest module 
  PyObject *pytest_mod = PyImport_ImportModule("pytest");
  // if NULL, exception was set, so the test failed. finalize + handle err
  if (pytest_mod == NULL) {
    Py_FinalizeEx_handle_err()
    return;
  }
  // try to access the main method from pytest
  PyObject *pytest_main = PyObject_GetAttrString(pytest_mod, "main");
  // if NULL, exception set. Py_DECREF pytest_mod and finalize + handle err
  if (pytest_main == NULL) {
    Py_DECREF(pytest_mod);
    Py_FinalizeEx_handle_err()
    return;
  }
  // run pytest.main; relies on local pytest.ini (returns exit code). on error,
  // exit code is NULL, so we use Py_XDECREF since we ignore the value
  Py_XDECREF(PyObject_CallObject(pytest_main, NULL));
  // Py_DECREF faulth_mod, faulth_enable, pytest_mod, pytest_main
  Py_DECREF(pytest_mod);
  Py_DECREF(pytest_main);
  // required finalization + handle any finalization error
  Py_FinalizeEx_handle_err()
}
END_TEST

/**
 * Create test suite `"pytest_suite"` using static test defined above.
 * 
 * Responsible for invoking `pytest`.
 * 
 * @param timeout `double` number of seconds for the test case's timeout
 * @returns libcheck `Suite *`, `NULL` on error
 */
Suite *make_pytest_suite(double timeout) {
  // if timeout is nonpositive, print error and return NULL
  if (timeout <= 0) {
    fprintf(stderr, "error: %s: timeout must be positive\n", __func__);
    return NULL;
  }
  // create suite called pytest_suite
  Suite *suite = suite_create("test_suite");
  // only one test case
  TCase *tc_core = tcase_create("core");
  // set test case timeout to timeout
  tcase_set_timeout(tc_core, timeout);
  // register case together with test, add to suite, and return suite
  tcase_add_test(tc_core, test_pytest);
  suite_add_tcase(suite, tc_core);
  return suite;
}
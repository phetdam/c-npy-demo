/**
 * @file test_suite.c
 * @brief Embeds Python interpreter to run `pytest` tests and yield test suite.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stdbool.h>
#include <stdio.h>

#include <check.h>

#include "pytest_suite.h"
#include "test_helpers.h"

/**
 * Runs `pytest` unit tests. `faulthandler` enabled.
 * 
 * See https://docs.python.org/3/library/faulthandler.html for details.
 * 
 * @note No need to call `Py_Initialize` and `Py_FinalizeEx` manually since the
 * `py_setup` and `py_teardown` fixtures handle that for us for each test.
 */ 
PY_C_API_REQUIRED START_TEST(test_pytest) {
  // import the faulthandler module. handle errors
  PyObject *faulth_mod = PyImport_ImportModule("faulthandler");
  if (faulth_mod == NULL) {
    ck_assert_msg(false, "unable to import faulthandler\n");
    return;
  }
  // try to access the faulthandler.enable method. Py_DECREF faulth_mod if err
  PyObject *faulth_enable = PyObject_GetAttrString(faulth_mod, "enable");
  if (faulth_enable == NULL) {
    Py_DECREF(faulth_mod);
    // note assert has to go after the Py_DECREF else the test is just aborted
    ck_assert_msg(false, "unable to import faulthandler.enable\n");
    return;
  }
  // enable faulthandler. on error, Py_DECREF faulth_mod, faulth_enable.
  PyObject *ignored = PyObject_CallObject(faulth_enable, NULL);
  Py_XDECREF(ignored);
  if (ignored == NULL) {
    Py_DECREF(faulth_mod);
    Py_DECREF(faulth_enable);
    ck_assert_msg(false, "call to faulthhandler.enable errored\n");
    return;
  }
  // import the pytest module 
  PyObject *pytest_mod = PyImport_ImportModule("pytest");
  // if NULL, exception was set, so the test failed. Py_DECREF as needed
  if (pytest_mod == NULL) {
    Py_DECREF(faulth_mod);
    Py_DECREF(faulth_enable);
    ck_assert_msg(false, "unable to import pytest\n");
    return;
  }
  // try to access the main method from pytest
  PyObject *pytest_main = PyObject_GetAttrString(pytest_mod, "main");
  // if NULL, exception set. Py_DECREF pytest_mod
  if (pytest_main == NULL) {
    Py_DECREF(faulth_mod);
    Py_DECREF(faulth_enable);
    Py_DECREF(pytest_mod);
    ck_assert_msg(false, "unable to import pytest.main\n");
    return;
  }
  // run pytest.main; relies on local pytest.ini (returns exit code). on error,
  // exit code is NULL, so we use Py_XDECREF since we ignore the value
  ignored = PyObject_CallObject(pytest_main, NULL);
  Py_XDECREF(ignored);
  // Py_DECREF faulth_mod, faulth_enable, pytest_mod, pytest_main
  Py_DECREF(faulth_mod);
  Py_DECREF(faulth_enable);
  Py_DECREF(pytest_mod);
  Py_DECREF(pytest_main);
  // if ignored is NULL, fail the test
  if (ignored == NULL) {
    ck_assert_msg(false, "call to pytest.main errored\n");
  }
}
END_TEST

/**
 * Create test suite `"pytest_suite"` using static test defined above.
 * 
 * Responsible for invoking `pytest`. Runs `py_setup` and `py_teardown` before
 * `pytest` session (and for any other unit tests, if added) so that a clean
 * Python interpreter exists for each test case.
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
  // add py_setup and py_teardown to tc_core test case (required)
  tcase_add_checked_fixture(tc_core, py_setup, py_teardown);
  // register case together with test, add to suite, and return suite
  tcase_add_test(tc_core, test_pytest);
  suite_add_tcase(suite, tc_core);
  return suite;
}
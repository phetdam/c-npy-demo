/**
 * @file test_suite.c
 * @brief Embeds Python interpreter to run `pytest` tests and yield test suite.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stdio.h>

#include <check.h>

#include "test_suite.h"
#include "timeitresult.h"

// whether to exit the test runner immediately the Py_FinalizeEx returns an
// error. set to false by default so other tests can run.
int Py_Finalize_err_stop;

/**
 * Runs `pytest` unit tests.
 */ 
START_TEST(test_pytest) {
  // required initialization function
  Py_Initialize();
  // import the pytest module 
  PyObject *module = PyImport_ImportModule("pytest");
  // if NULL, exception was set, so the test failed. finalize + handle err
  if (module == NULL) {
    Py_FinalizeEx_handle_err()
    return;
  }
  // try to access the main method from pytest
  PyObject *pytest_main = PyObject_GetAttrString(module, "main");
  // if NULL, exception set. Py_DECREF module and finalize + handle err
  if (pytest_main == NULL) {
    Py_DECREF(module);
    Py_FinalizeEx_handle_err()
    return;
  }
  // run pytest.main; relies on local pytest.ini (returns exit code). on error,
  // exit code is NULL, so we use Py_XDECREF since we ignore the value
  Py_XDECREF(PyObject_CallObject(pytest_main, NULL));
  // required finalization + handle any finalization error
  Py_FinalizeEx_handle_err()
}
END_TEST

/**
 * Test that `TimeitResult_validate_unit` works as expected.
 */
START_TEST(test_validate_unit) {
  // return false if arg is NULL
  ck_assert_msg(
    !TimeitResult_validate_unit(NULL), "%s: TimeitResult_validate_unit should "
    "return false if passed NULL pointer", __func__
  );
  // foobar is not a valid unit
  ck_assert_msg(
    !TimeitResult_validate_unit("foobar"), "%s: TimeitResult_validate_unit "
    "should not validate invalid unit \"foobar\"", __func__
  );
  // nsec is a valid unit
  ck_assert_msg(
    TimeitResult_validate_unit("nsec"), "%s: TimeitResult_validate_unit "
    "should validate valid unit \"nsec\"", __func__
  );
}

/**
 * Create test suite `"test_suite"` using static tests defined above.
 * 
 * @param timeout `double` number of seconds for the test case's timeout
 * @returns libcheck `Suite *`, `NULL` on error
 */
Suite *make_suite(double timeout) {
  // if timeout is nonpositive, print error and return NULL
  if (timeout <= 0) {
    fprintf(stderr, "error: %s: timeout must be positive\n", __func__);
    return NULL;
  }
  // create suite called test_suite
  Suite *suite = suite_create("test_suite");
  // our one and only test case, named core
  TCase *tc_core = tcase_create("core");
  // set test case timeout to timeout
  tcase_set_timeout(tc_core, timeout);
  // register case together with tests, add to suite, and return suite
  tcase_add_test(tc_core, test_pytest);
  tcase_add_test(tc_core, test_validate_unit);
  suite_add_tcase(suite, tc_core);
  return suite;
}
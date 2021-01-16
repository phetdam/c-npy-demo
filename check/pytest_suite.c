/**
 * @file pytest_suite.c
 * @brief Embeds Python interpreter to run `pytest` tests and yield test suite.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stdio.h>

#include <check.h>

#include "pytest_suite.h"

/**
 * Runs `pytest` unit tests.
 */ 
START_TEST(print_stuff)
{
  // required initialization function
  Py_Initialize();
  // load pytest main and run; relies on local pytest.ini
  PyRun_SimpleString("from pytest import main\nmain()\n");
  // required finalization function; if something went wrong, exit immediately
  if (Py_FinalizeEx() < 0) {
    // __func__ only defined in C99+
    fprintf(stderr, "error: %s: Py_FinalizeEx error\n", __func__);
    exit(120);
  }
}
END_TEST

/**
 * Create test suite `"pytest_suite"` with single case `"core"`.
 * 
 * @returns libcheck `Suite *`
 */
Suite *pytest_suite() {
  // create suite called pytest_suite
  Suite *suite = suite_create("pytest_suite");
  // our one and only test case, named core
  TCase *tc_core = tcase_create("core");
  // increase test timeout to 300s
  tcase_set_timeout(tc_core, 300);
  // register case together with test func, add to suite, and return suite
  tcase_add_test(tc_core, print_stuff);
  suite_add_tcase(suite, tc_core);
  return suite;
}
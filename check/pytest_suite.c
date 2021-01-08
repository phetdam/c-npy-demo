/**
 * @file pytest_suite.c
 * @brief Embeds Python interpreter to run `pytest` tests and yield test suite.
 */

#include <stdio.h>

#include <check.h>

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include "Python.h"

#include "pytest_suite.h"

/**
 * Unit test that prints some stuff and checks `3 == 3`.
 * 
 * @note Doesn't do anything for now except print stuff. Always passes. Since
 *     the macro defines `print_stuff` as static, the test suite creation
 *     function `pytest_suite` is nonstatic. Test catches `stdout` and `stderr`.
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
  // register case together with test func, add to suite, and return suite
  tcase_add_test(tc_core, print_stuff);
  suite_add_tcase(suite, tc_core);
  return suite;
}
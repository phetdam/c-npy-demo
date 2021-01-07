/**
 * @file pytest_suite.c
 * @brief Embeds Python interpreter to run `pytest` tests and yield test suite.
 */

#include <stdio.h>

#include <check.h>

#include "pytest_suite.h"

/**
 * Unit test that prints some stuff and checks `3 == 3`.
 * 
 * @note Doesn't do anything for now except print stuff. Always passes. Since
 *     the macro defines `print_stuff` as static, the test suite creation
 *     function `pytest_suite` is nonstatic.
 */ 
START_TEST (print_stuff)
{
  printf("hello in stdout\n");
  fprintf(stderr, "hello in stderr\n");
  // always true lmao
  ck_assert_int_eq(3, 3);
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
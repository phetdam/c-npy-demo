/**
 * @file timeitresult_suite.c
 * @brief Creates test suite for the `c_npy_demp.functimer.TimeitResult` class.
 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stdio.h>

#include <check.h>

#include "test_helpers.h"
#include "timeitresult.h"
#include "timeitresult_suite.h"

/**
 * Test that `TimeitResult_validate_unit` works as expected.
 */
NO_PY_C_API START_TEST(test_validate_unit) {
  // return false if arg is NULL
  ck_assert_msg(
    !TimeitResult_validate_unit(NULL), "TimeitResult_validate_unit should "
    "return false if passed NULL pointer"
  );
  // foobar is not a valid unit
  ck_assert_msg(
    !TimeitResult_validate_unit("foobar"), "TimeitResult_validate_unit should "
    "should not validate invalid unit \"foobar\""
  );
  // nsec is a valid unit
  ck_assert_msg(
    TimeitResult_validate_unit("nsec"), "TimeitResult_validate_unit should "
    "validate valid unit \"nsec\""
  );
} END_TEST

/**
 * Test that `TimeitResult_dealloc` raises appropriate exceptions.
 */
PY_C_API_REQUIRED START_TEST(test_dealloc) {
  // call and get borrowed reference to exception type
  TimeitResult_dealloc(NULL);
  PyObject *exc = PyErr_Occurred();
  // this pointer should not be NULL (exception was set)
  ck_assert_ptr_nonnull(exc);
  // check that exc is of type RuntimeError
  ck_assert_msg(
    PyErr_GivenExceptionMatches(exc, PyExc_RuntimeError),
    "TimeitResult_dealloc should set RuntimeError if given NULL pointer"
  );
} END_TEST

/**
 * Test that `TimeitResult_new` is argument safe for extern access.
 * 
 * @note Technically C API module functions should be static.
 */
PY_C_API_REQUIRED START_TEST(test_new_extern) {
  // dummy tuple needed to pass to args
  PyObject *args = PyTuple_New(0);
  // if NULL, test fails
  if (args == NULL) {
    fprintf(stderr, "error: %s: unable to allocate size-0 tuple\n", __func__);
    return;
  }
  // should set exception if first arg is NULL. exc is borrowed.
  TimeitResult_new(NULL, args, NULL);
  PyObject *exc = PyErr_Occurred();
  // exc should not be NULL
  ck_assert_ptr_nonnull(exc);
  // check that exc is RuntimeError
  ck_assert_msg(
    PyErr_GivenExceptionMatches(exc, PyExc_RuntimeError),
    "TimeitResult_new should set RuntimeError if type is NULL"
  );
  // should set exception if args is NULL. exc is borrowed. note the very
  // unsafe cast that is done for the type argument.
  TimeitResult_new((PyTypeObject *) args, NULL, NULL);
  exc = PyErr_Occurred();
  // Py_DECREF tuple since we don't need it anymore
  Py_DECREF(args);
  // exc should not be NULL and should be RuntimeError
  ck_assert_ptr_nonnull(exc);
  ck_assert_msg(
    PyErr_GivenExceptionMatches(exc, PyExc_RuntimeError),
    "TimeitResult_new should set RuntimeError if args is NULL"
  );
} END_TEST

/**
 * Test that `_TimeitResult_validate_units_bases` works as intended.
 * 
 * If this test fails, then `TimeitResult_units` and `TimeitResult_unit_bases`
 * have different lengths and thus the built module shouldn't be used.
 */
NO_PY_C_API START_TEST(test_units_bases_length) {
  // arrays that should make _TimeitResult_validate_units_bases return false
  char const *ar_1[] = {"one", "two", "three", NULL};
  double const br_1[] = {1, 2, 3, 4, 0};
  ck_assert_msg(
    !_TimeitResult_validate_units_bases(ar_1, br_1),
    "_TimeitResult_validate_units_bases should return false on ar_1, br_1"
  );
  // arrays that should make _TimeitResult_validate_units_bases return true
  char const *ar_2[] = {"one", "two", NULL};
  double const br_2[] = {1, 2, 0};
  ck_assert_msg(
    _TimeitResult_validate_units_bases(ar_2, br_2),
    "_TimeitResult_validate_units_bases should return true on ar_2, br_2"
  );
  // TimeitResult_validate_units_bases must return true
  ck_assert_msg(
    TimeitResult_validate_units_bases(),
    "TimeitResult_validate_units_bases returned false; check length of "
    "TimeitResult_units and TimeitResult_unit_bases"
  );
} END_TEST

/**
 * Create test suite `"timeitresult_suite"` using static tests defined above.
 * 
 * Invokes unit tests for `TimeitResult` in two cases. The first case,
 * `"py_core"` uses the `py_setup` and `py_teardown` functions to set up an
 * checked fixture (runs in forked address space at the start and end of each
 * unit test, so if the Python interpreter gets killed we can get a fresh one
 * for subsequent tests). The second case, `"c_core"`, doesn't use the Python C
 * API and has no need for other setup/teardown cases.
 * 
 * @param timeout `double` number of seconds for the test case's timeout
 * @returns libcheck `Suite *`, `NULL` on error
 */
Suite *make_timeitresult_suite(double timeout) {
  // if timeout is nonpositive, print error and return NULL
  if (timeout <= 0) {
    fprintf(stderr, "error: %s: timeout must be positive\n", __func__);
    return NULL;
  }
  // create suite called test_suite
  Suite *suite = suite_create("timeitresult_suite");
  // test case that contains unit tests that require Python C API
  TCase *tc_py_core = tcase_create("py_core");
  // test case that contains unit tests that don't require Python C API
  TCase *tc_c_core = tcase_create("c_core");
  // set test case timeouts to timeout
  tcase_set_timeout(tc_py_core, timeout);
  tcase_set_timeout(tc_c_core, timeout);
  // add py_setup and py_teardown to tc_py_core test case (required)
  tcase_add_checked_fixture(tc_py_core, py_setup, py_teardown);
  // register cases together with tests, add cases to suite, and return suite
  tcase_add_test(tc_py_core, test_dealloc);
  tcase_add_test(tc_py_core, test_new_extern);
  tcase_add_test(tc_c_core, test_validate_unit);
  tcase_add_test(tc_c_core, test_units_bases_length);
  suite_add_tcase(suite, tc_py_core);
  suite_add_tcase(suite, tc_c_core);
  return suite;
}
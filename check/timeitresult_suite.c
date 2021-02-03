/**
 * @file timeitresult_suite.c
 * @brief Creates test suite for the `c_npy_demp.functimer.TimeitResult` class.
 * @note If not compiled with -DC_NPY_DEMO_DEBUG then this file does nothing.
 */

// compilation unit active only if -DC_NPY_DEMO_DEBUG is passed to gcc
#ifdef C_NPY_DEMO_DEBUG

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stdbool.h>
#include <stdio.h>

#include <check.h>

#define NO_TEST_HELPERS_DEFINE
#include "test_helpers.h"

// only includes unit tests for the TimeitResult class if -DC_NPY_DEMO_DEBUG is
// passed to gcc on compilation else do nothing

// declarations for TimeitResult_units, TimeitResult_bases, and helper funcs
// in functimer.c. note everything is extern since C_NPY_DEMO_DEBUG defined.
extern char const * const TimeitResult_units[];
extern double const TimeitResult_unit_bases[];
int
TimeitResult_validate_unit(char const *);
char const *
TimeitResult_autounit(double const, double * const);
// include declaration for the TimeitResult_type PyTypeObject
extern PyTypeObject TimeitResult_type;
// include declaration for TimeitResult_new
PyObject *TimeitResult_new(PyTypeObject *, PyObject *, PyObject *);

/**
 * Check that `TimeitResult_units` and `TimeitResult_bases` have same length.
 * 
 * @note If the arrays has multiple trailing `NULL` values or `0` values, then
 *     the result of this function is inaccurate.
 * 
 * @returns `1` if lengths are matching, `0` otherwise
 */
int
TimeitResult_validate_units_bases() {
  int i = 0;
  // loop until we reach the end of one of the arrays
  while ((TimeitResult_units[i] != NULL) && (TimeitResult_unit_bases[i] != 0)) {
    i++;
  }
  // if last element of ar NULL and last element of br 0, then return true
  if ((TimeitResult_units[i] == NULL) && (TimeitResult_unit_bases[i] == 0)) {
    return true;
  }
  return false;
}

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
 * Test that `TimeitResult_validate_units_bases` works as intended.
 * 
 * If this test fails, then `TimeitResult_units` and `TimeitResult_unit_bases`
 * have different lengths and thus the built module shouldn't be used.
 */
NO_PY_C_API START_TEST(test_units_bases_length) {
  // TimeitResult_validate_units_bases must return true
  ck_assert_msg(
    TimeitResult_validate_units_bases(),
    "TimeitResult_validate_units_bases returned false; check length of "
    "TimeitResult_units and TimeitResult_unit_bases"
  );
} END_TEST

/**
 * Test that `TimeitResult_autounit` works as intended.
 */
NO_PY_C_API START_TEST(test_autounit) {
  // some times to pass to TimeitResult_autounit
  double t1, t2, t3;
  t1 = 1.2;
  t2 = 3e-3;
  t3 = 4.24e-5;
  // unit should be seconds for t1
  ck_assert_str_eq("sec", TimeitResult_autounit(t1, NULL));
  // unit should be milliseconds for t2
  ck_assert_str_eq("msec", TimeitResult_autounit(t2, NULL));
  // unit should be microseconds for t3
  ck_assert_str_eq("usec", TimeitResult_autounit(t3, NULL));
  // check that conversion works appropriately
  TimeitResult_autounit(t3, &t3);
  ck_assert_double_eq_tol(42.4, t3, 1e-8);
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
  // should set exception if first argument is not equal to the address of the
  // PyTypeObject TimeitResult_type. we use an unsafe cast to test.
  TimeitResult_new((PyTypeObject *) args, args, NULL);
  exc = PyErr_Occurred();
  // exc should not be NULL
  ck_assert_ptr_nonnull(exc);
  // check that exc is TypeError
  ck_assert_msg(
    PyErr_GivenExceptionMatches(exc, PyExc_TypeError),
    "TimeitResult_new should set TypeError if type is not &TimeitResult_type"
  );
  // should set exception if args is NULL. note use of TimeitResult_type
  // (non-static if compiled with DC_NPY_DEMO_DEBUG).
  TimeitResult_new(&TimeitResult_type, NULL, NULL);
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
 * Create test suite `"timeitresult_suite"` using static tests defined above.
 * 
 * Invokes unit tests for `TimeitResult` in two cases. The first case,
 * `"py_core"` uses the `py_setup` and `py_teardown` functions to set up an
 * checked fixture (runs in forked address space at the start and end of each
 * unit test, so if the Python interpreter gets killed we can get a fresh one
 * for subsequent tests). The second case, `"c_core"`, doesn't use the Python C
 * API and has no need for other setup/teardown cases.
 * 
 * @note Returns an empty suite if `C_NPY_DEMO_DEBUG` is not defined.
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
  tcase_add_test(tc_py_core, test_new_extern);
  tcase_add_test(tc_c_core, test_validate_unit);
  tcase_add_test(tc_c_core, test_units_bases_length);
  tcase_add_test(tc_c_core, test_autounit);
  suite_add_tcase(suite, tc_py_core);
  suite_add_tcase(suite, tc_c_core);
  return suite;
}

#endif /* C_NPY_DEMO_DEBUG */
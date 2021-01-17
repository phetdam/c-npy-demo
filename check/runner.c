/**
 * @file runner.c
 * @brief Main test runner.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// i use libcheck 0.15.2
#include <check.h>

#include "test_suite.h"

// program usage, nothing much for now
#define USAGE "usage: %s [-h]\n" \
  "libcheck runner. runs pytest by embedding the Python interpreter. invoke\n" \
  "with ./, i.e. from the same directory it is located in.\n", argv[0]

int main(int argc, char **argv) {
  // if no arguments provided, just run
  if (argc == 1) {}
  // one argument passed
  else if (argc == 2) {
    // if -h or --help, print usage
    if ((strcmp(argv[1], "-h") == 0) || (strcmp(argv[1], "--help") == 0)) {
      printf(USAGE);
      return EXIT_SUCCESS;
    }
    // else unknown argument
    fprintf(
      stderr, "%s: unknown argument '%s'. try %s --help for usage\n", argv[0],
      argv[1], argv[0]
    );
    return EXIT_FAILURE;
  }
  else {
    fprintf(
      stderr, "%s: too many arguments. try %s --help for usage\n", argv[0],
      argv[0]
    );
    return EXIT_FAILURE;
  }
  // instantiate our test suite. note this does not have to be freed!
  Suite *suite = make_suite(300);
  // if suite is NULL, there was an error, so print error + return EXIT_FAILURE
  if (suite == NULL) {
    fprintf(stderr, "error: %s: make_suite returned NULL\n", __func__);
    return EXIT_FAILURE;
  }
  // create our suite runner and run all tests (CK_ENV -> set CK_VERBOSITY and
  // if not set, default to CK_NORMAL, i.e. only show failed)
  SRunner *runner = srunner_create(suite);
  srunner_run_all(runner, CK_ENV);
  // get number of failed tests and free runner
  int n_failed = srunner_ntests_failed(runner);
  srunner_free(runner);
  // succeed/fail depending on value of number of failed cases
  return (n_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
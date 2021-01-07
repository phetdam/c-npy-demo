/**
 * @file runner.c
 * @brief Main test runner.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// installed libcheck 0.15.2
#include <check.h>

#include "pytest_suite.h"

// program usage, nothing much for now
#define USAGE "usage: %s [-h]\nsome usage\n", argv[0]

int main(int argc, char **argv) {
  // if no arguments provided, just run
  if (argc == 1) {
    ;
  }
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
  Suite *suite = pytest_suite();
  // create our suite runner and run all tests (CK_NORMAL -> only show failed)
  SRunner *runner = srunner_create(suite);
  srunner_run_all(runner, CK_NORMAL);
  // get number of failed tests and free runner
  int n_failed = srunner_ntests_failed(runner);
  srunner_free(runner);
  // succeed/fail depending on value of number of failed cases
  return (n_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
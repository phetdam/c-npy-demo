/**
 * @file runner.c
 * @brief Main test runner.
 */

#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// i use libcheck 0.15.2
#include <check.h>

#include "test_suite.h"

// long option names
#define help_longopt "help"
#define timeout_longopt "timeout"
#define verbose_longopt "verbose"
#define exit_py_longopt "exit-on-pyerr"

// long options for the test runner. values are equivalent to short flags.
static struct option runner_opts[] = {
  // pass this option to print usage
  {help_longopt, no_argument, NULL, 'h'},
  // pass with argument to specify the timeout duration (seconds)
  {timeout_longopt, required_argument, NULL, 't'},
  // force verbose output level. if not specified, defaults to CK_NORMAL
  {verbose_longopt, no_argument, NULL, 'v'},
  // stop runner on Py_FinalizeEx error. if not specified, false
  {exit_py_longopt, no_argument, NULL, 'E'}
};

// program usage
#define USAGE "usage: %s [-h] [-vE] [-t timeout]\n\n" \
  "libcheck runner. runs pytest by embedding the Python interpreter. invoke\n" \
  "with ./, i.e. from the same directory it is located in." \
  "\n\n" \
  "optional arguments:\n" \
  " -h, --help             show usage and exit\n" \
  " -t, --timeout timeout  specify libcheck timeout for a test case in\n" \
  "                        fractional seconds. defaults to 300.\n" \
  " -v, --verbose          print verbose test output, i.e. passing\n" \
  "                        CK_VERBOSE to srunner_run_all\n" \
  " -E, --exit-on-pyerr    exit test runner if Py_FinalizeEx errors, i.e.\n" \
  "                        an error occurred during finalization of the\n" \
  "                        Python interpreter state. usually unnecessary.\n", \
  argv[0]

int main(int argc, char **argv) {
  // verbosity flag, value of timeout. verbosity flag set to false initially
  int verbosity_flag;
  double timeout = 300;
  verbosity_flag = false;
  // get arguments using getopt_long
  while (true) {
    // long option index and value (short option flag) returned by getopt_long
    int opt_i, opt_flag;
    // get identifier (short flag) from getopt_long
    opt_flag = getopt_long(
      argc, (char * const *) argv, "ht:vE", runner_opts, &opt_i
    );
    // if opt_flag == -1, then we are done. break while
    if (opt_flag == -1) {
      break;
    }
    // switch on value of opt_flag
    switch (opt_flag) {
      // help flag passed; print usage and then exit
      case 'h':
        printf(USAGE);
        return EXIT_SUCCESS;
      // if verbose flag is passed, set it
      case 'v':
        verbosity_flag = true;
        break;
      // if timeout is passed, set it
      case 't':
        timeout = strtod(optarg, NULL);
        // if timeout is <= 0, the value is either invalid/there is an error,
        // so print to stderr and exit
        if (timeout <= 0) {
          fprintf(
            stderr, "error: %s: timeout value invalid. -t/--%s flag must be "
            "supplied with a positive double\n", argv[0], timeout_longopt
          );
          return EXIT_FAILURE;
        }
        break;
      // if we want to stop on execution on Py_FinalizeEx error, set
      // Py_Finalize_err_stop. not recommended.
      case 'E':
        fprintf(
          stderr, "warning: %s: -E/--%s specified. runner will exit on "
          "Py_FinalizeEx error\n", argv[0], exit_py_longopt
        );
        Py_Finalize_err_stop = true;
        break;
      // unknown option. getopt_long will print a message for us, so exit
      case '?':
        return EXIT_FAILURE;
      // something bad happened, so exit
      default:
        return 2;
    }
    // repeat the while loop until it is broken/we run into an error
  }
  // if optind < argc, then extraneous arguments were passed
  if (optind < argc) {
    fprintf(
      stderr, "%s: too many arguments. try %s -h/--%s for usage\n", argv[0],
      argv[0], help_longopt
    );
    return EXIT_FAILURE;
  }
  // done with option parsing. instantiate test suite; no free call necessary
  Suite *suite = make_suite(timeout);
  // if suite is NULL, there was an error, so print error + return EXIT_FAILURE
  if (suite == NULL) {
    fprintf(stderr, "error: %s: make_suite returned NULL\n", __func__);
    return EXIT_FAILURE;
  }
  // create our suite runner and run all tests (CK_ENV -> set CK_VERBOSITY and
  // if not set, default to CK_NORMAL, i.e. only show failed)
  SRunner *runner = srunner_create(suite);
  srunner_run_all(runner, verbosity_flag ? CK_VERBOSE : CK_NORMAL);
  // get number of failed tests and free runner
  int n_failed = srunner_ntests_failed(runner);
  srunner_free(runner);
  // succeed/fail depending on value of number of failed cases
  return (n_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
/**
 * @file pytest_suite.h
 * @brief Header file for `pytest_suite.c`.
 */

#ifndef TEST_SUITE_H
#define TEST_SUITE_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#endif /* PY_SSIZE_T_CLEAN */

#include <check.h>

// returns test suite pytest_suite. timeout is test case test timeout
Suite *make_pytest_suite(double);

#endif /* TEST_SUITE_H */
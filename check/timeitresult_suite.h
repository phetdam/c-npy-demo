/**
 * @file timeitresult_suite.h
 * @brief Header file for `timeitresult_suite.c`
 */

#ifndef TIMEITRESULT_SUITE_H
#define TIMEITRESULT_SUITE_H

#include <check.h>

// returns test suite timeitresult_suite. timeout is test case test timeout
Suite *make_timeitresult_suite(double timeout);

#endif /* TIMEITRESULT_SUITE_H */
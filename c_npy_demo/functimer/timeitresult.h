/**
 * @file timeitresult.h
 * @brief Declaration of `TimeitResult` struct in `functimer` and methods.
 */

#ifndef TIMEITRESULT_H
#define TIMEITRESULT_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#endif /* PY_SSIZE_T_CLEAN */

#include <stdbool.h>

// TimeitResult class name
#define TIMEITRESULT_NAME "TimeitResult"

// definition for the TimeitResult struct. not subclassable.
typedef struct {
  PyObject_HEAD
  // best per-loop runtime of the tested callable in units of unit
  double best;
  // string unit to use in the brief (report similar to timeit.main output).
  // either nsec, usec, msec, or sec, like in timeit.
  char const *unit;
  // number of loops the callable was run during each trial
  Py_ssize_t number;
  // number of trials that were run
  Py_ssize_t repeat;
  // tuples of per-loop runtimes, total runtimes for each trial. loop_times is
  // a cached property and will be created only upon access.
  PyObject *loop_times;
  PyObject *times;
  // precision to use when displaying best in brief. defaults to 1. in the
  // __new__ method precision will capped at 70.
  int precision;
  // cached property. Python string with output similar to timeit.main output
  PyObject *brief;
} TimeitResult;

// maximum precision value that may be passed to the TimeitResult constructor
#define TimeitResult_MAX_PRECISION 20
// list of valid values that unit can take. used to initialize
// TimeitResult_units and TimeitResult_UNITS_STR
#define TimeitResult_UNITS "nsec", "usec", "msec", "sec"
// valid values used to initialize TimeitResult_unit_bases. all values are
// interpreted as doubles (TimeitResult_unit_bases is double const array)
#define TimeitResult_UNIT_BASES 1e9, 1e6, 1e3, 1
/**
 * stringify combines varargs into a string, i.e. a, b, c -> "a, b, c", while
 * xstringify allows varargs to be macro expanded before stringification.
 * TimeitResult_UNITS_STR therefore simply escapes the quotes in
 * TimeitResult_UNITS and double quotes the whole thing.
 */
#define stringify(...) #__VA_ARGS__
#define xstringify(...) stringify(__VA_ARGS__)
#define TimeitResult_UNITS_STR xstringify(TimeitResult_UNITS)
// NULL-terminated array of strings indicating values unit can take
extern char const * const TimeitResult_units[];
/**
 * 0-terminated array of doubles, each ith value corresponding to a number
 * to multiply a time in seconds by to get time in the units given by the ith
 * unit in the TimeitResult_units NULL-terminated array
 */
extern double const TimeitResult_unit_bases[];

// helper macro to check that TimeitResult_units and TimeitResult_unit_bases
// are the same length. 1 if true, 0 otherwise. used during unit testing.
#define TimeitResult_validate_units_bases() \
  _TimeitResult_validate_units_bases( \
    TimeitResult_units, TimeitResult_unit_bases \
  )
/**
 * internal helper function to check that char const * const array terminated
 * with a single NULL and double const array terminated with single 0 have the
 * same length. see function comments for more details. 1 if true, 0 otherwise.
 * this function is used internally during unit testing.
 */
int _TimeitResult_validate_units_bases(
  char const * const * const, double const * const
);
// helper function for checking if unit has taken a valid string value, i.e.
// one of the values in TimeitResult_units. returns 1 if valid, 0 otherwise.
int TimeitResult_validate_unit(char const *);

// custom destructor (operates on self)
void TimeitResult_dealloc(TimeitResult *);
// custom __new__ implementation (subtype, args, kwargs). since we want the
// TimedResult to be immutable, we don't define a custom __init__ function.
PyObject *TimeitResult_new(PyTypeObject *, PyObject *, PyObject *);
// custom getter for loop_times so it works like @property decorator. first
// arg is self and the closure is unused.
PyObject *TimeitResult_getloop_times(TimeitResult *, void *);
// custom getter for brief so that it works like the @property decorator. first
// arg is self and the closure is unused.
PyObject *TimeitResult_getbrief(TimeitResult *, void *);
// custom __repr__ implementation (operates on self)
PyObject *TimeitResult_repr(TimeitResult *);

#endif /* TIMEITRESULT_H */
/**
 * @file timeitresult.h
 * @brief Declaration of `TimeitResult` struct in `functimer` and methods.
 */

#ifndef TIMEITRESULT_H
#define TIMEITRESULT_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"
#endif /* PY_SSIZE_T_CLEAN */

// definition for the TimeitResult struct
typedef struct {
    PyObject_HEAD
    // best per-loop runtime of the tested callable
    double best;
    // unit to use in the brief (report similar to timeit.main output)
    char const *unit;
    // number of loops the callable was run during each trial
    Py_ssize_t number;
    // number of trials that were run
    Py_ssize_t repeat;
    // tuples of per-loop runtimes, total runtimes for each trial
    PyObject *loop_times;
    PyObject *times;
    // cached property. Python string with output similar to timeit.main output
    PyObject *brief;
} TimeitResult;

// custom destructor (operates on self)
void TimeitResult_dealloc(PyObject *);
// custom __repr__ implementation (operate on self)
PyObject *TimeitResult_repr(PyObject *);
// custom __new__ implementation (subtype, args, kwargs)
PyObject *TimeitResult_new(PyTypeObject *, PyObject *, PyObject *);
// custom __init__ implementation (self, args, kwargs)
int TimeitResult_init(PyObject *, PyObject *, PyObject *);
// custom getter for brief so that it works like the @property decorator. first
// arg is self and the closure is unused.
PyObject *TimeitResult_getbrief(PyObject *, void *);

#endif /* TIMEITRESULT_H */
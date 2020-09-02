/**
 * @file root_find.h
 * @brief Declarations for functions defined in root_find.c.
 */

#ifndef ROOT_FIND_H
#define ROOT_FIND_H

#include <stdbool.h>

// name of method used to find root
#define HALLEY "halley"
#define NEWTON "newton"

// function name
#define _HALLEY_NEWTON_NAME "_halley_newton"

// reasons the function terminated, on normal run
#define HALLEY_NEWTON_TOL_REACHED "converged. absolute tolerance reached."
#define HALLEY_NEWTON_RTOL_REACHED "converged. relative tolerance reached."
#define HALLEY_NEWTON_MAXITER "failed to converge. reached max iteration."
#define HALLEY_NEWTON_DZERO "failed to converge. first derivative is zero."
#define HALLEY_NEWTON_INVALID_PARAM "invalid parameter(s)."

/**
 * struct holding results of scalar root-finding optimization routine. we define
 * convergence to mean that either rtol (relative tolerance) or tol (absolute
 * function tolerance was met during optimization).
 */
typedef struct {
  double res;          // result, i.e. guess of where root is
  int iters;           // number of iterations taken to arrive at res
  bool converged;      // true (1) if converged, false (0) otherwise
  const char *method;  // optimization method used
  const char *flag;    // gives reason for termination of function
} scl_rf_res;

/** 
 * Macro for populating auto scl_rf_res struct when an invalid parameter is
 * encountered in _halley_newton.
 * 
 * @param obj scl_rf_res auto struct
 * @param x Guessed value of the root
 * @returns Macro to populate obj appropriately if a parameter is valid.
 */
#define HALLEY_NEWTON_INVALID_PARAM_RES(obj, x) \
  obj.res = x; obj.iters = 0; obj.converged = false; obj.method = NULL; \
  obj.flag = HALLEY_NEWTON_INVALID_PARAM

// typedef pointer to scalar function with void * fixed args as scl_func_wargs
typedef double (*scl_func_wargs)(double, void *);

// struct is small enough that we just return directly without malloc
scl_rf_res _halley_newton(scl_func_wargs obj, double x0, scl_func_wargs obj_d1,
  scl_func_wargs obj_d2, void *_args, double tol, double rtol, int maxiter,
  bool debug);

#define halley_newton_debug(a, b, c, d, e, f, g, h) \
  _halley_newton(a, b, c, d, e, f, g, h, true)
#define halley_newton(a, b, c, d, e, f, g, h) \
  _halley_newton(a, b, c, d, e, f, g, h, false)
#define halley(a, b, c, d, e, f, g, h) halley_newton(a, b, c, d, e, f, g, h)
#define newton(a, b, c, e, f, g, h) halley_newton(a, b, c, NULL, e, f, g, h)

#endif /* ROOT_FIND_H */
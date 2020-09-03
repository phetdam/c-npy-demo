/**
 * @file root_find.c
 * @brief Implements Newton's and Halley's method for 1D root finding.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include "root_find.h"

/**
 * Newton's and Halley's method for one-dimensional real root-finding.
 * 
 * @note Influenced by scipy.optimize.newton. Requires first derivative.
 * 
 * @param obj The function we want to find a root for
 * @param x0 Initial guess for the root
 * @param obj_d1 First derivative of obj
 * @param obj_d2 Second derivative of obj. Pass NULL to use Newton's method, 
 * else Halley's method will be used by default.
 * @param _args Pointer to a struct containing fixed arguments for obj, obj_d1,
 * and obj_d2 is it is not NULL.
 * @param tol Absolute tolerance of difference between guess and actual root.
 * @param rtol Relative tolerance of difference between guesses.
 * @param maxiter Maximum number of iterations to run before termination.
 * @param debug true to print debug messages, false for silence except on error.
 * @returns scl_rf_res struct containing res, the guessed root, iters, the
 * total number of iterations ran, converged, true (1) if converged and false
 * (0) otherwise, method, either "halley" or "newton", flag, which gives the
 * reason for termination.
 */
scl_rf_res _halley_newton(scl_func_wargs obj, double x0, scl_func_wargs obj_d1,
  scl_func_wargs obj_d2, void *_args, double tol, double rtol, int maxiter,
  bool debug) {
  // auto struct to hold results
  scl_rf_res res;
  // if obj is NULL, we don't have objective, so print error and return
  if (obj == NULL) {
    fprintf(stderr, "%s: no objective provided. stopping.\n",
      _HALLEY_NEWTON_NAME);
    HALLEY_NEWTON_INVALID_PARAM_RES(res, x0);
    return res;
  }
  // if obj_d1 is NULL, we don't have first deriv, so print error and return
  if (obj_d1 == NULL) {
    fprintf(stderr, "%s: no first derivative provided. stopping.\n",
      _HALLEY_NEWTON_NAME);
    HALLEY_NEWTON_INVALID_PARAM_RES(res, x0);
    return res;
  }
  // if no second deriv, fall back to newton's method, else use halley's
  if (obj_d2 == NULL) {
    if (debug) {
      printf("%s: no second derivative. using newton's method\n", 
        _HALLEY_NEWTON_NAME);
    }
    res.method = NEWTON;
  }
  else {
    res.method = HALLEY;
  }
  // if _args is NULL, print error and return
  if (_args == NULL) {
    fprintf(stderr, "%s: error: _args is NULL. stopping.\n",
      _HALLEY_NEWTON_NAME);
    HALLEY_NEWTON_INVALID_PARAM_RES(res, x0);
    return res;
  }
  // tol and rtol must be nonnegative and maxiter must be positive
  if (tol < 0) {
    fprintf(stderr, "%s: error: tol must be nonnegative\n",
      _HALLEY_NEWTON_NAME);
    HALLEY_NEWTON_INVALID_PARAM_RES(res, x0);
    return res;
  }
  if (rtol < 0) {
    fprintf(stderr, "%s: error: rtol must be nonnegative\n",
      _HALLEY_NEWTON_NAME);
    HALLEY_NEWTON_INVALID_PARAM_RES(res, x0);
    return res;
  }
  if (maxiter <= 0) {
    fprintf(stderr, "%s: error: maxiter must be positive\n",
      _HALLEY_NEWTON_NAME);
    HALLEY_NEWTON_INVALID_PARAM_RES(res, x0);
    return res;
  }
  // get current + prev guess, current tol, current relative tol, and iter count
  double guess, prev_guess, cur_tol, cur_rtol;
  int iternum;
  guess = x0;
  cur_tol = INFINITY;
  cur_rtol = INFINITY;
  iternum = 0;
  // function value, function first derivative, function second derivative
  double fval, d1_val, d2_val;
  // while not converged
  while ((cur_tol > tol) && (cur_rtol > rtol) && (iternum < maxiter)) {
    // evaluate function and first derivative at guess
    fval = (*obj)(guess, _args);
    d1_val = (*obj_d1)(guess, _args);
    // if first derivative is 0, immediately return after populating res
    if (d1_val == 0) {
      res.res = guess;
      res.iters = iternum;
      res.converged = false;
      res.flag = HALLEY_NEWTON_DZERO;
      if (debug) {
        printf("%s: first derivative is 0. convergence failed.\n",
          _HALLEY_NEWTON_NAME);
      }
      return res;
    }
    // save guess as prev guess
    prev_guess = guess;
    // if obj_d2 is NULL, use newton's method, else use halley's method
    if (obj_d2 == NULL) {
      guess = guess - fval / d1_val;
    }
    else {
      d2_val = (*obj_d2)(guess, _args);
      guess = guess - (2 * fval * d1_val) /
        (2 * pow(d1_val, 2) - fval * d2_val);
    }
    // record cur_tol and cur_rtol
    cur_tol = abs(guess - x0);
    cur_rtol = abs(guess - prev_guess);

    // increment iteration count
    iternum = iternum + 1;
  }
  // finish populating and return
  res.res = guess;
  res.iters = iternum;
  // determine how the function converged
  if (cur_tol < tol) {
    res.flag = HALLEY_NEWTON_TOL_REACHED;
    res.converged = true;
  }
  else if (cur_rtol < tol) {
    res.flag = HALLEY_NEWTON_RTOL_REACHED;
    res.converged = true;
  }
  else {
    res.flag = HALLEY_NEWTON_MAXITER;
    res.converged = false;
  }
  if (debug) {
    printf("%s: execution successful. results:\n\n"
      "res:       %f\niters:     %d\nconverged: %s\nmethod:    %s\nflag:      "
      "%s\n\n", _HALLEY_NEWTON_NAME, res.res, res.iters,
      res.converged ? "true" : "false", res.method, res.flag);
  }
  return res;
}
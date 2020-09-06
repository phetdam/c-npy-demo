/**
 * @file euro_options.h
 * @brief Declarations for functions defined in euro_options.c.
 */

#ifndef IMPLIED_VOL_H
#define IMPLIED_VOL_H

#include <stdbool.h>

#include "root_find.h"

// bump implied vol to machine epsilon if it is <= 0
#define BUMP_VOL(x) x = (x <= 0) ? 2.22e-16 : x;

// price, vega, and volga functions for black and bachelier model
double black_price(double fwd, double strike, double ttm, double ivol,
  double df, int is_call);

#define black_call_price(a, b, c, d, e) black_price(a, b, c, d, e, 1)
#define black_put_price(a, b, c, d, e) black_price(a, b, c, d, e, -1)

double black_vega(double fwd, double strike, double ttm, double ivol, 
  double df, int is_call);
double black_volga(double fwd, double strike, double ttm, double ivol,
  double df, int is_call);
double bachelier_price(double fwd, double strike, double ttm, double ivol,
  double df, int is_call);

#define bachelier_price_call(a, b, c, d, e) bachelier_price(a, b, c, d, e, 1)
#define bachelier_price_put(a, b, c, d, e) bachelier_price(a, b, c, d, e, -1)

double bachelier_vega(double fwd, double strike, double ttm, double ivol,
  double df);
double bachelier_volga(double fwd, double strike, double ttm, double ivol,
  double df);

// struct for fixed args when objective functions are called by optimizer
typedef struct {
  double price;    // true option price
  double fwd;      // forward level
  double strike;   // strike
  double ttm;      // time to maturity in years
  double df;       // discount factor in (0, 1]
  int is_call;     // +/-1 for call/put
} vol_obj_args;

// create using macro (substitution) as auto variable. faster than using malloc.
#define vol_obj_args_anew(p, f, s, t, d, i) {p, f, s, t, d, i}
// populate an already existing vol_obj_args voa
#define vol_obj_args_afill(voa, p, f, s, t, d, i) \
  voa.price = p; voa.fwd = f; voa.strike = s; voa.ttm = t; voa.df = d; \
  voa.is_call = i

// create new vol_obj_args using malloc
vol_obj_args *vol_obj_args_mnew(double price, double fwd, double strike,
  double ttm, double df, int is_call);

// objective functions for solving for black and bachelier implied vol
double black_vol_obj(double ivol, void *_args);
double bachelier_vol_obj(double ivol, void *_args);

// first and second derivatives wrt ivol of the above objectives
double black_vol_obj_d1(double ivol, void *_args);
double black_vol_obj_d2(double ivol, void *_args);
double bachelier_vol_obj_d1(double ivol, void *_args);
double bachelier_vol_obj_d2(double ivol, void *_args);

// names of the implied volatility functions
#define _BLACK_VOL_NAME "_black_vol"
#define _BACHELIER_VOL_NAME "_bachelier_vol"

// functions + macros to compute implied volatilities from vol_obj_args data
scl_rf_res _black_vol(vol_obj_args *odata, scl_opt_flag method, double x0,
  double tol, double rtol, int maxiter, bool debug);
scl_rf_res _bachelier_vol(vol_obj_args *odata, scl_opt_flag method, double x0,
  double tol, double rtol, int maxiter, bool debug);

#define black_vol(o, m, x, t, r, i) _black_vol(o, m, x, t, r, i, false)
#define black_vold(o, m) black_vol(o, m, 0.5, 1.48e-8, 0, 50)
#define bachelier_vol(o, m, x, t, r, i) _bachelier_vol(o, m, x, t, r, i, false)
#define bachelier_vold(o, m) bachelier_vol(o, m, 0.5, 1.48e-8, 0, 50)

// flags for indicating volatility type, function name, and another int typedef
#define BLACK_VOL_FLAG 0
#define BACHELIER_VOL_FLAG 1
#define _IMP_VOL_VEC_NAME "imp_vol_vec"
typedef int vol_t_flag;

/**
 * typedef for function pointer with same signature as _black_vol and
 * _bachelier_vol for convenience. long signature.
 */
typedef scl_rf_res (*scl_vol_func)(vol_obj_args *, scl_opt_flag, double, double,
  double, int, bool);

// computes implied vol for an entire array of options data
void _imp_vol_vec(vol_obj_args *odata, double *vdata, long n_pts,
  vol_t_flag vol_type, scl_opt_flag method, double x0, double tol, double rtol,
  int maxiter, int n_threads, bool mask_neg, bool debug);

#define imp_vol_vec(o, np, v, m, x, t, r, i, nt) \
  _imp_vol_vec(o, np, v, m, x, t, r, i, nt, false)
#define imp_vol_vecd(o, np, v, m, nt) \
  _imp_vol_vec(o, np, v, m, 0.5, 1.48e-8, 0, 50, nt, false)


#endif /* IMPLIED_VOL_H */
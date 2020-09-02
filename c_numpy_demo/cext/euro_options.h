/**
 * @file euro_options.h
 * @brief Declarations for functions defined in euro_options.c.
 */

#ifndef IMPLIED_VOL_H
#define IMPLIED_VOL_H

// create using macro (substitution) as auto variable. faster than using malloc.
#define vol_obj_args_anew(a, b, c, d, e, f) {a, b, c, d, e, f}

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

#endif /* IMPLIED_VOL_H */
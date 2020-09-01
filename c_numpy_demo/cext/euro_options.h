/**
 * @file euro_options.h
 * @brief Declarations for functions defined in euro_options.c.
 */

#ifndef IMPLIED_VOL_H
#define IMPLIED_VOL_H

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
double _bachelier_vega(double fwd, double strike, double ttm, double ivol,
  double df, int is_call);
#define bachelier_vega(a, b, c, d, e) _bachelier_vega(a, b, c, d, e, 1)
double _bachelier_volga(double fwd, double strike, double ttm, double ivol,
  double df, int is_call);
#define bachelier_volga(a, b, c, d, e) _bachelier_volga(a, b, c, d, e, 1)

#endif /* IMPLIED_VOL_H */
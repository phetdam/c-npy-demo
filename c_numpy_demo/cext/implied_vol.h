/**
 * @file implied_vol.h
 * @brief Function definitions for implied volatility code in implied_vol.c
 */

#ifndef IMPLIED_VOL_H
#define IMPLIED_VOL_H

double normal_pdf(double x, double mu, double sigma);
#define std_normal_pdf(x) normal_pdf(x, 0, 1)
double normal_cdf(double x, double mu, double sigma);
#define std_normal_cdf(x) normal_cdf(x, 0, 1)
double black_price(double fwd, double strike, double ttm, double ivol,
  double df, int is_call);
double bachelier_price(double fwd, double strike, double ttm, double ivol,
  double df, int is_call);

#endif /* IMPLIED_VOL_H */
/**
 * @file euro_options.c
 * @brief European option price and related functions.
 * @note No longer trivial code. Will most likely be used in py-impvol.
 */

#include <stdio.h>
#include <math.h>

#include "gauss.h"
#include "euro_options.h"

/**
 * Compute the Black price of a European option.
 * 
 * @param fwd Current level of the forward (underlying) in units of price
 * @param strike Option strike, must be same units as fwd
 * @param ttm Time to maturity in years
 * @param ivol Black implied volatility / 100, i.e. percentage / 100
 * @param df Optional discount factor in (0, 1].
 * @param is_call +1 for call, -1 for put.
 * @returns Black price for a European option.
 */
double black_price(double fwd, double strike, double ttm, double ivol,
  double df, int is_call) {
  // compute total volatility, ivol * sqrt(ttm
  double tot_vol;
  tot_vol = ivol * sqrt(ttm);
  // compute log of fwd / strike
  double log_m;
  log_m = log(fwd / strike);
  // compute d1 and d2
  double d1, d2;
  d1 = log_m / tot_vol + tot_vol / 2;
  d2 = d1 - tot_vol;
  // compute the price. call if +1, put if -1
  if (is_call == 1) {
    return df * (fwd * std_normal_cdf(d1) - strike * std_normal_cdf(d2));
  }
  else if (is_call == -1) {
    return df * (strike * std_normal_cdf(-d2) - fwd * std_normal_cdf(-d1));
  }
  // else print error and return -HUGE_VAL if is_call is incorrect
  fprintf(stderr, "black_price: is_call must be +/-1\n");
  return NAN;
}

/**
 * Black option vega.
 * 
 * @param fwd Current level of the forward (underlying) in units of price
 * @param strike Option strike, must be same units as fwd
 * @param ttm Time to maturity in years
 * @param ivol Black implied volatility / 100, i.e. percentage / 100
 * @param df Optional discount factor in (0, 1].
 * @param is_call +1 for call, -1 for put.
 * @returns Black option vega.
 */
double black_vega(double fwd, double strike, double ttm, double ivol, 
  double df, int is_call) {
  // compute log(F / K) total volatility
  double log_m, tot_vol;
  log_m = log(fwd / strike);
  tot_vol = ivol * sqrt(ttm);
  // compute d1 and d2
  double d1, d2;
  d1 = log_m / tot_vol + tot_vol / 2;
  d2 = d1 - tot_vol;
  // compute partial of d1 wrt ivol and partial of d2 wrt ivol
  double d1dvol, d2dvol;
  d1dvol = -log_m / tot_vol / ivol + sqrt(ttm) / 2;
  d2dvol = d1dvol - sqrt(ttm);
  // if +1, call, if -1, put, else print error and return NAN
  if (is_call == 1) {
    return df * (fwd * std_normal_pdf(d1) * d1dvol -
      strike * std_normal_pdf(d2) * d2dvol);
  }
  else if (is_call == -1) {
    return df * (fwd * std_normal_pdf(-d1) * d1dvol - 
      strike * std_normal_pdf(-d2) * d2dvol);
  }
  fprintf(stderr, "black_vega: is_call must be +/-1\n");
  return NAN;
}

/**
 *  Black option volga.
 * 
 * @param fwd Current level of the forward (underlying) in units of price
 * @param strike Option strike, must be same units as fwd
 * @param ttm Time to maturity in years
 * @param ivol Black implied volatility / 100, i.e. percentage / 100
 * @param df Optional discount factor in (0, 1].
 * @param is_call +1 for call, -1 for put.
 * @returns Black option volga.
 */
double black_volga(double fwd, double strike, double ttm, double ivol,
  double df, int is_call) {
  // compute log(F / K) total volatility
  double log_m, tot_vol;
  log_m = log(fwd / strike);
  tot_vol = ivol * sqrt(ttm);
  // compute d1 and d2
  double d1, d2;
  d1 = log_m / tot_vol + tot_vol / 2;
  d2 = d1 - tot_vol;
  // compute partial of d1 wrt ivol and partial of d2 wrt ivol
  double d1dvol, d2dvol;
  d1dvol = -log_m / tot_vol / ivol + sqrt(ttm) / 2;
  d2dvol = d1dvol - sqrt(ttm);
  // compute partial of d1dvol wrt ivol
  double dd_dvolvol;
  dd_dvolvol = log_m / tot_vol / pow(ivol, 2);
  // if +1, call, if -1, put, else print error and return NAN
  if (is_call == 1) {
    return df * (fwd * std_normal_pdf(d1) * (dd_dvolvol - d1 * d1dvol) -
      strike * std_normal_pdf(d2) * (dd_dvolvol - d2 * d2dvol));
  }
  else if (is_call == -1) {
    return df * (fwd * std_normal_pdf(-d1) * (dd_dvolvol - d1 * d1dvol) -
      strike * std_normal_pdf(-d2) * (dd_dvolvol - d2 * d2dvol));
  }
  fprintf(stderr, "black_volga: is_call must be +/-1\n");
  return NAN;
}

/**
 * Compute the Bachelier price of a European option.
 * 
 * @note
 * Interest rate options such as Eurodollar options are typically quoted in IMM
 * index points, which have the same units as percentage. In this case, the
 * Bachelier implied volatility is typically in basis points (0.01 of a
 * percentage point), so if price is in IMM index points, the basis point
 * volatility needs to be divided by 100 before being used in the function.
 * 
 * @param fwd Current level of the forward (underlying) in units of price
 * @param strike Option strike, must be same units as fwd
 * @param ttm Time to maturity in years
 * @param ivol Bachelier implied volatility, usually bps / 100 if price is in
 *             IMM index points.
 * @param df Optional discount factor in (0, 1].
 * @param is_call +1 for call, -1 for put.
 * @returns Bachelier price for a European option.
 */
double bachelier_price(double fwd, double strike, double ttm, double ivol,
  double df, int is_call) {
  // print error and return -INFINITY on error if is_call is not +/- 1
  if ((is_call != 1) && (is_call != -1)) {
    fprintf(stderr, "bachelier_price: is_call must be +/-1\n");
    return -INFINITY;
  }
  // compute total volatility
  double tot_vol;
  tot_vol = ivol * sqrt(ttm);
  // compute intrinsic value per unit of total vol
  double std_val;
  std_val = is_call * (fwd - strike) / tot_vol;
  // compute price
  return df * (is_call * (fwd - strike) * std_normal_cdf(std_val) + tot_vol *
    std_normal_pdf(std_val));
}

/**
 * Bachelier option vega.
 * 
 * @note
 * Dummy parameter allows us to use a function pointer in rootfind.c. If you
 * need to call this directly, it is better to use the macro without underscore.
 * 
 * @param fwd Current level of the forward (underlying) in units of price
 * @param strike Option strike, must be same units as fwd
 * @param ttm Time to maturity in years
 * @param ivol Bachelier implied volatility, usually bps / 100 if price is in
 *             IMM index points.
 * @param df Optional discount factor in (0, 1].
 * @param is_call +1 for call, -1 for put. Dummy parameter.
 * @returns Bachelier option vega.
 */
double _bachelier_vega(double fwd, double strike, double ttm, double ivol,
  double df, int is_call) {
  return df * std_normal_pdf((fwd - strike) / (ivol * sqrt(ttm)));
}

/**
 * Bachelier option volga.
 * 
 * @note Dummy parameter makes function pointer in rootfind.c.
 * 
 * @param fwd Current level of the forward (underlying) in units of price
 * @param strike Option strike, must be same units as fwd
 * @param ttm Time to maturity in years
 * @param ivol Bachelier implied volatility, usually bps / 100 if price is in
 *             IMM index points.
 * @param df Optional discount factor in (0, 1].
 * @param is_call +1 for call, -1 for put. Dummy parameter.
 * @returns Bachelier option volga.
 */
double _bachelier_volga(double fwd, double strike, double ttm, double ivol,
  double df, int is_call) {
  return bachelier_vega(fwd, strike, ttm, ivol, df) *
    pow((fwd - strike) / (ivol * sqrt(ttm)), 2) / ivol;
}
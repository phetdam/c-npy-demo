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
 * @param fwd Current level of the forward (underlying) in units of price
 * @param strike Option strike, must be same units as fwd
 * @param ttm Time to maturity in years
 * @param ivol Bachelier implied volatility, usually bps / 100 if price is in
 *             IMM index points.
 * @param df Optional discount factor in (0, 1].
 * @returns Bachelier option vega.
 */
double bachelier_vega(double fwd, double strike, double ttm, double ivol,
  double df) {
  return df * std_normal_pdf((fwd - strike) / (ivol * sqrt(ttm)));
}

/**
 * Bachelier option volga.
 * 
 * @param fwd Current level of the forward (underlying) in units of price
 * @param strike Option strike, must be same units as fwd
 * @param ttm Time to maturity in years
 * @param ivol Bachelier implied volatility, usually bps / 100 if price is in
 *             IMM index points.
 * @param df Optional discount factor in (0, 1].
 * @returns Bachelier option volga.
 */
double bachelier_volga(double fwd, double strike, double ttm, double ivol,
  double df) {
  return bachelier_vega(fwd, strike, ttm, ivol, df) *
    pow((fwd - strike) / (ivol * sqrt(ttm)), 2) / ivol;
}

/**
 * Create new vol_obj_args struct using malloc.
 * 
 * @note Use the macro to get a struct as an auto variable, which is faster.
 * 
 * @param price The true option price.
 * @param fwd Current level of the forward (underlying) in units of price
 * @param strike Option strike, must be same units as fwd
 * @param ttm Time to maturity in years
 * @param df Optional discount factor in (0, 1].
 * @param is_call +1 for call, -1 for put.
 * @returns vol_obj_args *
 */
vol_obj_args *vol_obj_args_mnew(double price, double fwd, double strike,
  double ttm, double df, int is_call) {
  vol_obj_args *out;
  out = (vol_obj_args *) malloc(sizeof(vol_obj_args));
  out->price = price;
  out->fwd = fwd;
  out->strike = strike;
  out->ttm = ttm;
  out->df = df;
  out->is_call = is_call;
  return out;
}

/**
 * Objective function to solve for Black implied vol, the function's root.
 * 
 * @param ivol Level of Black implied vol used to guess the option price.
 * @param args Pointer to vol_obj_args struct
 * @returns Difference between actual and guessed Black price.
 */
double black_vol_obj(double ivol, void *_args) {
  vol_obj_args *args;
  args = (vol_obj_args *) _args;
  // get a guess price
  double guess;
  guess = black_price(args->fwd, args->strike, args->ttm, ivol, args->df,
    args->is_call);
  // return difference (derivative of obj has same sign as derivative of guess)
  return guess - args->price;
}

/**
 * Objective function to solve for Bachelier implied vol, the function's root.
 * 
 * @param ivol Level of Bachelier implied vol used to guess the option's price.
 * @param args Pointer to vol_obj_args struct
 * @returns Difference between actual and guess Bachelier price.
 */
double bachelier_vol_obj(double ivol, void *_args) {
  vol_obj_args *args;
  args = (vol_obj_args *) _args;
  // get a guess price
  double guess;
  guess = bachelier_price(args->fwd, args->strike, args->ttm, ivol, args->df,
    args->is_call);
  // return difference (derivative of obj has same sign as derivative of guess)
  return guess - args->price;
}

/**
 * First derivative of Black objective function.
 * 
 * @param ivol Level of Black implied vol used to guess the option price.
 * @param args Pointer to vol_obj_args struct
 * @returns First derivative of guessed Black price.
 */
double black_vol_obj_d1(double ivol, void *_args) {
  vol_obj_args *args;
  args = (vol_obj_args *) _args;
  return black_vega(args->fwd, args->strike, args->ttm, ivol, args->df,
    args->is_call);
}

/**
 * Second derivative of Black objective function.
 * 
 * @param ivol Level of Black implied vol used to guess the option price.
 * @param args Pointer to vol_obj_args struct
 * @returns Second derivative of guessed Black price.
 */
double black_vol_obj_d2(double ivol, void *_args) {
  vol_obj_args *args;
  args = (vol_obj_args *) _args;
  return black_volga(args->fwd, args->strike, args->ttm, ivol, args->df,
    args->is_call);
}

/**
 * First derivative of Bachelier objective function.
 * 
 * @param ivol Level of Bachelier implied vol used to guess the option price.
 * @param args Pointer to vol_obj_args struct
 * @returns First derivative of guessed Bachelier price.
 */
double bachelier_vol_obj_d1(double ivol, void *_args) {
  vol_obj_args *args;
  args = (vol_obj_args *) _args;
  return bachelier_vega(args->fwd, args->strike, args->ttm, ivol, args->df);
}

/**
 * Second derivative of Bachelier objective function.
 * 
 * @param ivol Level of Bachelier implied vol used to guess the option price.
 * @param args Pointer to vol_obj_args struct
 * @returns Second derivative of guessed Bachelier price.
 */
double bachelier_vol_obj_d2(double ivol, void *_args) {
  vol_obj_args *args;
  args = (vol_obj_args *) _args;
  return bachelier_volga(args->fwd, args->strike, args->ttm, ivol, args->df);
}
/**
 * @file implied_vol.c
 * @brief Code to get Black and Bachelier implied vol using Newton's method.
 * @note Successfull compilation requires glibc.
 */

#include <stdio.h>
#include <math.h>

#include "implied_vol.h"

/**
 * Gaussian probability density function.
 * 
 * @param x A scalar point
 * @param mu Mean of corresponding normal distribution.
 * @param sigma Standard deviation of the corresponding normal distribution.
 * @returns The density at x
 */
double normal_pdf(double x, double mu, double sigma) {
  return exp(pow((x - mu) / sigma, 2) / -2) / (sigma * sqrt(2 * M_PI));
}

/**
 * Approximation of Gaussian cumulative density function.
 * 
 * Uses the Zelen & Severo (1964) approximation found in Abramowitz and Stegun's
 * book @a Handbook of mathematical functions with formulas, graphs, and
 * mathematical tables @a.
 * 
 * @param x A scalar point
 * @param mu Mean of corresponding normal distribution.
 * @param sigma Standard deviation of the corresponding normal distribution.
 * @returns The cumulative density at x
 */
double normal_cdf(double x, double mu, double sigma) {
  double t;
  t = 1 / (1 + 0.2316419 * x);
  return 1 - normal_pdf(x, mu, sigma) * (0.31938153 * t - 0.356563782 * 
    pow(t, 2) + 1.781477937 * pow(t, 3) - 1.821255978 * pow(t, 4) +
    1.330274429 * pow(t, 5));
}

/**
 * Compute the Black price of a European option.
 * 
 * @param fwd Current level of the forward (underlying) in units of price
 * @param strike Option strike, must be same units as fwd
 * @param ttm Time to maturity in years
 * @param ivol Black implied volatility / 100, i.e. percentage / 100
 * @param df Optional discount factor in (0, 1].
 * @param is_call +1 for call, -1 for put.
 * @returns European option price.
 */
double black_price(double fwd, double strike, double ttm, double ivol,
  double df, int is_call) {
  /* compute total variance and total volatility, pow(ivol, 2) * ttm and
  ivol * sqrt(ttm) respectively */
  double tot_var, tot_vol;
  tot_var = pow(ivol, 2) * ttm;
  tot_vol = ivol * sqrt(ttm);
  // compute log of fwd / strike
  double log_m;
  log_m = log(fwd / strike);
  // compute d1 and d2
  double d1, d2;
  d1 = (log_m + tot_var / 2) / tot_vol;
  d2 = (log_m - tot_var / 2) / tot_vol;
  // compute the price. call if +1, put if -1
  if (is_call == 1) {
    return df * (fwd * std_normal_cdf(d1) - strike * std_normal_cdf(d2));
  }
  else if (is_call == -1) {
    return df * (strike * std_normal_cdf(-d2) - fwd * std_normal_cdf(-d1));
  }
  // else print error and return -HUGE_VAL if is_call is incorrect
  fprintf(stderr, "black_price: is_call must be +/-1\n");
  return -INFINITY;
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
 * @returns European option price.
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
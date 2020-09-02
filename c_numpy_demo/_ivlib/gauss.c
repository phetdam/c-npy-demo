/**
 * @file gauss.c
 * @brief Normal distribution-related functions.
 */

#include <math.h>

#include "gauss.h"

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
 * Uses the included erf function approximation to compute the normal cdf.
 * 
 * @param x A scalar point
 * @param mu Mean of corresponding normal distribution.
 * @param sigma Standard deviation of the corresponding normal distribution.
 * @returns The cumulative density at x
 */
double normal_cdf(double x, double mu, double sigma) {
  return 0.5 + erf((x - mu) / sigma * M_SQRT1_2) / 2;
}
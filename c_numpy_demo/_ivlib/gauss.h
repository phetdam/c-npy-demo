/**
 * @file gauss.h
 * @brief Header file for functions defined in gauss.c.
 * @note Macros are for convenience.
 */

#ifndef GAUSS_H
#define GAUSS_H

double normal_pdf(double x, double mu, double sigma);
#define std_normal_pdf(x) normal_pdf(x, 0, 1)
double normal_cdf(double x, double mu, double sigma);
#define std_normal_cdf(x) normal_cdf(x, 0, 1)

#endif /* GAUSS_H */
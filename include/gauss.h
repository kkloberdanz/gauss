#ifndef GAUSS_H
#define GAUSS_H

#include <stddef.h>

double gauss_dot_f64(double *a, double *b, size_t size);
void gauss_add_f64(double *dst, double *a, double *b, size_t size);

#endif /* GAUSS_H */

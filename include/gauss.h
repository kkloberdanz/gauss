#ifndef GAUSS_H
#define GAUSS_H

#include <stddef.h>

void gauss_init(void);
void gauss_close(void);

double gauss_vec_dot_f64(double *a, double *b, size_t size);
double gauss_vec_norm_f64(double *a, size_t size);
double gauss_vec_sumabs_f64(double *a, size_t size);

size_t gauss_vec_index_max_f64(double *a, size_t size);

void gauss_vec_add_f64(double *dst, double *a, double *b, size_t size);
void gauss_vec_mul_f64(double *dst, double *a, double *b, size_t size);

#endif /* GAUSS_H */

#ifndef GAUSS_BLAS_LEVEL1_H
#define GAUSS_BLAS_LEVEL1_H

#include <stddef.h>

#include "util.h"

void gauss_vec_scale_f64(double *dst, double *a, size_t size, double scalar);

double gauss_vec_dot_f64(double *a, double *b, size_t size);

double gauss_vec_l2norm_f64(double *a, size_t size);

double gauss_vec_l1norm_f64(double *a, size_t size);

size_t gauss_vec_index_max_f64(double *a, size_t size);

gauss_Error gauss_vec_dot_f32(float *a, float *b, size_t size, float *out);

#endif /* GAUSS_BLAS_LEVEL1_H */

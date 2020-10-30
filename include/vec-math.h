#ifndef SIMD_MATH_X86_64
#define SIMD_MATH_X86_64

#include <stdio.h>

#ifdef __AVX512F__
#    define SIMD_ALIGN __attribute__ ((aligned (128)))
#    define SIMD_ALIGN_SIZE 128
#else
#    ifdef __AVX__
#        define SIMD_ALIGN __attribute__ ((aligned (64)))
#        define SIMD_ALIGN_SIZE 64
#    else
#        ifdef __SSE2__
#            define SIMD_ALIGN __attribute__ ((aligned (32)))
#            define SIMD_ALIGN_SIZE 32
#        else
#           define UNKNOWN_SIMD
#        endif
#    endif
#endif

void gauss_add_double_array(
    double *result,
    const double *a,
    const double *b,
    size_t size
);

void gauss_div_double_array(
    double *result,
    const double *a,
    const double *b,
    size_t size
);

void gauss_mul_double_array(
    double *result,
    const double *a,
    const double *b,
    size_t size
);

void gauss_add_double_array(
    double *result,
    const double *a,
    const double *b,
    size_t size
);

void gauss_sub_double_scalar(
    double *result,
    const double *a,
    const double b,
    size_t size
);

void gauss_sqrt_double_array(double *result, const double *a, size_t size);

void gauss_sqrt_float_array(float *result, const float *a, size_t size);

void *gauss_simd_alloc(size_t size);

double gauss_double_array_at(const double *arr, size_t i);

void gauss_set_double_array_at(double *arr, size_t i, double value);

double gauss_mean_double_array(const double *a, const size_t size);

double gauss_vec_sum_f64(double *a, size_t size);

#endif /* SIMD_MATH_X86_64 */

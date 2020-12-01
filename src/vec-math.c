#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef __AVX__
    #include <emmintrin.h>
#endif

#ifdef __SSE__
    #include <immintrin.h>
#endif

#include "vec-math.h"

void gauss_sqrt_float_array(
    float *result,
    const float *a,
    size_t size
) {
    size_t i = 0;

#ifdef __AVX512F__
    /* AVX-512 */
    for(; i < (size & ~0xf); i += 16) {
        const __m512 kA8 = _mm512_load_ps(&a[i]);
        const __m512 kRes = _mm512_sqrt_ps(kA8);
        _mm512_stream_ps(&result[i], kRes);
    }
#endif

#ifdef __AVX__
    /* AVX loop */
    for (; i < (size & ~0x7); i += 8) {
        const __m256 kA4 = _mm256_load_ps(&a[i]);
        const __m256 kRes = _mm256_sqrt_ps(kA4);
        _mm256_stream_ps(&result[i], kRes);
    }
#endif

#ifdef __SSE2__
    /* SSE2 loop */
    for (; i < (size & ~0x3); i += 4) {
        const __m128 kA2 = _mm_load_ps(&a[i]);
        const __m128 kRes = _mm_sqrt_ps(kA2);
        _mm_stream_ps(&result[i], kRes);
    }
#endif

    /* Serial loop */
    for(; i < size; i++) {
        result[i] = sqrt(a[i]);
    }
}

void gauss_sqrt_double_array(
    double *result,
    const double *a,
    size_t size
) {
    size_t i = 0;
#ifdef __AVX512F__
    /* AVX-512 */
    for(; i < (size & ~0x7); i += 8) {
        const __m512d kA8 = _mm512_load_pd(&a[i]);
        const __m512d kRes = _mm512_sqrt_pd(kA8);
        _mm512_stream_pd(&result[i], kRes);
    }
#endif

#ifdef __AVX__
    /* AVX loop */
    for (; i < (size & ~0x3); i += 4) {
        const __m256d kA4 = _mm256_load_pd(&a[i]);
        const __m256d kRes = _mm256_sqrt_pd(kA4);
        _mm256_stream_pd(&result[i], kRes);
    }
#endif

#ifdef __SSE2__
    /* SSE2 loop */
    for (; i < (size & ~0x1); i += 2) {
        const __m128d kA2 = _mm_load_pd(&a[i]);
        const __m128d kRes = _mm_sqrt_pd(kA2);
        _mm_stream_pd(&result[i], kRes);
    }
#endif
    /* Serial loop */
    for(; i < size; i++) {
        result[i] = sqrt(a[i]);
    }
}

void gauss_div_double_array(
    double *result,
    const double *a,
    const double *b,
    size_t size
) {
    size_t i = 0;

#ifdef __AVX512F__
    /* AVX-512 */
    for(; i < (size & ~0x7); i += 8) {
        const __m512d kA8 = _mm512_load_pd(&a[i]);
        const __m512d kB8 = _mm512_load_pd(&b[i]);
        const __m512d kRes = _mm512_div_pd(kA8, kB8);
        _mm512_stream_pd(&result[i], kRes);
    }
#endif

#ifdef __AVX__
    /* AVX loop */
    for (; i < (size & ~0x3); i += 4) {
        const __m256d kA4 = _mm256_load_pd(&a[i]);
        const __m256d kB4 = _mm256_load_pd(&b[i]);
        const __m256d kRes = _mm256_div_pd(kA4, kB4);
        _mm256_stream_pd(&result[i], kRes);
    }
#endif

#ifdef __SSE2__
    /* SSE2 loop */
    for (; i < (size & ~0x1); i += 2) {
        const __m128d kA2 = _mm_load_pd(&a[i]);
        const __m128d kB2 = _mm_load_pd(&b[i]);
        const __m128d kRes = _mm_div_pd(kA2, kB2);
        _mm_stream_pd(&result[i], kRes);
    }
#endif

    /* Serial loop */
    for(; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

void gauss_floordiv_double_array(
    double *result,
    const double *a,
    const double *b,
    size_t size
) {
    size_t i = 0;

#ifdef __AVX512F__
    /* AVX-512 */
    for(; i < (size & ~0x7); i += 8) {
        const __m512d kA8 = _mm512_load_pd(&a[i]);
        const __m512d kB8 = _mm512_load_pd(&b[i]);
        const __m512d kRes = _mm512_div_pd(kA8, kB8);
        const __m512d rounded_down = _mm512_floor_pd(kRes);
        _mm512_stream_pd(&result[i], rounded_down);
    }
#endif

#ifdef __AVX__
    /* AVX loop */
    for (; i < (size & ~0x3); i += 4) {
        const __m256d kA4 = _mm256_load_pd(&a[i]);
        const __m256d kB4 = _mm256_load_pd(&b[i]);
        const __m256d kRes = _mm256_div_pd(kA4, kB4);
        const __m256d rounded_down = _mm256_floor_pd(kRes);
        _mm256_stream_pd(&result[i], rounded_down);
    }
#endif

#ifdef __SSE4_1__
    /* SSE2 loop */
    for (; i < (size & ~0x1); i += 2) {
        const __m128d kA2 = _mm_load_pd(&a[i]);
        const __m128d kB2 = _mm_load_pd(&b[i]);
        const __m128d kRes = _mm_div_pd(kA2, kB2);
        const __m128d rounded_down = _mm_floor_pd(kRes);
        _mm_stream_pd(&result[i], rounded_down);
    }
#endif

    /* Serial loop */
    for(; i < size; i++) {
        result[i] = floor(a[i] + b[i]);
    }
}

void gauss_mul_double_array(
    double *result,
    const double *a,
    const double *b,
    size_t size
) {
    size_t i = 0;

#ifdef __AVX512F__
    /* AVX-512 */
    for(; i < (size & ~0x7); i += 8) {
        const __m512d kA8 = _mm512_load_pd(&a[i]);
        const __m512d kB8 = _mm512_load_pd(&b[i]);
        const __m512d kRes = _mm512_mul_pd(kA8, kB8);
        _mm512_stream_pd(&result[i], kRes);
    }
#endif

#ifdef __AVX__
    /* AVX loop */
    for (; i < (size & ~0x3); i += 4) {
        const __m256d kA4 = _mm256_load_pd(&a[i]);
        const __m256d kB4 = _mm256_load_pd(&b[i]);
        const __m256d kRes = _mm256_mul_pd(kA4, kB4);
        _mm256_stream_pd(&result[i], kRes);
    }
#endif

#ifdef __SSE2__
    /* SSE2 loop */
    for (; i < (size & ~0x1); i += 2) {
        const __m128d kA2 = _mm_load_pd(&a[i]);
        const __m128d kB2 = _mm_load_pd(&b[i]);
        const __m128d kRes = _mm_mul_pd(kA2, kB2);
        _mm_stream_pd(&result[i], kRes);
    }
#endif
    /* Serial loop */
    for(; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

void gauss_add_double_array(
    double *result,
    const double *a,
    const double *b,
    size_t size
) {
    size_t i = 0;

#ifdef __AVX512F__
    /* AVX-512 */
    for(; i < (size & ~0x7); i += 8) {
        const __m512d kA8 = _mm512_load_pd(&a[i]);
        const __m512d kB8 = _mm512_load_pd(&b[i]);
        const __m512d kRes = _mm512_add_pd(kA8, kB8);
        _mm512_stream_pd(&result[i], kRes);
    }
#endif

#ifdef __AVX__
    /* AVX loop */
    for (; i < (size & ~0x3); i += 4) {
        const __m256d kA4 = _mm256_load_pd(&a[i]);
        const __m256d kB4 = _mm256_load_pd(&b[i]);
        const __m256d kRes = _mm256_add_pd(kA4, kB4);
        _mm256_stream_pd(&result[i], kRes);
    }
#endif

#ifdef __SSE2__
    /* SSE2 loop */
    for (; i < (size & ~0x1); i += 2) {
        const __m128d kA2 = _mm_load_pd(&a[i]);
        const __m128d kB2 = _mm_load_pd(&b[i]);
        const __m128d kRes = _mm_add_pd(kA2, kB2);
        _mm_stream_pd(&result[i], kRes);
    }
#endif
    /* Serial loop */
    for(; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

void gauss_sub_double_array(
    double *result,
    const double *a,
    const double *b,
    size_t size
) {
    size_t i = 0;

#ifdef __AVX512F__
    /* AVX-512 */
    for(; i < (size & ~0x7); i += 8) {
        const __m512d kA8 = _mm512_load_pd(&a[i]);
        const __m512d kB8 = _mm512_load_pd(&b[i]);
        const __m512d kRes = _mm512_sub_pd(kA8, kB8);
        _mm512_stream_pd(&result[i], kRes);
    }
#endif

#ifdef __AVX__
    /* AVX loop */
    for (; i < (size & ~0x3); i += 4) {
        const __m256d kA4 = _mm256_load_pd(&a[i]);
        const __m256d kB4 = _mm256_load_pd(&b[i]);
        const __m256d kRes = _mm256_sub_pd(kA4, kB4);
        _mm256_stream_pd(&result[i], kRes);
    }
#endif

#ifdef __SSE2__
    /* SSE2 loop */
    for (; i < (size & ~0x1); i += 2) {
        const __m128d kA2 = _mm_load_pd(&a[i]);
        const __m128d kB2 = _mm_load_pd(&b[i]);
        const __m128d kRes = _mm_sub_pd(kA2, kB2);
        _mm_stream_pd(&result[i], kRes);
    }
#endif
    /* Serial loop */
    for(; i < size; i++) {
        result[i] = a[i] - b[i];
    }
}

void gauss_add_double_scalar(
    double *result,
    const double *a,
    const double b,
    size_t size
) {
    size_t i = 0;
    for (; i < size; i++) {
        result[i] = a[i] + b;
    }
}

void gauss_sub_double_scalar(
    double *result,
    const double *a,
    const double b,
    size_t size
) {
    size_t i = 0;
    for (; i < size; i++) {
        result[i] = a[i] - b;
    }
}

void gauss_floordiv_double_scalar(
    double *result,
    const double *a,
    const double b,
    size_t size
) {
    size_t i = 0;
    for (; i < size; i++) {
        result[i] = floor(a[i] / b);
    }
}

void gauss_div_double_scalar(
    double *result,
    const double *a,
    const double b,
    size_t size
) {
    size_t i = 0;
    for (; i < size; i++) {
        result[i] = a[i] / b;
    }
}

static int compare_double(const void *a, const void *b) {
    if (*(double *)a < *(double *)b) {
        return -1;
    } else if (*(double *)a > *(double *)b) {
        return 1;
    } else {
        return 0;
    }
}

static char is_even(size_t n) {
    return (n & 1) == 0;
}

double gauss_median_double_array(
    double *scratch,
    const double *a,
    const size_t size
) {
    double med;

    memcpy(scratch, a, sizeof(double) * size);
    qsort(scratch, size, sizeof(double), compare_double);
    if (is_even(size)) {
        med = scratch[size / 2];
    } else {
        size_t idx = size / 2;
        med = (scratch[idx] + scratch[idx + 1]) / 2;
    }
    return med;
}

void gauss_vec_add_f64(double *dst, double *a, double *b, size_t size) {
    size_t i;
    for (i = 0; i < size; i++) {
        dst[i] = a[i] + b[i];
    }
}

void gauss_vec_mul_f64(double *dst, double *a, double *b, size_t size) {
    size_t i;
    for (i = 0; i < size; i++) {
        dst[i] = a[i] * b[i];
    }
}
/*
double gauss_standard_deviation_f64(double *a, size_t size) {
    return sqrt(gauss_variance_f64(a, size));
}
*/

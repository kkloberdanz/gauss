#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <dlfcn.h>
#include <string.h>

#include "handler.h"
#include "util.h"
#include "blas-level1.h"
#include "opencl.h"

void gauss_vec_scale_f64(double *dst, double *a, size_t size, double scalar) {
    if (has_openblas) {
        memcpy(dst, a, size * sizeof(double));
        _gauss_cblas_dscal(size, scalar, dst, 1);
    } else {
        size_t i;
        for (i = 0; i < size; i++) {
            dst[i] = a[i] * scalar;
        }
    }
}

void gauss_vec_scale_f32(float *dst, float *a, size_t size, float scalar) {
    if (has_openblas) {
        memcpy(dst, a, size * sizeof(float));
        _gauss_cblas_sscal(size, scalar, dst, 1);
    } else {
        size_t i;
        for (i = 0; i < size; i++) {
            dst[i] = a[i] * scalar;
        }
    }
}

gauss_Error gauss_vec_dot_cl_float(float *a, float *b, size_t size, float *out) {
    gauss_Error status_code = gauss_clblas_sdot(
        size,
        a,
        1,
        b,
        1,
        out
    );
    return status_code;
}

float gauss_vec_dot_f32(float *a, float *b, size_t size) {
    float acc = 0.0;
    size_t i;

    if (has_openblas) {
        acc = _gauss_cblas_sdot(size, a, 1, b, 1);
    } else {
        /* if nothing better exists, brute force it */
        for (i = 0; i < size; i++) {
            acc += a[i] * b[i];
        }
    }
    return acc;
}

double gauss_vec_dot_f64(double *a, double *b, size_t size) {
    double acc = 0.0;
    size_t i;

    if (has_openblas) {
        acc = _gauss_cblas_ddot(size, a, 1, b, 1);
    } else {
        /* if nothing better exists, brute force it */
        for (i = 0; i < size; i++) {
            acc += a[i] * b[i];
        }
    }
    return acc;
}

double gauss_vec_l2norm_f64(double *a, size_t size) {
    size_t i;
    double norm = 0.0;

    if (has_openblas) {
        norm = _gauss_cblas_dnrm2(size, a, 1);
    } else {
        /* if nothing better exists, brute force it */
        for (i = 0; i < size; i++) {
            norm += a[i] * a[i];
        }
        norm = sqrt(norm);
    }
    return norm;
}

double gauss_vec_l1norm_f64(double *a, size_t size) {
    size_t i;
    double acc = 0.0;

    if (has_openblas) {
        acc = _gauss_cblas_dasum(size, a, 1);
    } else {
        /* if nothing better exists, brute force it */
        for (i = 0; i < size; i++) {
            acc += fabs(a[i]);
        }
    }
    return acc;
}

size_t gauss_vec_index_max_f64(double *a, size_t size) {
    size_t i;
    size_t idx;

    if (has_openblas) {
        idx = _gauss_cblas_idamax(size, a, 1);
    } else {
        /* if nothing better exists, brute force it */
        idx = a[0];
        for (i = 0; i < size; i++) {
            idx = a[i] > idx ? a[i] : idx;
        }
    }
    return idx;
}

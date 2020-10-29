#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <dlfcn.h>
#include <cblas.h>
#include <string.h>

#include "../include/gauss.h"
#include "simd-math-x86_64.h"

static bool has_openblas = false;
/* TODO: use these to provide best implementation for the data given
static bool has_mkl = false;
static bool has_opencl = false;
static bool has_cuda = false;
*/

void *openblas_handle = NULL;

double (*_gauss_cblas_ddot)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST double *x,
    OPENBLAS_CONST blasint incx,
    OPENBLAS_CONST double *y,
    OPENBLAS_CONST blasint incy
);

double (*_gauss_cblas_dnrm2)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST double *x,
    OPENBLAS_CONST blasint incx
);

double (*_gauss_cblas_dasum)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST double *x,
    OPENBLAS_CONST blasint incx
);

size_t (*_gauss_cblas_idamax)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST double *x,
    OPENBLAS_CONST blasint incx
);

size_t (*_gauss_cblas_dscal)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST double a,
    OPENBLAS_CONST double *x,
    OPENBLAS_CONST blasint incx
);

void gauss_init(void) {
    openblas_handle = dlopen("libopenblas.so", RTLD_LAZY|RTLD_GLOBAL);
    if (openblas_handle) {
        has_openblas = true;
        _gauss_cblas_ddot = dlsym(openblas_handle, "cblas_ddot");
        _gauss_cblas_dnrm2 = dlsym(openblas_handle, "cblas_dnrm2");
        _gauss_cblas_dasum = dlsym(openblas_handle, "cblas_dasum");
        _gauss_cblas_idamax = dlsym(openblas_handle, "cblas_idamax");
        _gauss_cblas_dscal = dlsym(openblas_handle, "cblas_dscal");
    }
}

void gauss_close(void) {
    if (openblas_handle) {
        dlclose(openblas_handle);
    }
}

void gauss_vec_scale_f64(double *dst, double *a, size_t size, double scalar) {
    memcpy(dst, a, size * sizeof(double));
    if (has_openblas) {
        _gauss_cblas_dscal(size, scalar, dst, 1);
    } else {
        size_t i;
        for (i = 0; i < size; i++) {
            dst[i] = a[i] * scalar;
        }
    }
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

double gauss_vec_sum_f64(double *a, size_t size) {
    size_t i;
    double acc = 0.0;

    for (i = 0; i < size; i++) {
        acc += a[i];
    }
    return acc;
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
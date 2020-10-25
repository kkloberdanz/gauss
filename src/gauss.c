#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include <dlfcn.h>

#include <cblas.h>

#include "../include/gauss.h"

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

void gauss_init(void) {
    openblas_handle = dlopen("libopenblas.so", RTLD_LAZY|RTLD_GLOBAL);
    if (openblas_handle) {
        has_openblas = true;
        _gauss_cblas_ddot = (double (*)(
            OPENBLAS_CONST blasint n,
            OPENBLAS_CONST double *x,
            OPENBLAS_CONST blasint incx,
            OPENBLAS_CONST double *y,
            OPENBLAS_CONST blasint incy)
        )dlsym(openblas_handle, "cblas_ddot");

        _gauss_cblas_dnrm2 = (double (*)(
            OPENBLAS_CONST blasint n,
            OPENBLAS_CONST double *x,
            OPENBLAS_CONST blasint incx
        )
        )dlsym(openblas_handle, "cblas_dnrm2");

        _gauss_cblas_dasum = (double (*)(
            OPENBLAS_CONST blasint n,
            OPENBLAS_CONST double *x,
            OPENBLAS_CONST blasint incx
        )
        )dlsym(openblas_handle, "cblas_dasum");
    }
}

void gauss_close(void) {
    if (openblas_handle) {
        dlclose(openblas_handle);
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

double gauss_vec_norm_f64(double *a, size_t size) {
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

double gauss_vec_sumabs_f64(double *a, size_t size) {
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

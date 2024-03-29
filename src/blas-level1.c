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

gauss_Error gauss_vec_dot(gauss_Mem *a, gauss_Mem *b, void *out) {
    gauss_Error error = gauss_OK;

    if (a->kind != b->kind) {
        return gauss_MISMATCHED_TYPES;
    }

    if (a->nmemb != b->nmemb) {
        return gauss_MISMATCHED_DIMENSIONS;
    }

    switch (a->kind) {
        case gauss_FLOAT:
            *(float *)out = gauss_vec_dot_f32(
                a->data.flt,
                b->data.flt,
                a->nmemb
            );
            break;

        case gauss_DOUBLE:
            *(double *)out = gauss_vec_dot_f64(
                a->data.dbl,
                b->data.dbl,
                a->nmemb
            );
            break;

        case gauss_CL_FLOAT: {
            float cl_out = 0.0;
            error = gauss_clblas_sdot(
                a->nmemb,
                a,
                1,
                b,
                1,
                &cl_out /* result from dot product */
            );
            if (!error) {
                *(float *)out = cl_out;
            }
            break;
        }
    }

    return error;
}

double gauss_vec_sum_f64(double *a, size_t size) {
    size_t i;
    double acc = 0.0;

    for (i = 0; i < size; i++) {
        acc += a[i];
    }
    return acc;
}

float gauss_vec_sum_f32(float *a, size_t size) {
    size_t i;
    float acc = 0.0;

    for (i = 0; i < size; i++) {
        acc += a[i];
    }
    return acc;
}

gauss_Error gauss_vec_sum(gauss_Mem *a, void *out) {
    gauss_Error error = gauss_OK;

    switch (a->kind) {
        case gauss_FLOAT:
            *(float *)out = gauss_vec_sum_f32(
                a->data.flt,
                a->nmemb
            );
            break;

        case gauss_DOUBLE:
            *(double *)out = gauss_vec_sum_f64(
                a->data.dbl,
                a->nmemb
            );
            break;

        case gauss_CL_FLOAT:
            /* TODO: implement opencl sum */
            break;
    }

    return error;
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

float gauss_vec_l1norm_f32(float *a, size_t size) {
    size_t i;
    float acc = 0.0;

    if (has_openblas) {
        acc = _gauss_cblas_sasum(size, a, 1);
    } else {
        /* if nothing better exists, brute force it */
        for (i = 0; i < size; i++) {
            acc += fabs(a[i]);
        }
    }
    return acc;
}

gauss_Error gauss_vec_l1norm(gauss_Mem *a, void *out) {
    gauss_Error error = gauss_OK;

    switch (a->kind) {
        case gauss_FLOAT:
            *(float *)out = gauss_vec_l1norm_f32(
                a->data.flt,
                a->nmemb
            );
            break;

        case gauss_DOUBLE:
            *(double *)out = gauss_vec_l1norm_f64(
                a->data.dbl,
                a->nmemb
            );
            break;

        case gauss_CL_FLOAT:
            /* TODO: implement opencl l1norm */
            break;
    }

    return error;
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

float gauss_vec_l2norm_f32(float *a, size_t size) {
    size_t i;
    float norm = 0.0;

    if (has_openblas) {
        norm = _gauss_cblas_snrm2(size, a, 1);
    } else {
        /* if nothing better exists, brute force it */
        for (i = 0; i < size; i++) {
            norm += a[i] * a[i];
        }
        norm = sqrt(norm);
    }
    return norm;
}

gauss_Error gauss_vec_l2norm(gauss_Mem *a, void *out) {
    gauss_Error error = gauss_OK;
    switch (a->kind) {
        case gauss_FLOAT:
            *(float *)out = gauss_vec_l2norm_f32(a->data.flt, a->nmemb);
            break;

        case gauss_DOUBLE:
            *(double *)out = gauss_vec_l2norm_f64(a->data.dbl, a->nmemb);
            break;

        case gauss_CL_FLOAT: {
            float cl_out = 0.0;
            error = gauss_clblas_snrm2(a, &cl_out);
            if (!error) {
                *(float *)out = cl_out;
            }
            break;
        }
    }
    return error;
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

size_t gauss_vec_index_max_f32(float *a, size_t size) {
    size_t i;
    size_t idx;

    if (has_openblas) {
        idx = _gauss_cblas_isamax(size, a, 1);
    } else {
        /* if nothing better exists, brute force it */
        idx = a[0];
        for (i = 0; i < size; i++) {
            idx = a[i] > idx ? a[i] : idx;
        }
    }
    return idx;
}

size_t gauss_vec_argmax(gauss_Mem *a, void *out) {
    gauss_Error error = gauss_OK;
    switch (a->kind) {
        case gauss_FLOAT:
            *(size_t *)out = gauss_vec_index_max_f32(a->data.flt, a->nmemb);
            break;

        case gauss_DOUBLE:
            *(size_t *)out = gauss_vec_index_max_f64(a->data.dbl, a->nmemb);
            break;

        case gauss_CL_FLOAT:
            /* TODO: implement for OpenCL */
            break;
    }
    return error;
}

gauss_Error gauss_vec_argmin(const gauss_Mem *a, size_t *out) {
    size_t i;
    gauss_Error error = gauss_OK;
    size_t min_i = 0;

    switch (a->kind) {
        case gauss_FLOAT: {
            float *arr = a->data.flt;
            float min_element = arr[0];
            for (i = 0; i < a->nmemb; i++) {
                if (arr[i] < min_element) {
                    min_element = arr[i];
                    min_i = i;
                }
            }
            *out = min_i;
            break;
        }

        case gauss_DOUBLE: {
            double *arr = a->data.dbl;
            double min_element = arr[0];
            for (i = 0; i < a->nmemb; i++) {
                if (arr[i] < min_element) {
                    min_element = arr[i];
                    min_i = i;
                }
            }
            *out = min_i;
            break;
        }

        case gauss_CL_FLOAT:
            /* TODO */
            break;
    }
    return error;
}

gauss_Error gauss_vec_scale(
    gauss_Mem *dst,
    gauss_Mem *a,
    size_t size,
    void *scalar
) {
    if (dst->kind != a->kind) {
        fprintf(
            stderr,
            "mismatched types on gauss_vec_scale, %d != %d",
            dst->kind,
            a->kind
        );
    }

    switch (dst->kind) {
        case gauss_DOUBLE:
            gauss_vec_scale_f64(
                dst->data.dbl,
                a->data.dbl,
                size,
                *(double *)scalar
            );
            break;

        case gauss_FLOAT:
            gauss_vec_scale_f32(
                dst->data.flt,
                a->data.flt,
                size,
                *(float *)scalar
            );
            break;

        case gauss_CL_FLOAT:
            break;
    }

    return gauss_OK;
}

gauss_Error gauss_vec_mean(const gauss_Mem *a, void *out) {
    gauss_Error error = gauss_OK;
    size_t i = 0;

    switch (a->kind) {
        case gauss_FLOAT: {
            float acc = 0.0;
            float *arr = a->data.flt;
            for (i = 0; i < a->nmemb; i++) {
                acc += arr[i];
            }
            *(float *)out = acc / a->nmemb;
            break;
        }

        case gauss_DOUBLE: {
            double acc = 0.0;
            double *arr = a->data.dbl;
            for (i = 0; i < a->nmemb; i++) {
                acc += arr[i];
            }
            *(double *)out = acc / a->nmemb;
            break;
        }

        case gauss_CL_FLOAT:
            break;
    }
    return error;
}

gauss_Error gauss_vec_variance(gauss_Mem *a, void *out) {
    gauss_Error error = gauss_OK;

    switch (a->kind) {
        case gauss_DOUBLE: {
            double acc = 0.0;
            size_t i;
            double mean;
            double *arr = a->data.dbl;

            TRY(gauss_vec_mean(a, &mean));

            for (i = 0; i < a->nmemb; i++) {
                acc += (arr[i] - mean) * (arr[i] - mean);
            }
            *(double *)out = acc / a->nmemb;
            break;
        }

        case gauss_FLOAT: {
            float acc = 0.0;
            size_t i;
            float mean;
            float *arr = a->data.flt;

            TRY(gauss_vec_mean(a, &mean));

            for (i = 0; i < a->nmemb; i++) {
                acc += (arr[i] - mean) * (arr[i] - mean);
            }
            *(float *)out = acc / a->nmemb;
            break;
        }

        case gauss_CL_FLOAT:
            /* TODO */
            break;
    }
    return error;
}

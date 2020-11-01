#include <stdio.h>

#include "handler.h"
#include "vec-math.h"
#include "blas-level1.h"

gauss_Error gauss_vec_scale(gauss_Mem *dst, gauss_Mem *a, size_t size, void *scalar) {
    if (dst->kind != a->kind) {
        fprintf(
            stderr,
            "mismatched types on gauss_vec_scale, %d != %d",
            dst->kind,
            a->kind
        );
    }

    switch (dst->kind) {
        case gauss_RECCOMENDED:
            break;

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
                *(double *)scalar
            );
            break;

        case gauss_CL_FLOAT:
            break;
    }

    return gauss_OK;
}

gauss_Error gauss_vec_dot(void *dst, gauss_Mem *a, gauss_Mem *b, size_t size) {
    switch (a->kind) {
        case gauss_RECCOMENDED:
            break;

        case gauss_DOUBLE: {
            double *dbl_dst = dst;
            *dbl_dst = gauss_vec_dot_f64(a->data.dbl, b->data.dbl, size);
            break;
        }

        case gauss_FLOAT: {
            float *flt_dst = dst;
            *flt_dst = gauss_vec_dot_f32(a->data.flt, b->data.flt, size);
            break;
        }

        case gauss_CL_FLOAT:
            break;
    }

    return gauss_OK;
}

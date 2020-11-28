#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>

#include "handler.h"
#include "opencl.h"

bool has_openblas = false;
bool has_clblas = false;
/* TODO: use these to provide best implementation for the data given
static bool has_mkl = false;
static bool has_opencl = false;
static bool has_cuda = false;
*/

void *openblas_handle = NULL;
void *clblas_handle = NULL;

double (*_gauss_cblas_ddot)(
    const blasint n,
    const double *x,
    const blasint incx,
    const double *y,
    const blasint incy
);

double (*_gauss_cblas_dnrm2)(
    const blasint n,
    const double *x,
    const blasint incx
);

double (*_gauss_cblas_dasum)(
    const blasint n,
    const double *x,
    const blasint incx
);

size_t (*_gauss_cblas_idamax)(
    const blasint n,
    const double *x,
    const blasint incx
);

size_t (*_gauss_cblas_dscal)(
    const blasint n,
    const double a,
    const double *x,
    const blasint incx
);

float (*_gauss_cblas_sdot)(
    const blasint n,
    const float *x,
    const blasint incx,
    const float *y,
    const blasint incy
);

float (*_gauss_cblas_snrm2)(
    const blasint n,
    const float *x,
    const blasint incx
);

float (*_gauss_cblas_sasum)(
    const blasint n,
    const float *x,
    const blasint incx
);

size_t (*_gauss_cblas_isamax)(
    const blasint n,
    const float *x,
    const blasint incx
);

size_t (*_gauss_cblas_sscal)(
    const blasint n,
    const float a,
    const float *x,
    const blasint incx
);

void gauss_init(void) {
    openblas_handle = dlopen("libopenblas.so", RTLD_LAZY|RTLD_GLOBAL);
    if (openblas_handle) {
        has_openblas = true;

        /* double implementations */
        _gauss_cblas_ddot = dlsym(openblas_handle, "cblas_ddot");
        _gauss_cblas_dnrm2 = dlsym(openblas_handle, "cblas_dnrm2");
        _gauss_cblas_dasum = dlsym(openblas_handle, "cblas_dasum");
        _gauss_cblas_idamax = dlsym(openblas_handle, "cblas_idamax");
        _gauss_cblas_dscal = dlsym(openblas_handle, "cblas_dscal");

        /* float implementations */
        _gauss_cblas_sdot = dlsym(openblas_handle, "cblas_sdot");
        _gauss_cblas_snrm2 = dlsym(openblas_handle, "cblas_snrm2");
        _gauss_cblas_sasum = dlsym(openblas_handle, "cblas_sasum");
        _gauss_cblas_isamax = dlsym(openblas_handle, "cblas_isamax");
        _gauss_cblas_sscal = dlsym(openblas_handle, "cblas_sscal");
    }

    clblas_handle = dlopen("libclBLAS.so", RTLD_LAZY|RTLD_GLOBAL);
    if (clblas_handle) {
        if (gauss_init_opencl() == gauss_OK) {
            has_clblas = true;
        } else {
            dlclose(clblas_handle);
        }
    }
}

void gauss_close(void) {
    if (openblas_handle) {
        dlclose(openblas_handle);
    }
    if (has_clblas) {
        gauss_close_opencl();
        dlclose(clblas_handle);
    }
}

gauss_Error gauss_set_buffer(gauss_Mem *dst, void *src) {
    gauss_MemKind kind = dst->kind;
    gauss_Error error = gauss_OK;
    switch (kind) {
        case gauss_CL_FLOAT: {
            float *as_float = (float *)src;
            cl_int err = clEnqueueWriteBuffer(
                gauss_get_queue(),
                dst->data.cl_float,
                CL_TRUE,
                0,
                (dst->nmemb * sizeof(cl_float)),
                as_float,
                0,
                NULL,
                NULL
            );
            if (err) {
                error = gauss_CL_ERROR;
            }
            break;
        }

        case gauss_FLOAT:
            memcpy(dst->data.flt, src, sizeof(float) * dst->nmemb);
            break;

        case gauss_DOUBLE:
            memcpy(dst->data.dbl, src, sizeof(double) * dst->nmemb);
            break;
    }
    return error;
}

gauss_MemKind gauss_determine_best_backend(void) {
    if (has_clblas) {
        return gauss_CL_FLOAT;
    } else {
        return gauss_DOUBLE;
    }
}

gauss_Mem *gauss_alloc(size_t nmemb, int kind) {
    gauss_Mem *ptr = malloc(sizeof(gauss_Mem));
    if (!ptr) {
        goto fail;
    }

    if (kind == -1) {
        kind = gauss_determine_best_backend();
    }

    ptr->kind = kind;
    ptr->data.vd = NULL;
    ptr->nmemb = nmemb;
    switch (kind) {
        case gauss_FLOAT:
            ptr->data.flt = gauss_simd_alloc(sizeof(float) * nmemb);
            break;

        case gauss_DOUBLE:
            ptr->data.dbl = gauss_simd_alloc(sizeof(double) * nmemb);
            break;

        case gauss_CL_FLOAT: {
            cl_context ctx = gauss_get_cl_ctx();
            cl_int err = 0;
            cl_mem cl_buf = clCreateBuffer(
                ctx,
                CL_MEM_READ_ONLY,
                (nmemb * sizeof(cl_float)),
                NULL,
                &err
            );
            if (err) {
                goto free_ptr;
            }
            ptr->data.cl_float = cl_buf;
            break;
        }

        default:
            fprintf(stderr, "invalid gauss type: %d\n", kind);
            exit(EXIT_FAILURE);
    }

    if (!ptr->data.vd) {
        goto free_ptr;
    }

    return ptr;

free_ptr:
    free(ptr);
fail:
    return NULL;
}

void gauss_free(gauss_Mem *ptr) {
    if (ptr) {
        switch (ptr->kind) {
            case gauss_CL_FLOAT:
                clReleaseMemObject(ptr->data.cl_float);
                break;

            case gauss_FLOAT:
                free(ptr->data.flt);
                break;

            case gauss_DOUBLE:
                free(ptr->data.dbl);
                break;
        }
        ptr->data.vd = NULL;
        free(ptr);
    }
}

const char *gauss_get_dtype(const gauss_Mem *obj) {
    switch (obj->kind) {
        case gauss_DOUBLE:
            return "double";
        case gauss_FLOAT:
            return "float";
        case gauss_CL_FLOAT:
            return "cl_float";
    }
    return "invalid";
}

const char *gauss_error_to_string(const gauss_Error e) {
    switch (e) {
        case gauss_OK:
            return "ok";
        case gauss_MISMATCHED_TYPES:
            return "mismatched types";
        case gauss_MISMATCHED_DIMENSIONS:
            return "mismatched dimensions";
        case gauss_OUT_OF_MEMORY:
            return "out of memory";
        case gauss_GENERIC_ERROR:
            return "generic error";
        case gauss_CL_ERROR:
            return "cl error";
    }
    return "invalid error code";
}

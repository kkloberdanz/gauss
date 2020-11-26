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

float (*_gauss_cblas_sdot)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST float *x,
    OPENBLAS_CONST blasint incx,
    OPENBLAS_CONST float *y,
    OPENBLAS_CONST blasint incy
);

float (*_gauss_cblas_snrm2)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST float *x,
    OPENBLAS_CONST blasint incx
);

float (*_gauss_cblas_sasum)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST float *x,
    OPENBLAS_CONST blasint incx
);

size_t (*_gauss_cblas_isamax)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST float *x,
    OPENBLAS_CONST blasint incx
);

size_t (*_gauss_cblas_sscal)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST float a,
    OPENBLAS_CONST float *x,
    OPENBLAS_CONST blasint incx
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
            /* TODO: dlsym the necessary clBLAS functions
             * That way one does not have to build gauss with clBLAS
             */

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

/**
 * Allocates a gauss memory object.
 * if kind is -1, then gauss will decide what is the best backend to use
 * at runtime
 */
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

/*
 * Read a single value from gauss memory
 */
void gauss_read_value(gauss_Mem *obj, void *out) {
    switch (obj->kind) {
        case gauss_CL_FLOAT: {
            cl_float dotProduct; /* result from dot product */
            cl_int err;

            /* Allocate 1 element space for dotProduct */
            obj->data.cl_float = clCreateBuffer(
                gauss_get_cl_ctx(),
                CL_MEM_WRITE_ONLY,
                (sizeof(cl_float)),
                NULL,
                &err
            );


            /* TODO: wait for cl events, enqueu read buffer, and set out
             * to read buffer*/
            *(float *)out = 42.0;
            /* Wait for calculations to be finished. */
            err = clWaitForEvents(1, &(obj->event));

            /* Fetch results of calculations from GPU memory. */
            err = clEnqueueReadBuffer(gauss_get_queue(), obj->data.cl_float,
                    CL_TRUE, 0, sizeof(cl_float), &dotProduct, 0, NULL, NULL
            );
            *(float *)out = dotProduct;

            /* Release OpenCL events. */
            clReleaseEvent(obj->event);
            break;
        }

        case gauss_FLOAT:
            *(float *)out = *obj->data.flt;
            break;

        case gauss_DOUBLE:
            *(double *)out = *obj->data.dbl;
            break;
    }
}

/*
 * Read a buffer from gauss memory
 */
void *gauss_read_buffer(gauss_Mem *obj) {
    switch (obj->kind) {
        case gauss_CL_FLOAT:
            /* TODO: wait for cl events, enqueue read buffer,
             * and return read buffer*/
            return NULL;

        case gauss_FLOAT:
        case gauss_DOUBLE:
            return obj->data.vd;
    }
}

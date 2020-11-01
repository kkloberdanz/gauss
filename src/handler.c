#include <stdlib.h>
#include <dlfcn.h>

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

gauss_Mem *gauss_alloc(size_t nmemb, gauss_MemKind kind) {
    gauss_Mem *ptr = malloc(sizeof(gauss_Mem));
    if (!ptr) {
        goto fail;
    }

    ptr->kind = kind;
    ptr->data.vd = NULL;
    switch (kind) {
        case gauss_RECCOMENDED:
            /* TODO:
             * let gauss reccomend a backend that is best suited to
             * your system */
            break;

        case gauss_FLOAT:
            ptr->data.flt = gauss_simd_alloc(sizeof(float) * nmemb);
            break;

        case gauss_DOUBLE:
            ptr->data.dbl = gauss_simd_alloc(sizeof(double) * nmemb);
            break;

        case gauss_CL_FLOAT:
            /* TODO:
             * create function to allocate and enqueue an OpenCL buffer */
            break;
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

#include "../include/gauss.h"
#include "../include/opencl.h"

static bool has_openblas = false;
static bool has_clblas = false;
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

    clblas_handle = dlopen("libclBLAS.so", RTLD_LAZY|RTLD_GLOBAL);
    if (clblas_handle) {
        has_clblas = true;
        gauss_init_opencl();
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

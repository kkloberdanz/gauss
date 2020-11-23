#ifndef GAUSS_HANDLER_H
#define GAUSS_HANDLER_H

#include <cblas.h>
#include <stdbool.h>

#include "opencl.h"

extern double (*_gauss_cblas_ddot)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST double *x,
    OPENBLAS_CONST blasint incx,
    OPENBLAS_CONST double *y,
    OPENBLAS_CONST blasint incy
);

extern double (*_gauss_cblas_dnrm2)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST double *x,
    OPENBLAS_CONST blasint incx
);

extern double (*_gauss_cblas_dasum)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST double *x,
    OPENBLAS_CONST blasint incx
);

extern size_t (*_gauss_cblas_idamax)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST double *x,
    OPENBLAS_CONST blasint incx
);

extern size_t (*_gauss_cblas_dscal)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST double a,
    OPENBLAS_CONST double *x,
    OPENBLAS_CONST blasint incx
);

extern float (*_gauss_cblas_sdot)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST float *x,
    OPENBLAS_CONST blasint incx,
    OPENBLAS_CONST float *y,
    OPENBLAS_CONST blasint incy
);

extern float (*_gauss_cblas_snrm2)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST float *x,
    OPENBLAS_CONST blasint incx
);

extern float (*_gauss_cblas_sasum)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST float *x,
    OPENBLAS_CONST blasint incx
);

extern size_t (*_gauss_cblas_isamax)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST float *x,
    OPENBLAS_CONST blasint incx
);

extern size_t (*_gauss_cblas_sscal)(
    OPENBLAS_CONST blasint n,
    OPENBLAS_CONST float a,
    OPENBLAS_CONST float *x,
    OPENBLAS_CONST blasint incx
);

void gauss_init(void);

void gauss_close(void);

extern bool has_openblas;
extern bool has_clblas;
extern void *openblas_handle;
extern void *clblas_handle;

#endif /* GAUSS_HANDLER_H */

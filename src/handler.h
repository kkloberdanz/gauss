#ifndef GAUSS_HANDLER_H
#define GAUSS_HANDLER_H

#include <stdbool.h>
#include <stdint.h>

#include "opencl.h"

typedef int64_t blasint;

extern double (*_gauss_cblas_ddot)(
    const blasint n,
    const double *x,
    const blasint incx,
    const double *y,
    const blasint incy
);

extern double (*_gauss_cblas_dnrm2)(
    const blasint n,
    const double *x,
    const blasint incx
);

extern double (*_gauss_cblas_dasum)(
    const blasint n,
    const double *x,
    const blasint incx
);

extern size_t (*_gauss_cblas_idamax)(
    const blasint n,
    const double *x,
    const blasint incx
);

extern size_t (*_gauss_cblas_dscal)(
    const blasint n,
    const double a,
    const double *x,
    const blasint incx
);

extern float (*_gauss_cblas_sdot)(
    const blasint n,
    const float *x,
    const blasint incx,
    const float *y,
    const blasint incy
);

extern float (*_gauss_cblas_snrm2)(
    const blasint n,
    const float *x,
    const blasint incx
);

extern float (*_gauss_cblas_sasum)(
    const blasint n,
    const float *x,
    const blasint incx
);

extern size_t (*_gauss_cblas_isamax)(
    const blasint n,
    const float *x,
    const blasint incx
);

extern size_t (*_gauss_cblas_sscal)(
    const blasint n,
    const float a,
    const float *x,
    const blasint incx
);

void gauss_init(void);

void gauss_close(void);

extern bool has_openblas;
extern bool has_clblas;
extern void *openblas_handle;
extern void *clblas_handle;

#endif /* GAUSS_HANDLER_H */

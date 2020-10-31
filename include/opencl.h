#ifndef GAUSS_OPENCL_H
#define GAUSS_OPENCL_H

#include <clBLAS.h>

#include "util.h"

gauss_Error gauss_close_opencl(void);
gauss_Error gauss_init_opencl(void);
gauss_Error gauss_clblas_sdot(
    const size_t N,
    cl_float X[],
    const int incx,
    cl_float Y[],
    const int incy,
    float *out
);

#endif /* GAUSS_OPENCL_H */

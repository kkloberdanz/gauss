#ifndef GAUSS_OPENCL_H
#define GAUSS_OPENCL_H

#define CL_TARGET_OPENCL_VERSION 120

#include <clBLAS.h>

#include "util.h"
#include "alloc.h"

cl_context gauss_get_cl_ctx(void);

cl_command_queue gauss_get_queue(void);

gauss_Error gauss_close_opencl(void);

gauss_Error gauss_init_opencl(void);

gauss_Error gauss_clblas_sdot(
    const size_t N,
    gauss_Mem *X,
    const int incx,
    gauss_Mem *Y,
    const int incy,
    float *out /* result from dot product */
);

#endif /* GAUSS_OPENCL_H */

/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
/* Include CLBLAS header. It automatically includes needed OpenCL header,
 * so we can drop out explicit inclusion of cl.h header.
 */
#include <clBLAS.h>
/* This example uses predefined matrices and their characteristics for
 * simplicity purpose.
 */

#include "../include/util.h"

static cl_command_queue queue = 0;
static cl_context ctx = 0;

gauss_Error gauss_init_opencl(void) {
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetPlatformIDs() failed with %d\n", err );
        return gauss_GENERIC_ERROR;
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetDeviceIDs() failed with %d\n", err );
        return gauss_GENERIC_ERROR;
    }
    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateContext() failed with %d\n", err );
        return gauss_GENERIC_ERROR;
    }
    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateCommandQueue() failed with %d\n", err );
        clReleaseContext(ctx);
        return gauss_GENERIC_ERROR;
    }
    /* Setup clblas. */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        printf("clblasSetup() failed with %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return gauss_GENERIC_ERROR;
    }
    return gauss_OK;
}

/* Prepare OpenCL memory objects and place matrices inside them. */
gauss_Error gauss_clblas_sdot(
    const size_t N,
    cl_float X[],
    const int incx,
    cl_float Y[],
    const int incy,
    float *out /* result from dot product */
) {
    cl_float dotProduct; /* result from dot product */
    cl_int err;
    gauss_Error status_code = gauss_OK;
    int lenX = 1 + (N-1)*abs(incx);
    int lenY = 1 + (N-1)*abs(incy);
    cl_mem bufX, bufY, bufDotP, scratchBuff;
    cl_event event = NULL;

    bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (lenX*sizeof(cl_float)), NULL, &err);
    bufY = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (lenY*sizeof(cl_float)), NULL, &err);
    // Allocate 1 element space for dotProduct
    bufDotP = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (sizeof(cl_float)), NULL, &err);
    // Allocate minimum of N elements
    scratchBuff = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (N*sizeof(cl_float)), NULL, &err);
    err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0, (lenX*sizeof(cl_float)), X, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufY, CL_TRUE, 0, (lenY*sizeof(cl_float)), Y, 0, NULL, NULL);
    /* Call clblas function. */
    err = clblasSdot( N, bufDotP, 0, bufX, 0, incx, bufY, 0, incy, scratchBuff,
                                    1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSdot() failed with %d\n", err);
        status_code = gauss_GENERIC_ERROR;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufDotP, CL_TRUE, 0, sizeof(cl_float),
                                    &dotProduct, 0, NULL, NULL);
        printf("Result dot product: %f\n", dotProduct);
    }
    *out = dotProduct;
    /* Release OpenCL events. */
    clReleaseEvent(event);
    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);
    clReleaseMemObject(bufDotP);
    clReleaseMemObject(scratchBuff);
    return status_code;
}

gauss_Error gauss_close_opencl() {
    /* Finalize work with clblas. */
    clblasTeardown();
    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return gauss_OK;
}

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "util.h"
#include "opencl.h"
#include "handler.h"

static cl_command_queue queue = 0;
static cl_context ctx = 0;

cl_context gauss_get_cl_ctx(void) {
    return ctx;
}

cl_command_queue gauss_get_queue(void) {
    return queue;
}

gauss_Error gauss_init_opencl(void) {
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, 0, 0};

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("clGetPlatformIDs() failed with %d\n", err);
        return gauss_CL_ERROR;
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("clGetDeviceIDs() failed with %d\n", err);
        return gauss_CL_ERROR;
    }
    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateContext() failed with %d\n", err);
        return gauss_CL_ERROR;
    }
    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateCommandQueue() failed with %d\n", err);
        clReleaseContext(ctx);
        return gauss_CL_ERROR;
    }
    /* Setup clblas. */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        printf("clblasSetup() failed with %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return gauss_CL_ERROR;
    }
    return gauss_OK;
}

gauss_Error gauss_close_opencl() {
    /* Finalize work with clblas. */
    clblasTeardown();
    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return gauss_OK;
}

gauss_Error gauss_clblas_sdot(
    const size_t N,
    gauss_Mem *X,
    const int incx,
    gauss_Mem *Y,
    const int incy,
    float *out /* result from dot product */
) {
    cl_float dotProduct; /* result from dot product */
    cl_int err;
    gauss_Error status_code = gauss_OK;
    cl_mem bufX, bufY, bufDotP, scratchBuff;
    cl_event event = NULL;

    bufX = X->data.cl_float;
    bufY = Y->data.cl_float;

    /* Allocate 1 element space for dotProduct */
    bufDotP = clCreateBuffer(
        ctx,
        CL_MEM_WRITE_ONLY,
        (sizeof(cl_float)),
        NULL,
        &err
    );

    /* Allocate minimum of N elements */
    scratchBuff = clCreateBuffer(
        ctx,
        CL_MEM_READ_WRITE,
        (N*sizeof(cl_float)),
        NULL,
        &err
    );

    /* Call clblas function. */
    err = clblasSdot(N, bufDotP, 0, bufX, 0, incx, bufY, 0, incy, scratchBuff,
         1, &queue, 0, NULL, &event);

    if (err != CL_SUCCESS) {
        status_code = gauss_CL_ERROR;
    } else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufDotP, CL_TRUE, 0, sizeof(cl_float),
                                    &dotProduct, 0, NULL, NULL);
        *out = dotProduct;
    }

    /* Release OpenCL events. */
    clReleaseEvent(event);

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufDotP);
    clReleaseMemObject(scratchBuff);
    return status_code;
}

gauss_Error gauss_clblas_snrm2(gauss_Mem *obj, float *out) {
    gauss_Error status_code = gauss_OK;
    cl_int err;
    cl_mem bufNRM2;
    cl_mem scratchBuff;
    cl_event event = NULL;
    cl_float NRM2;
    const int inc = 1;
    const size_t size = obj->nmemb;

    fprintf(stderr, "size: %zu\n", obj->nmemb);
    fprintf(stderr, "out: %f\n", *out);

    /* Allocate 1 element space for NRM2 */
    bufNRM2 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        (sizeof(cl_float)), NULL, &err
    );

    /* Allocate minimum of N elements */
    scratchBuff = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        (2*size*sizeof(cl_float)), NULL, &err
    );

    err = clEnqueueWriteBuffer(
        queue, obj->data.cl_float, CL_TRUE, 0,
        (size*sizeof(cl_float)), obj->data.cl_float,
        0, NULL, NULL
    );

    /* Call clblas function. */
    err = clblasSnrm2(size, bufNRM2, 0, obj->data.cl_float, 0, inc,
        scratchBuff, 1, &queue, 0, NULL, &event
    );
    if (err != CL_SUCCESS) {
        status_code = gauss_CL_ERROR;
    } else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufNRM2, CL_TRUE, 0, sizeof(cl_float),
                                    &NRM2, 0, NULL, NULL);
        *out = NRM2;
    }
    /* Release OpenCL events. */
    clReleaseEvent(event);

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufNRM2);
    clReleaseMemObject(scratchBuff);

    return status_code;
}

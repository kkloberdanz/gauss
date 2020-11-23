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

/*
static void print_arr_f32(float *arr, size_t size) {
    size_t i = 0;
    for (i = 0; i < size - 1; i++) {
        printf("%f, ", arr[i]);
    }
    if (i > 0) {
        printf("%f", arr[i]);
    }
    printf("\n");
}
*/

void gauss_opencl_free(cl_mem ptr) {
    clReleaseMemObject(ptr);
}

gauss_Error gauss_enqueue_gpu_memory(float *ptr, size_t nmemb) {
    int err = gauss_OK;
    cl_mem buf = clCreateBuffer(
        ctx,
        CL_MEM_READ_ONLY,
        (nmemb * sizeof(cl_float)),
        NULL,
        &err
    );
    err = clEnqueueWriteBuffer(
        queue,
        buf,
        CL_TRUE,
        0,
        (nmemb * sizeof(cl_float)),
        ptr,
        0,
        NULL,
        NULL
    );
    if (err) {
        return gauss_CL_ERROR;
    } else {
        return gauss_OK;
    }
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

    fprintf(stderr, "using opencl\n");

/*
    bufX = clCreateBuffer(
        ctx,
        CL_MEM_READ_ONLY,
        (lenX*sizeof(cl_float)),
        NULL,
        &err
    );

    bufY = clCreateBuffer(
        ctx,
        CL_MEM_READ_ONLY,
        (lenY*sizeof(cl_float)),
        NULL,
        &err
    );
*/

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

/*
    err = clEnqueueWriteBuffer(
        queue,
        bufX,
        CL_TRUE,
        0,
        (lenX*sizeof(cl_float)),
        X,
        0,
        NULL,
        NULL
    );

    err = clEnqueueWriteBuffer(
        queue,
        bufY,
        CL_TRUE,
        0,
        (lenY*sizeof(cl_float)),
        Y,
        0,
        NULL,
        NULL
    );
*/

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

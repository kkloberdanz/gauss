#ifndef GAUSS_UTIL_H
#define GAUSS_UTIL_H

#include <stddef.h>

void *gauss_simd_alloc(size_t size);

typedef enum gauss_Error {
    gauss_OUT_OF_MEMORY = -1,
    gauss_OK = 0,
    gauss_GENERIC_ERROR = 1,
    gauss_CL_ERROR = 2,
    gauss_MISMATCHED_TYPES = 3,
    gauss_MISMATCHED_DIMENSIONS = 4
} gauss_Error;

#endif /* GAUSS_UTIL_H */

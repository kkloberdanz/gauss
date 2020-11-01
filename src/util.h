#ifndef GAUSS_UTIL_H
#define GAUSS_UTIL_H

#include <stddef.h>

void *gauss_simd_alloc(size_t size);

typedef enum gauss_Error {
    gauss_OUT_OF_MEMORY = -1,
    gauss_OK = 0,
    gauss_GENERIC_ERROR = 1
} gauss_Error;

#endif /* GAUSS_UTIL_H */

#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>

void *gauss_simd_alloc(size_t size);

typedef enum gauss_Error {
    gauss_OUT_OF_MEMORY = -1,
    gauss_OK = 0
} gauss_Error;

#endif /* UTIL_H */

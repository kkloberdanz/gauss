#include <stdlib.h>

#include "vec-math.h"
#include "util.h"
#include "alloc.h"

void *aligned_alloc(size_t, size_t);

void *gauss_simd_alloc(size_t size) {
#ifndef UNKNOWN_SIMD
    return aligned_alloc(SIMD_ALIGN_SIZE, size);
#else
    return malloc(size);
#endif
}

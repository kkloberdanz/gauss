#include <stdlib.h>

#include "vec-math.h"
#include "util.h"

void *aligned_alloc(size_t, size_t);

void *gauss_simd_alloc(size_t size) {
#ifndef UNKNOWN_SIMD
    return aligned_alloc(SIMD_ALIGN_SIZE, size);
#else
    return malloc(size);
#endif
}

double gauss_double_array_at(const double *arr, size_t i) {
    return arr[i];
}

void gauss_set_double_array_at(double *arr, size_t i, double value) {
    arr[i] = value;
}

void gauss_set_float_array_at(float *arr, size_t i, float value) {
    arr[i] = value;
}

void gauss_free(void *ptr) {
    free(ptr);
}

#include <stdio.h>
#include <stdint.h>

#include "../include/gauss.h"

void gauss_add_f64(double *dst, double *a, double *b, size_t size) {
    size_t i;
    for (i = 0; i < size; i++) {
        dst[i] = a[i] + b[i];
    }
}

double gauss_dot_f64(double *a, double *b, size_t size) {
    double acc = 0.0;
    size_t i;

    for (i = 0; i < size; i++) {
        acc += a[i] * b[i];
    }
    return acc;
}

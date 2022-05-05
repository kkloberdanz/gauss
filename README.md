# gauss

General Algorithmic Unified Statistical Solvers

## Build

Ensure you have the OpenCL and clBLAS libraries installed, then run:

```
make -j`nproc`
```

This will leave you with the library `libgauss.so`.

## Notes

Gauss was intended to be used from Python as a GPU accelerated NumPy like
library, but it can also be used from pure C. You can see examples of using
gauss in C in the file `test/test.c`

Gauss has a few dependencies, such as OpenCL and clBLAS. These libraries are
used to enable GPU compute. You can find the usage of these libraries in
`opencl.c`.

Gauss can optionally benefit by having openBLAS installed, and will search for
this library on startup and use `dlopen` to dynamically link with it at startup.
You can see in the function `void gauss_init(void)` where this takes place.
OpenBLAS can greatly speed up compute, but if it is not installed, then gauss
will fall back to pure C implementations of the BLAS algorithms.

Some algorithms that are not provided by BLAS have been implemented in C using
AVX and SSE SIMD intrinsics. These implementations can be found in `vec-math.c`.

## Examples

```C
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "../src/blas-level1.h"
#include "../src/util.h"
#include "../src/vec-math.h"
#include "../src/handler.h"

bool floats_are_same(float a, float b) {
    const float epsilon = 0.001;
    return fabs(a - b) <= epsilon * fabs(a);
}

bool doubles_are_same(double a, double b) {
    const double epsilon = 0.0001;
    return fabs(a - b) <= epsilon * fabs(a);
}

int main(void) {
    const size_t size = 100;
    double *dst = malloc(sizeof(double) * size);
    double *a = malloc(sizeof(double) * size);
    double *b = malloc(sizeof(double) * size);
    float *c = malloc(sizeof(float) * size);
    float *d = malloc(sizeof(float) * size);
    size_t i;
    double ans;
    size_t idx;

    gauss_init();

    for (i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 2;
        c[i] = i;
        d[i] = i * 2;
    }

    gauss_vec_add_f64(dst, a, b, size);
    for (i = 0; i < size; i++) {
        assert(dst[i] == a[i] + b[i]);
    }

    gauss_vec_mul_f64(dst, a, b, size);
    for (i = 0; i < size; i++) {
        assert(dst[i] == a[i] * b[i]);
    }

    ans = gauss_vec_dot_f64(a, b, size);
    assert(doubles_are_same(ans, 656700.0));

    ans = gauss_vec_l2norm_f64(a, size);
    assert(doubles_are_same(ans, 573.018324));

    ans = gauss_vec_l1norm_f64(a, size);
    assert(doubles_are_same(ans, 4950.0));

    idx = gauss_vec_index_max_f64(a, size);
    assert(idx == 99);

    free(dst);
    free(a);
    free(b);
    free(c);
    free(d);
    gauss_close();

    puts("PASS");
    return 0;
}
```

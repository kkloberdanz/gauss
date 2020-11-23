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
 /*   float ans_f32 = 0.0;*/
    size_t idx;
/*    gauss_Error err = gauss_OK;*/

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

/*
    err = gauss_vec_dot_f32(c, d, size, &ans_f32);
    assert(err == gauss_OK);
    assert(floats_are_same(ans_f32, 656700.0));
*/

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

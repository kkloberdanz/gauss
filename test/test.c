#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "../include/gauss.h"

bool doubles_are_same(double a, double b) {
    const double epsilon = 0.0001;
    return fabs(a - b) <= epsilon * fabs(a);
}

int main(void) {
    const size_t size = 100;
    double *dst = malloc(sizeof(double) * size);
    double *a = malloc(sizeof(double) * size);
    double *b = malloc(sizeof(double) * size);
    size_t i;
    double ans;
    size_t idx;

    gauss_init();
    for (i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 2;
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

    ans = gauss_vec_norm_f64(a, size);
    assert(doubles_are_same(ans, 573.018324));

    ans = gauss_vec_sumabs_f64(a, size);
    assert(doubles_are_same(ans, 4950.0));

    idx = gauss_vec_index_max_f64(a, size);
    assert(idx == 99);

    free(dst);
    free(a);
    free(b);
    gauss_close();

    puts("PASS");
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "../include/gauss.h"

int main(void) {
    const size_t size = 100;
    double *dst = malloc(sizeof(double) * size);
    double *a = malloc(sizeof(double) * size);
    double *b = malloc(sizeof(double) * size);
    size_t i;
    double ans;

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
    assert(ans == 656700.0);

    free(dst);
    free(a);
    free(b);
    gauss_close();

    puts("PASS");
    return 0;
}

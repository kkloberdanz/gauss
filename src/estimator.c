#include <stdlib.h>

#include "vec-math.h"
#include "util.h"

/*
def ordinary_least_squares(x, y):
    mean_x = x.mean()
    mean_y = y.mean()
    m = (((x - mean_x) * (y - mean_y)).sum() /
         ((x - mean_x).square()).sum())
    b = mean_y - m * mean_x
    return m, b
*/

/*
static void print_arr(double *arr, size_t size) {
    size_t i = 0;
    for (i = 0; i < size - 1; i++) {
        printf("%f, ", arr[i]);
    }
    if (i > 0) {
        printf("%f", arr[i]);
    }
    printf("\n");
}
*/

gauss_Error ordinary_least_squares(
    const double *x,
    const double *y,
    const size_t size,
    double *out_m,
    double *out_b
) {
    double numerator_sum;
    double denominator_sum;
    double m;
    double mean_x = gauss_mean_double_array(x, size);
    double mean_y = gauss_mean_double_array(y, size);
    double *squared_data = NULL;

    double *x_diff_from_mean = gauss_simd_alloc(sizeof(double) * size);
    if (!x_diff_from_mean) {
        goto exit_oom;
    }

    double *y_diff_from_mean = gauss_simd_alloc(sizeof(double) * size);
    if (!y_diff_from_mean) {
        goto free_x_diff;
    }

    double *m_mul_term = gauss_simd_alloc(sizeof(double) * size);
    if (!y_diff_from_mean) {
        goto free_y_diff;
    }

    gauss_sub_double_scalar(x_diff_from_mean, x, mean_x, size);
    gauss_sub_double_scalar(y_diff_from_mean, y, mean_y, size);

    gauss_mul_double_array(
        m_mul_term, x_diff_from_mean, y_diff_from_mean, size
    );

    /* reuse this buffer to avoid an extra allocation */
    squared_data = y_diff_from_mean;

    numerator_sum = gauss_vec_sum_f64(m_mul_term, size);

    gauss_mul_double_array(
        squared_data, x_diff_from_mean, x_diff_from_mean, size
    );

    denominator_sum = gauss_vec_sum_f64(squared_data, size);

    m = numerator_sum / denominator_sum;

    *out_b = mean_y - m * mean_x;

    *out_m = m;

    free(m_mul_term);
    free(y_diff_from_mean);
    free(x_diff_from_mean);

    return gauss_OK;

/*
free_m_mul_term:
    free(m_mul_term);
*/
free_y_diff:
    free(y_diff_from_mean);
free_x_diff:
    free(x_diff_from_mean);
exit_oom:
    return gauss_OUT_OF_MEMORY;
}

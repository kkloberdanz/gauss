#include <stdlib.h>

#include "../include/vec-math.h"
#include "../include/util.h"

gauss_Error guass_mean_squared_error(
    double *y_true,
    double *y_predicted,
    size_t size,
    double *out_mse
) {
    double *buf = gauss_simd_alloc(sizeof(double) * size);
    double mse;

    if (!buf) {
        return gauss_OUT_OF_MEMORY;
    }

    gauss_sub_double_array(buf, y_true, y_predicted, size);
    gauss_mul_double_array(buf, buf, buf, size);
    mse = gauss_vec_sum_f64(buf, size);
    *out_mse = mse / size;

    free(buf);

    return gauss_OK;
}

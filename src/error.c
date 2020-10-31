#include <stdlib.h>

#include "../include/vec-math.h"
#include "../include/util.h"
#include "../include/blas-level1.h"

gauss_Error guass_mean_squared_error(
    double *y_true,
    double *y_predicted,
    size_t size,
    double *out_mse
) {
    double *buf = gauss_simd_alloc(sizeof(double) * size);
    double summation;

    if (!buf) {
        return gauss_OUT_OF_MEMORY;
    }

    /* buf = y_true - y_predicted */
    gauss_sub_double_array(buf, y_true, y_predicted, size);

    /* buf = buf^2 */
    gauss_mul_double_array(buf, buf, buf, size);

    /* summation = sum(buf) */
    summation = gauss_vec_l1norm_f64(buf, size);

    *out_mse = summation / size;

    free(buf);

    return gauss_OK;
}

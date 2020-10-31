#ifndef ERROR_H
#define ERROR_H

#include "util.h"

gauss_Error guass_mean_squared_error(
    double *y_true,
    double *y_predicted,
    size_t size,
    double *out_mse
);

#endif /* ERROR_H */

#ifndef ALLOC_H
#define ALLOC_H

#include <clBLAS.h>

typedef enum gauss_MemKind {
    gauss_FLOAT = 1,
    gauss_DOUBLE = 2,
    gauss_CL_FLOAT = 3
} gauss_MemKind;

typedef struct gauss_Mem {
    gauss_MemKind kind;
    size_t nmemb;
    union {
        void *vd;
        double *dbl;
        float *flt;
        cl_mem cl_float;
    } data;
} gauss_Mem;

#endif /* ALLOC_H */

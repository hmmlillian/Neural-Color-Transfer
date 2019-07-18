#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <cublas_v2.h>
#include "HelperCuda.h"

void solve_ls_cg_gpu(int size, int constraints, double* A, int* columns, int* rowindex, double* x, double* b, int nonzeros, double tolerance, int maxitrs);
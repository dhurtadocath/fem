/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * xgetrf.h
 *
 * Code generation for function 'xgetrf'
 *
 */

#pragma once

/* Include files */
#include "fullBFGStoCompileToMex_types.h"
#include "rtwtypes.h"
#include "emlrt.h"
#include "mex.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Function Declarations */
void xgetrf(const emlrtStack *sp, int32_T m, int32_T n, emxArray_real_T *A,
            int32_T lda, int32_T ipiv_data[], int32_T ipiv_size[2]);

/* End of code generation (xgetrf.h) */

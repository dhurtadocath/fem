/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * xtrsm.h
 *
 * Code generation for function 'xtrsm'
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
void xtrsm(int32_T m, int32_T n, const emxArray_real_T *A, int32_T lda,
           emxArray_real_T *B, int32_T ldb);

/* End of code generation (xtrsm.h) */

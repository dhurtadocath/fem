/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * norm.h
 *
 * Code generation for function 'norm'
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
real_T b_norm(const emxArray_real_T *x);

real_T c_norm(const real_T x_data[], int32_T x_size);

/* End of code generation (norm.h) */

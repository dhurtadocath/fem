/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * eml_mtimes_helper.h
 *
 * Code generation for function 'eml_mtimes_helper'
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
void b_dynamic_size_checks(const emlrtStack *sp, const emxArray_real_T *a,
                           const emxArray_real_T *b, int32_T innerDimA,
                           int32_T innerDimB);

void c_dynamic_size_checks(const emlrtStack *sp, const int32_T a_size[2],
                           int32_T b_size, int32_T innerDimA,
                           int32_T innerDimB);

void d_dynamic_size_checks(const emlrtStack *sp, int32_T a_size, int32_T b_size,
                           int32_T innerDimA, int32_T innerDimB);

void dynamic_size_checks(const emlrtStack *sp, const emxArray_real_T *a,
                         int32_T b_size, int32_T innerDimA, int32_T innerDimB);

/* End of code generation (eml_mtimes_helper.h) */

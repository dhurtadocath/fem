/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * eye.c
 *
 * Code generation for function 'eye'
 *
 */

/* Include files */
#include "eye.h"
#include "eml_int_forloop_overflow_check.h"
#include "fullBFGStoCompileToMex_data.h"
#include "fullBFGStoCompileToMex_emxutil.h"
#include "fullBFGStoCompileToMex_types.h"
#include "rt_nonfinite.h"

/* Variable Definitions */
static emlrtRSInfo mc_emlrtRSI = {
    50,                                                           /* lineNo */
    "eye",                                                        /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/lib/matlab/elmat/eye.m" /* pathName */
};

static emlrtRSInfo nc_emlrtRSI = {
    96,                                                           /* lineNo */
    "eye",                                                        /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/lib/matlab/elmat/eye.m" /* pathName */
};

static emlrtRSInfo oc_emlrtRSI = {
    21,                           /* lineNo */
    "checkAndSaturateExpandSize", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/"
    "checkAndSaturateExpandSize.m" /* pathName */
};

static emlrtRTEInfo c_emlrtRTEI = {
    58,                   /* lineNo */
    23,                   /* colNo */
    "assertValidSizeArg", /* fName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/"
    "assertValidSizeArg.m" /* pName */
};

static emlrtRTEInfo j_emlrtRTEI = {
    94,                                                           /* lineNo */
    5,                                                            /* colNo */
    "eye",                                                        /* fName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/lib/matlab/elmat/eye.m" /* pName */
};

/* Function Definitions */
void eye(const emlrtStack *sp, real_T varargin_1, emxArray_real_T *b_I)
{
  emlrtStack b_st;
  emlrtStack st;
  real_T *I_data;
  int32_T i;
  int32_T loop_ub;
  st.prev = sp;
  st.tls = sp->tls;
  st.site = &mc_emlrtRSI;
  b_st.prev = &st;
  b_st.tls = st.tls;
  b_st.site = &oc_emlrtRSI;
  if (varargin_1 != varargin_1) {
    emlrtErrorWithMessageIdR2018a(
        &b_st, &c_emlrtRTEI, "Coder:MATLAB:NonIntegerInput",
        "Coder:MATLAB:NonIntegerInput", 4, 12, MIN_int32_T, 12, MAX_int32_T);
  }
  i = b_I->size[0] * b_I->size[1];
  b_I->size[0] = (int32_T)varargin_1;
  b_I->size[1] = (int32_T)varargin_1;
  emxEnsureCapacity_real_T(sp, b_I, i, &j_emlrtRTEI);
  I_data = b_I->data;
  loop_ub = (int32_T)varargin_1 * (int32_T)varargin_1;
  for (i = 0; i < loop_ub; i++) {
    I_data[i] = 0.0;
  }
  if ((int32_T)varargin_1 > 0) {
    st.site = &nc_emlrtRSI;
    if ((int32_T)varargin_1 > 2147483646) {
      b_st.site = &mb_emlrtRSI;
      check_forloop_overflow_error(&b_st);
    }
    i = (uint16_T)(int32_T)varargin_1;
    for (loop_ub = 0; loop_ub < i; loop_ub++) {
      I_data[loop_ub + b_I->size[0] * loop_ub] = 1.0;
    }
  }
}

/* End of code generation (eye.c) */

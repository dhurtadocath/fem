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
#include "mwmathutil.h"

/* Variable Definitions */
static emlrtRSInfo tc_emlrtRSI = {
    50,    /* lineNo */
    "eye", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/elmat/eye.m" /* pathName */
};

static emlrtRSInfo uc_emlrtRSI = {
    96,    /* lineNo */
    "eye", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/elmat/eye.m" /* pathName */
};

static emlrtRSInfo vc_emlrtRSI = {
    21,                           /* lineNo */
    "checkAndSaturateExpandSize", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/checkAndSaturateExpandSize.m" /* pathName */
};

static emlrtRTEInfo d_emlrtRTEI = {
    58,                   /* lineNo */
    23,                   /* colNo */
    "assertValidSizeArg", /* fName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/assertValidSizeArg.m" /* pName */
};

static emlrtRTEInfo j_emlrtRTEI = {
    94,    /* lineNo */
    5,     /* colNo */
    "eye", /* fName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/elmat/eye.m" /* pName */
};

/* Function Definitions */
void eye(const emlrtStack *sp, real_T varargin_1, emxArray_real_T *b_I)
{
  emlrtStack b_st;
  emlrtStack st;
  int32_T i;
  int32_T loop_ub;
  int32_T m_tmp_tmp;
  st.prev = sp;
  st.tls = sp->tls;
  st.site = &tc_emlrtRSI;
  b_st.prev = &st;
  b_st.tls = st.tls;
  b_st.site = &vc_emlrtRSI;
  if ((varargin_1 != varargin_1) || muDoubleScalarIsInf(varargin_1)) {
    emlrtErrorWithMessageIdR2018a(
        &b_st, &d_emlrtRTEI, "Coder:MATLAB:NonIntegerInput",
        "Coder:MATLAB:NonIntegerInput", 4, 12, MIN_int32_T, 12, MAX_int32_T);
  }
  m_tmp_tmp = (int32_T)varargin_1;
  i = b_I->size[0] * b_I->size[1];
  b_I->size[0] = (int32_T)varargin_1;
  b_I->size[1] = (int32_T)varargin_1;
  emxEnsureCapacity_real_T(sp, b_I, i, &j_emlrtRTEI);
  loop_ub = (int32_T)varargin_1 * (int32_T)varargin_1;
  for (i = 0; i < loop_ub; i++) {
    b_I->data[i] = 0.0;
  }
  if ((int32_T)varargin_1 > 0) {
    st.site = &uc_emlrtRSI;
    if ((1 <= (int32_T)varargin_1) && ((int32_T)varargin_1 > 2147483646)) {
      b_st.site = &mb_emlrtRSI;
      check_forloop_overflow_error(&b_st);
    }
    for (loop_ub = 0; loop_ub < m_tmp_tmp; loop_ub++) {
      b_I->data[loop_ub + b_I->size[0] * loop_ub] = 1.0;
    }
  }
}

/* End of code generation (eye.c) */

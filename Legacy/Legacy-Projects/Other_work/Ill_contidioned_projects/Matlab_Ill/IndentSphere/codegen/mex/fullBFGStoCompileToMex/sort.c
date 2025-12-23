/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * sort.c
 *
 * Code generation for function 'sort'
 *
 */

/* Include files */
#include "sort.h"
#include "eml_int_forloop_overflow_check.h"
#include "fullBFGStoCompileToMex_data.h"
#include "rt_nonfinite.h"
#include "sortIdx.h"

/* Type Definitions */
#ifndef struct_emxArray_int32_T_570
#define struct_emxArray_int32_T_570
struct emxArray_int32_T_570 {
  int32_T data[570];
};
#endif /* struct_emxArray_int32_T_570 */
#ifndef typedef_emxArray_int32_T_570
#define typedef_emxArray_int32_T_570
typedef struct emxArray_int32_T_570 emxArray_int32_T_570;
#endif /* typedef_emxArray_int32_T_570 */

/* Variable Definitions */
static emlrtRSInfo fb_emlrtRSI = {
    76,     /* lineNo */
    "sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sort.m" /* pathName
                                                                        */
};

static emlrtRSInfo gb_emlrtRSI = {
    79,     /* lineNo */
    "sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sort.m" /* pathName
                                                                        */
};

static emlrtRSInfo hb_emlrtRSI = {
    81,     /* lineNo */
    "sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sort.m" /* pathName
                                                                        */
};

static emlrtRSInfo ib_emlrtRSI = {
    84,     /* lineNo */
    "sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sort.m" /* pathName
                                                                        */
};

static emlrtRSInfo jb_emlrtRSI = {
    87,     /* lineNo */
    "sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sort.m" /* pathName
                                                                        */
};

static emlrtRSInfo kb_emlrtRSI = {
    90,     /* lineNo */
    "sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sort.m" /* pathName
                                                                        */
};

/* Function Definitions */
void sort(const emlrtStack *sp, real_T x_data[], const int32_T *x_size)
{
  emlrtStack b_st;
  emlrtStack st;
  emxArray_int32_T_570 nd_emlrtRSI;
  real_T vwork_data[570];
  int32_T dim;
  int32_T k;
  int32_T vlen;
  int32_T vstride;
  int32_T vwork_size;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  dim = 0;
  if (*x_size != 1) {
    dim = -1;
  }
  if (dim + 2 <= 1) {
    vwork_size = *x_size;
  } else {
    vwork_size = 1;
  }
  vlen = vwork_size - 1;
  st.site = &fb_emlrtRSI;
  vstride = 1;
  for (k = 0; k <= dim; k++) {
    vstride *= *x_size;
  }
  st.site = &gb_emlrtRSI;
  st.site = &hb_emlrtRSI;
  if (vstride > 2147483646) {
    b_st.site = &mb_emlrtRSI;
    check_forloop_overflow_error(&b_st);
  }
  for (dim = 0; dim < vstride; dim++) {
    st.site = &ib_emlrtRSI;
    for (k = 0; k <= vlen; k++) {
      vwork_data[k] = x_data[dim + k * vstride];
    }
    st.site = &jb_emlrtRSI;
    sortIdx(&st, vwork_data, &vwork_size, nd_emlrtRSI.data);
    st.site = &kb_emlrtRSI;
    for (k = 0; k <= vlen; k++) {
      x_data[dim + k * vstride] = vwork_data[k];
    }
  }
}

/* End of code generation (sort.c) */

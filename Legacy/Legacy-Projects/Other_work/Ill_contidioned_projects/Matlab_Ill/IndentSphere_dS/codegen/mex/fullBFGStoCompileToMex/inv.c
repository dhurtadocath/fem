/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * inv.c
 *
 * Code generation for function 'inv'
 *
 */

/* Include files */
#include "inv.h"
#include "fullBFGStoCompileToMex_emxutil.h"
#include "fullBFGStoCompileToMex_mexutil.h"
#include "fullBFGStoCompileToMex_types.h"
#include "norm.h"
#include "rt_nonfinite.h"
#include "warning.h"
#include "xgetrf.h"
#include "xtrsm.h"
#include "mwmathutil.h"

/* Variable Definitions */
static emlrtRSInfo pc_emlrtRSI = {
    21,                                                            /* lineNo */
    "inv",                                                         /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/matfun/inv.m" /* pathName
                                                                    */
};

static emlrtRSInfo qc_emlrtRSI = {
    22,                                                            /* lineNo */
    "inv",                                                         /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/matfun/inv.m" /* pathName
                                                                    */
};

static emlrtRSInfo rc_emlrtRSI = {
    173,                                                           /* lineNo */
    "invNxN",                                                      /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/matfun/inv.m" /* pathName
                                                                    */
};

static emlrtRSInfo dd_emlrtRSI = {
    42,                                                            /* lineNo */
    "checkcond",                                                   /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/matfun/inv.m" /* pathName
                                                                    */
};

static emlrtRSInfo ed_emlrtRSI = {
    46,                                                            /* lineNo */
    "checkcond",                                                   /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/matfun/inv.m" /* pathName
                                                                    */
};

static emlrtMCInfo c_emlrtMCI = {
    53,        /* lineNo */
    19,        /* colNo */
    "flt2str", /* fName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/flt2str.m" /* pName
                                                                           */
};

static emlrtRTEInfo d_emlrtRTEI = {
    14,                                                            /* lineNo */
    15,                                                            /* colNo */
    "inv",                                                         /* fName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/matfun/inv.m" /* pName */
};

static emlrtRTEInfo k_emlrtRTEI = {
    19,                                                            /* lineNo */
    5,                                                             /* colNo */
    "inv",                                                         /* fName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/matfun/inv.m" /* pName */
};

static emlrtRTEInfo l_emlrtRTEI = {
    170,                                                           /* lineNo */
    1,                                                             /* colNo */
    "inv",                                                         /* fName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/matfun/inv.m" /* pName */
};

static emlrtRTEInfo m_emlrtRTEI = {
    173,                                                           /* lineNo */
    2,                                                             /* colNo */
    "inv",                                                         /* fName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/matfun/inv.m" /* pName */
};

static emlrtRTEInfo n_emlrtRTEI = {
    164,                                                           /* lineNo */
    14,                                                            /* colNo */
    "inv",                                                         /* fName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/matfun/inv.m" /* pName */
};

static emlrtRSInfo kd_emlrtRSI = {
    53,        /* lineNo */
    "flt2str", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/flt2str.m" /* pathName
                                                                           */
};

/* Function Declarations */
static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                               const emlrtMsgIdentifier *parentId,
                               char_T y[14]);

static const mxArray *b_sprintf(const emlrtStack *sp, const mxArray *m1,
                                const mxArray *m2, emlrtMCInfo *location);

static void emlrt_marshallIn(const emlrtStack *sp,
                             const mxArray *a__output_of_sprintf_,
                             const char_T *identifier, char_T y[14]);

static void invNxN(const emlrtStack *sp, const emxArray_real_T *x,
                   emxArray_real_T *y);

static void s_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId, char_T ret[14]);

/* Function Definitions */
static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                               const emlrtMsgIdentifier *parentId, char_T y[14])
{
  s_emlrt_marshallIn(sp, emlrtAlias(u), parentId, y);
  emlrtDestroyArray(&u);
}

static const mxArray *b_sprintf(const emlrtStack *sp, const mxArray *m1,
                                const mxArray *m2, emlrtMCInfo *location)
{
  const mxArray *pArrays[2];
  const mxArray *m;
  pArrays[0] = m1;
  pArrays[1] = m2;
  return emlrtCallMATLABR2012b((emlrtConstCTX)sp, 1, &m, 2, &pArrays[0],
                               "sprintf", true, location);
}

static void emlrt_marshallIn(const emlrtStack *sp,
                             const mxArray *a__output_of_sprintf_,
                             const char_T *identifier, char_T y[14])
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  b_emlrt_marshallIn(sp, emlrtAlias(a__output_of_sprintf_), &thisId, y);
  emlrtDestroyArray(&a__output_of_sprintf_);
}

static void invNxN(const emlrtStack *sp, const emxArray_real_T *x,
                   emxArray_real_T *y)
{
  emlrtStack st;
  emxArray_real_T *b_x;
  const real_T *x_data;
  real_T *b_x_data;
  real_T *y_data;
  int32_T ipiv_data[729];
  int32_T ipiv_size[2];
  int32_T b_n;
  int32_T i;
  int32_T k;
  int32_T n;
  int32_T yk;
  int16_T p_data[729];
  st.prev = sp;
  st.tls = sp->tls;
  x_data = x->data;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtConstCTX)sp);
  n = x->size[0];
  i = y->size[0] * y->size[1];
  y->size[0] = x->size[0];
  y->size[1] = x->size[1];
  emxEnsureCapacity_real_T(sp, y, i, &l_emlrtRTEI);
  y_data = y->data;
  b_n = x->size[0] * x->size[1];
  for (i = 0; i < b_n; i++) {
    y_data[i] = 0.0;
  }
  emxInit_real_T(sp, &b_x, &n_emlrtRTEI);
  i = b_x->size[0] * b_x->size[1];
  b_x->size[0] = x->size[0];
  b_x->size[1] = x->size[1];
  emxEnsureCapacity_real_T(sp, b_x, i, &m_emlrtRTEI);
  b_x_data = b_x->data;
  for (i = 0; i < b_n; i++) {
    b_x_data[i] = x_data[i];
  }
  st.site = &rc_emlrtRSI;
  xgetrf(&st, x->size[0], x->size[0], b_x, x->size[0], ipiv_data, ipiv_size);
  b_x_data = b_x->data;
  if (x->size[0] < 1) {
    b_n = 0;
  } else {
    b_n = x->size[0];
  }
  if (b_n > 0) {
    p_data[0] = 1;
    yk = 1;
    for (k = 2; k <= b_n; k++) {
      yk++;
      p_data[k - 1] = (int16_T)yk;
    }
  }
  i = ipiv_size[1];
  for (k = 0; k < i; k++) {
    b_n = ipiv_data[k];
    if (b_n > k + 1) {
      yk = p_data[b_n - 1];
      p_data[b_n - 1] = p_data[k];
      p_data[k] = (int16_T)yk;
    }
  }
  for (k = 0; k < n; k++) {
    int16_T i1;
    i1 = p_data[k];
    y_data[k + y->size[0] * (i1 - 1)] = 1.0;
    for (b_n = k + 1; b_n <= n; b_n++) {
      if (y_data[(b_n + y->size[0] * (i1 - 1)) - 1] != 0.0) {
        i = b_n + 1;
        for (yk = i; yk <= n; yk++) {
          y_data[(yk + y->size[0] * (i1 - 1)) - 1] -=
              y_data[(b_n + y->size[0] * (i1 - 1)) - 1] *
              b_x_data[(yk + b_x->size[0] * (b_n - 1)) - 1];
        }
      }
    }
  }
  xtrsm(x->size[0], x->size[0], b_x, x->size[0], y, x->size[0]);
  emxFree_real_T(sp, &b_x);
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtConstCTX)sp);
}

static void s_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId, char_T ret[14])
{
  static const int32_T dims[2] = {1, 14};
  emlrtCheckBuiltInR2012b((emlrtConstCTX)sp, msgId, src, "char", false, 2U,
                          (const void *)&dims[0]);
  emlrtImportCharArrayR2015b((emlrtConstCTX)sp, src, &ret[0], 14);
  emlrtDestroyArray(&src);
}

void inv(const emlrtStack *sp, const emxArray_real_T *x, emxArray_real_T *y)
{
  static const int32_T iv[2] = {1, 6};
  static const char_T rfmt[6] = {'%', '1', '4', '.', '6', 'e'};
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack st;
  const mxArray *b_y;
  const mxArray *m;
  const real_T *x_data;
  real_T *y_data;
  int32_T i;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  x_data = x->data;
  if (x->size[0] != x->size[1]) {
    emlrtErrorWithMessageIdR2018a(sp, &d_emlrtRTEI, "Coder:MATLAB:square",
                                  "Coder:MATLAB:square", 0);
  }
  if ((x->size[0] == 0) || (x->size[1] == 0)) {
    int32_T loop_ub;
    i = y->size[0] * y->size[1];
    y->size[0] = x->size[0];
    y->size[1] = x->size[1];
    emxEnsureCapacity_real_T(sp, y, i, &k_emlrtRTEI);
    y_data = y->data;
    loop_ub = x->size[0] * x->size[1];
    for (i = 0; i < loop_ub; i++) {
      y_data[i] = x_data[i];
    }
  } else {
    real_T n1x;
    real_T n1xinv;
    real_T rc;
    st.site = &pc_emlrtRSI;
    invNxN(&st, x, y);
    st.site = &qc_emlrtRSI;
    n1x = b_norm(x);
    n1xinv = b_norm(y);
    rc = 1.0 / (n1x * n1xinv);
    if ((n1x == 0.0) || (n1xinv == 0.0) || (rc == 0.0)) {
      b_st.site = &dd_emlrtRSI;
      warning(&b_st);
    } else if (muDoubleScalarIsNaN(rc) || (rc < 2.2204460492503131E-16)) {
      char_T str[14];
      b_st.site = &ed_emlrtRSI;
      b_y = NULL;
      m = emlrtCreateCharArray(2, &iv[0]);
      emlrtInitCharArrayR2013a(&b_st, 6, m, &rfmt[0]);
      emlrtAssign(&b_y, m);
      c_st.site = &kd_emlrtRSI;
      emlrt_marshallIn(
          &c_st, b_sprintf(&c_st, b_y, emlrt_marshallOut(rc), &c_emlrtMCI),
          "<output of sprintf>", str);
      b_st.site = &ed_emlrtRSI;
      b_warning(&b_st, str);
    }
  }
}

/* End of code generation (inv.c) */

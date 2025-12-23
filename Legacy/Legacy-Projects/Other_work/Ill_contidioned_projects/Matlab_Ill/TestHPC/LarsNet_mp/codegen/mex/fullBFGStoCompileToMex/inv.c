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
static emlrtRSInfo wc_emlrtRSI = {
    21,    /* lineNo */
    "inv", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/matfun/inv.m" /* pathName */
};

static emlrtRSInfo xc_emlrtRSI = {
    22,    /* lineNo */
    "inv", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/matfun/inv.m" /* pathName */
};

static emlrtRSInfo yc_emlrtRSI = {
    173,      /* lineNo */
    "invNxN", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/matfun/inv.m" /* pathName */
};

static emlrtRSInfo kd_emlrtRSI = {
    42,          /* lineNo */
    "checkcond", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/matfun/inv.m" /* pathName */
};

static emlrtRSInfo ld_emlrtRSI = {
    46,          /* lineNo */
    "checkcond", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/matfun/inv.m" /* pathName */
};

static emlrtMCInfo c_emlrtMCI = {
    53,        /* lineNo */
    19,        /* colNo */
    "flt2str", /* fName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/flt2str.m" /* pName */
};

static emlrtRTEInfo e_emlrtRTEI = {
    14,    /* lineNo */
    15,    /* colNo */
    "inv", /* fName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/matfun/inv.m" /* pName */
};

static emlrtRTEInfo k_emlrtRTEI = {
    19,    /* lineNo */
    5,     /* colNo */
    "inv", /* fName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/matfun/inv.m" /* pName */
};

static emlrtRTEInfo l_emlrtRTEI = {
    170,   /* lineNo */
    1,     /* colNo */
    "inv", /* fName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/matfun/inv.m" /* pName */
};

static emlrtRTEInfo m_emlrtRTEI = {
    173,   /* lineNo */
    2,     /* colNo */
    "inv", /* fName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/matfun/inv.m" /* pName */
};

static emlrtRTEInfo n_emlrtRTEI = {
    164,   /* lineNo */
    14,    /* colNo */
    "inv", /* fName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/matfun/inv.m" /* pName */
};

static emlrtRSInfo rd_emlrtRSI = {
    53,        /* lineNo */
    "flt2str", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/flt2str.m" /* pathName */
};

/* Function Declarations */
static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                               const emlrtMsgIdentifier *parentId,
                               char_T y[14]);

static const mxArray *b_sprintf(const emlrtStack *sp, const mxArray *b,
                                const mxArray *c, emlrtMCInfo *location);

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

static const mxArray *b_sprintf(const emlrtStack *sp, const mxArray *b,
                                const mxArray *c, emlrtMCInfo *location)
{
  const mxArray *pArrays[2];
  const mxArray *m;
  pArrays[0] = b;
  pArrays[1] = c;
  return emlrtCallMATLABR2012b((emlrtCTX)sp, 1, &m, 2, &pArrays[0],
                               (const char_T *)"sprintf", true, location);
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
  int32_T ipiv_data[729];
  int32_T p_data[729];
  int32_T ipiv_size[2];
  int32_T b_i;
  int32_T b_n;
  int32_T i;
  int32_T k;
  int32_T n;
  int32_T yk;
  st.prev = sp;
  st.tls = sp->tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  n = x->size[0];
  i = y->size[0] * y->size[1];
  y->size[0] = x->size[0];
  y->size[1] = x->size[1];
  emxEnsureCapacity_real_T(sp, y, i, &l_emlrtRTEI);
  b_n = x->size[0] * x->size[1];
  for (i = 0; i < b_n; i++) {
    y->data[i] = 0.0;
  }
  emxInit_real_T(sp, &b_x, 2, &n_emlrtRTEI, true);
  i = b_x->size[0] * b_x->size[1];
  b_x->size[0] = x->size[0];
  b_x->size[1] = x->size[1];
  emxEnsureCapacity_real_T(sp, b_x, i, &m_emlrtRTEI);
  b_n = x->size[0] * x->size[1];
  for (i = 0; i < b_n; i++) {
    b_x->data[i] = x->data[i];
  }
  st.site = &yc_emlrtRSI;
  xgetrf(&st, x->size[0], x->size[0], b_x, x->size[0], ipiv_data, ipiv_size);
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
      p_data[k - 1] = yk;
    }
  }
  i = ipiv_size[1];
  for (k = 0; k < i; k++) {
    b_n = ipiv_data[k];
    if (b_n > k + 1) {
      yk = p_data[b_n - 1];
      p_data[b_n - 1] = p_data[k];
      p_data[k] = yk;
    }
  }
  for (k = 0; k < n; k++) {
    i = p_data[k];
    y->data[k + y->size[0] * (i - 1)] = 1.0;
    for (yk = k + 1; yk <= n; yk++) {
      if (y->data[(yk + y->size[0] * (i - 1)) - 1] != 0.0) {
        b_n = yk + 1;
        for (b_i = b_n; b_i <= n; b_i++) {
          y->data[(b_i + y->size[0] * (i - 1)) - 1] -=
              y->data[(yk + y->size[0] * (i - 1)) - 1] *
              b_x->data[(b_i + b_x->size[0] * (yk - 1)) - 1];
        }
      }
    }
  }
  xtrsm(x->size[0], x->size[0], b_x, x->size[0], y, x->size[0]);
  emxFree_real_T(&b_x);
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

static void s_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId, char_T ret[14])
{
  static const int32_T dims[2] = {1, 14};
  emlrtCheckBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"char",
                          false, 2U, (void *)&dims[0]);
  emlrtImportCharArrayR2015b((emlrtCTX)sp, src, &ret[0], 14);
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
  real_T n1x;
  real_T n1xinv;
  real_T rc;
  int32_T i;
  int32_T loop_ub;
  char_T str[14];
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  if (x->size[0] != x->size[1]) {
    emlrtErrorWithMessageIdR2018a(sp, &e_emlrtRTEI, "Coder:MATLAB:square",
                                  "Coder:MATLAB:square", 0);
  }
  if ((x->size[0] == 0) || (x->size[1] == 0)) {
    i = y->size[0] * y->size[1];
    y->size[0] = x->size[0];
    y->size[1] = x->size[1];
    emxEnsureCapacity_real_T(sp, y, i, &k_emlrtRTEI);
    loop_ub = x->size[0] * x->size[1];
    for (i = 0; i < loop_ub; i++) {
      y->data[i] = x->data[i];
    }
  } else {
    st.site = &wc_emlrtRSI;
    invNxN(&st, x, y);
    st.site = &xc_emlrtRSI;
    n1x = b_norm(x);
    n1xinv = b_norm(y);
    rc = 1.0 / (n1x * n1xinv);
    if ((n1x == 0.0) || (n1xinv == 0.0) || (rc == 0.0)) {
      b_st.site = &kd_emlrtRSI;
      warning(&b_st);
    } else if (muDoubleScalarIsNaN(rc) || (rc < 2.2204460492503131E-16)) {
      b_st.site = &ld_emlrtRSI;
      b_y = NULL;
      m = emlrtCreateCharArray(2, &iv[0]);
      emlrtInitCharArrayR2013a(&b_st, 6, m, &rfmt[0]);
      emlrtAssign(&b_y, m);
      c_st.site = &rd_emlrtRSI;
      emlrt_marshallIn(
          &c_st, b_sprintf(&c_st, b_y, emlrt_marshallOut(rc), &c_emlrtMCI),
          "<output of sprintf>", str);
      b_st.site = &ld_emlrtRSI;
      b_warning(&b_st, str);
    }
  }
}

/* End of code generation (inv.c) */

/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * xgetrf.c
 *
 * Code generation for function 'xgetrf'
 *
 */

/* Include files */
#include "xgetrf.h"
#include "fullBFGStoCompileToMex_types.h"
#include "rt_nonfinite.h"
#include "lapacke.h"
#include "mwmathutil.h"
#include <stddef.h>

/* Variable Definitions */
static emlrtRSInfo ed_emlrtRSI = {
    27,       /* lineNo */
    "xgetrf", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/+lapack/xgetrf.m" /* pathName */
};

static emlrtRSInfo fd_emlrtRSI = {
    91,             /* lineNo */
    "ceval_xgetrf", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/+lapack/xgetrf.m" /* pathName */
};

static emlrtRTEInfo h_emlrtRTEI = {
    47,          /* lineNo */
    13,          /* colNo */
    "infocheck", /* fName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/+lapack/infocheck.m" /* pName */
};

static emlrtRTEInfo i_emlrtRTEI = {
    44,          /* lineNo */
    13,          /* colNo */
    "infocheck", /* fName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/+lapack/infocheck.m" /* pName */
};

/* Function Definitions */
void xgetrf(const emlrtStack *sp, int32_T m, int32_T n, emxArray_real_T *A,
            int32_T lda, int32_T ipiv_data[], int32_T ipiv_size[2])
{
  static const char_T fname[19] = {'L', 'A', 'P', 'A', 'C', 'K', 'E',
                                   '_', 'd', 'g', 'e', 't', 'r', 'f',
                                   '_', 'w', 'o', 'r', 'k'};
  ptrdiff_t ipiv_t_data[729];
  ptrdiff_t info_t;
  emlrtStack b_st;
  emlrtStack st;
  int32_T info;
  int32_T ipiv_t_size;
  st.prev = sp;
  st.tls = sp->tls;
  st.site = &ed_emlrtRSI;
  b_st.prev = &st;
  b_st.tls = st.tls;
  if ((A->size[0] == 0) || (A->size[1] == 0)) {
    ipiv_size[0] = 1;
    ipiv_size[1] = 0;
  } else {
    ipiv_t_size = muIntScalarMin_sint32(m, n);
    ipiv_t_size = (int16_T)muIntScalarMax_sint32(ipiv_t_size, 1);
    info_t = LAPACKE_dgetrf_work(102, (ptrdiff_t)m, (ptrdiff_t)n, &A->data[0],
                                 (ptrdiff_t)lda, &ipiv_t_data[0]);
    ipiv_size[0] = 1;
    ipiv_size[1] = ipiv_t_size;
    b_st.site = &fd_emlrtRSI;
    info = (int32_T)info_t;
    if (info < 0) {
      if (info == -1010) {
        emlrtErrorWithMessageIdR2018a(&b_st, &i_emlrtRTEI, "MATLAB:nomem",
                                      "MATLAB:nomem", 0);
      } else {
        emlrtErrorWithMessageIdR2018a(
            &b_st, &h_emlrtRTEI, "Coder:toolbox:LAPACKCallErrorInfo",
            "Coder:toolbox:LAPACKCallErrorInfo", 5, 4, 19, &fname[0], 12, info);
      }
    }
    ipiv_t_size--;
    for (info = 0; info <= ipiv_t_size; info++) {
      ipiv_data[info] = (int32_T)ipiv_t_data[info];
    }
  }
}

/* End of code generation (xgetrf.c) */

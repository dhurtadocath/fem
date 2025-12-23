/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * sortIdx.c
 *
 * Code generation for function 'sortIdx'
 *
 */

/* Include files */
#include "sortIdx.h"
#include "eml_int_forloop_overflow_check.h"
#include "fullBFGStoCompileToMex_data.h"
#include "rt_nonfinite.h"
#include "mwmathutil.h"
#include <string.h>

/* Variable Definitions */
static emlrtRSInfo nb_emlrtRSI = {
    105,       /* lineNo */
    "sortIdx", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo ob_emlrtRSI = {
    301,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo pb_emlrtRSI = {
    309,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo qb_emlrtRSI = {
    310,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo rb_emlrtRSI = {
    318,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo sb_emlrtRSI = {
    326,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo tb_emlrtRSI = {
    381,                      /* lineNo */
    "initialize_vector_sort", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo ub_emlrtRSI = {
    409,                      /* lineNo */
    "initialize_vector_sort", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo vb_emlrtRSI = {
    416,                      /* lineNo */
    "initialize_vector_sort", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo wb_emlrtRSI = {
    576,                /* lineNo */
    "merge_pow2_block", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo xb_emlrtRSI = {
    578,                /* lineNo */
    "merge_pow2_block", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo yb_emlrtRSI = {
    606,                /* lineNo */
    "merge_pow2_block", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo ac_emlrtRSI = {
    488,           /* lineNo */
    "merge_block", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo cc_emlrtRSI = {
    496,           /* lineNo */
    "merge_block", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo dc_emlrtRSI = {
    503,           /* lineNo */
    "merge_block", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo ec_emlrtRSI = {
    550,     /* lineNo */
    "merge", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo fc_emlrtRSI = {
    519,     /* lineNo */
    "merge", /* fcnName */
    "/usr/local/MATLAB/R2024a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

/* Function Declarations */
static void merge(const emlrtStack *sp, int32_T idx_data[], real_T x_data[],
                  int32_T offset, int32_T np, int32_T nq, int32_T iwork_data[],
                  real_T xwork_data[]);

static void merge_block(const emlrtStack *sp, int32_T idx_data[],
                        real_T x_data[], int32_T offset, int32_T n,
                        int32_T preSortLevel, int32_T iwork_data[],
                        real_T xwork_data[]);

/* Function Definitions */
static void merge(const emlrtStack *sp, int32_T idx_data[], real_T x_data[],
                  int32_T offset, int32_T np, int32_T nq, int32_T iwork_data[],
                  real_T xwork_data[])
{
  emlrtStack b_st;
  emlrtStack st;
  int32_T j;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  if (nq != 0) {
    int32_T iout;
    int32_T n_tmp;
    int32_T p;
    int32_T q;
    n_tmp = np + nq;
    st.site = &fc_emlrtRSI;
    if (n_tmp > 2147483646) {
      b_st.site = &mb_emlrtRSI;
      check_forloop_overflow_error(&b_st);
    }
    for (j = 0; j < n_tmp; j++) {
      iout = offset + j;
      iwork_data[j] = idx_data[iout];
      xwork_data[j] = x_data[iout];
    }
    p = 0;
    q = np;
    iout = offset - 1;
    int32_T exitg1;
    do {
      exitg1 = 0;
      iout++;
      if (xwork_data[p] <= xwork_data[q]) {
        idx_data[iout] = iwork_data[p];
        x_data[iout] = xwork_data[p];
        if (p + 1 < np) {
          p++;
        } else {
          exitg1 = 1;
        }
      } else {
        idx_data[iout] = iwork_data[q];
        x_data[iout] = xwork_data[q];
        if (q + 1 < n_tmp) {
          q++;
        } else {
          q = iout - p;
          st.site = &ec_emlrtRSI;
          if ((p + 1 <= np) && (np > 2147483646)) {
            b_st.site = &mb_emlrtRSI;
            check_forloop_overflow_error(&b_st);
          }
          for (j = p + 1; j <= np; j++) {
            iout = q + j;
            idx_data[iout] = iwork_data[j - 1];
            x_data[iout] = xwork_data[j - 1];
          }
          exitg1 = 1;
        }
      }
    } while (exitg1 == 0);
  }
}

static void merge_block(const emlrtStack *sp, int32_T idx_data[],
                        real_T x_data[], int32_T offset, int32_T n,
                        int32_T preSortLevel, int32_T iwork_data[],
                        real_T xwork_data[])
{
  emlrtStack st;
  int32_T bLen;
  int32_T nPairs;
  int32_T nTail;
  st.prev = sp;
  st.tls = sp->tls;
  nPairs = n >> preSortLevel;
  bLen = 1 << preSortLevel;
  while (nPairs > 1) {
    int32_T tailOffset;
    if ((nPairs & 1) != 0) {
      nPairs--;
      tailOffset = bLen * nPairs;
      nTail = n - tailOffset;
      if (nTail > bLen) {
        st.site = &ac_emlrtRSI;
        merge(&st, idx_data, x_data, offset + tailOffset, bLen, nTail - bLen,
              iwork_data, xwork_data);
      }
    }
    tailOffset = bLen << 1;
    nPairs >>= 1;
    for (nTail = 0; nTail < nPairs; nTail++) {
      st.site = &cc_emlrtRSI;
      merge(&st, idx_data, x_data, offset + nTail * tailOffset, bLen, bLen,
            iwork_data, xwork_data);
    }
    bLen = tailOffset;
  }
  if (n > bLen) {
    st.site = &dc_emlrtRSI;
    merge(&st, idx_data, x_data, offset, bLen, n - bLen, iwork_data,
          xwork_data);
  }
}

int32_T sortIdx(const emlrtStack *sp, real_T x_data[], const int32_T *x_size,
                int32_T idx_data[])
{
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack st;
  int32_T iwork_data[570];
  int32_T b;
  int32_T b_b;
  int32_T i1;
  int32_T idx_size;
  int32_T k;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  d_st.prev = &c_st;
  d_st.tls = c_st.tls;
  idx_size = *x_size;
  if (idx_size - 1 >= 0) {
    memset(&idx_data[0], 0, (uint32_T)idx_size * sizeof(int32_T));
  }
  if (*x_size != 0) {
    real_T xwork_data[570];
    real_T x4[4];
    int32_T bLen;
    int32_T bLen2;
    int32_T i;
    int32_T i2;
    int32_T i3;
    int32_T i4;
    int32_T ib;
    int32_T idx_tmp;
    int32_T wOffset_tmp;
    int16_T idx4[4];
    st.site = &nb_emlrtRSI;
    if (idx_size - 1 >= 0) {
      memset(&iwork_data[0], 0, (uint32_T)idx_size * sizeof(int32_T));
    }
    b_st.site = &ob_emlrtRSI;
    x4[0] = 0.0;
    idx4[0] = 0;
    x4[1] = 0.0;
    idx4[1] = 0;
    x4[2] = 0.0;
    idx4[2] = 0;
    x4[3] = 0.0;
    idx4[3] = 0;
    bLen2 = 0;
    ib = 0;
    c_st.site = &tb_emlrtRSI;
    for (k = 0; k < idx_size; k++) {
      if (muDoubleScalarIsNaN(x_data[k])) {
        idx_tmp = (idx_size - bLen2) - 1;
        idx_data[idx_tmp] = k + 1;
        xwork_data[idx_tmp] = x_data[k];
        bLen2++;
      } else {
        ib++;
        idx4[ib - 1] = (int16_T)(k + 1);
        x4[ib - 1] = x_data[k];
        if (ib == 4) {
          real_T d;
          real_T d1;
          ib = k - bLen2;
          if (x4[0] <= x4[1]) {
            i1 = 1;
            i2 = 2;
          } else {
            i1 = 2;
            i2 = 1;
          }
          if (x4[2] <= x4[3]) {
            i3 = 3;
            i4 = 4;
          } else {
            i3 = 4;
            i4 = 3;
          }
          d = x4[i3 - 1];
          d1 = x4[i1 - 1];
          if (d1 <= d) {
            d1 = x4[i2 - 1];
            if (d1 <= d) {
              i = i1;
              bLen = i2;
              i1 = i3;
              i2 = i4;
            } else if (d1 <= x4[i4 - 1]) {
              i = i1;
              bLen = i3;
              i1 = i2;
              i2 = i4;
            } else {
              i = i1;
              bLen = i3;
              i1 = i4;
            }
          } else {
            d = x4[i4 - 1];
            if (d1 <= d) {
              if (x4[i2 - 1] <= d) {
                i = i3;
                bLen = i1;
                i1 = i2;
                i2 = i4;
              } else {
                i = i3;
                bLen = i1;
                i1 = i4;
              }
            } else {
              i = i3;
              bLen = i4;
            }
          }
          idx_data[ib - 3] = idx4[i - 1];
          idx_data[ib - 2] = idx4[bLen - 1];
          idx_data[ib - 1] = idx4[i1 - 1];
          idx_data[ib] = idx4[i2 - 1];
          x_data[ib - 3] = x4[i - 1];
          x_data[ib - 2] = x4[bLen - 1];
          x_data[ib - 1] = x4[i1 - 1];
          x_data[ib] = x4[i2 - 1];
          ib = 0;
        }
      }
    }
    wOffset_tmp = *x_size - bLen2;
    if (ib > 0) {
      int8_T perm[4];
      perm[1] = 0;
      perm[2] = 0;
      perm[3] = 0;
      if (ib == 1) {
        perm[0] = 1;
      } else if (ib == 2) {
        if (x4[0] <= x4[1]) {
          perm[0] = 1;
          perm[1] = 2;
        } else {
          perm[0] = 2;
          perm[1] = 1;
        }
      } else if (x4[0] <= x4[1]) {
        if (x4[1] <= x4[2]) {
          perm[0] = 1;
          perm[1] = 2;
          perm[2] = 3;
        } else if (x4[0] <= x4[2]) {
          perm[0] = 1;
          perm[1] = 3;
          perm[2] = 2;
        } else {
          perm[0] = 3;
          perm[1] = 1;
          perm[2] = 2;
        }
      } else if (x4[0] <= x4[2]) {
        perm[0] = 2;
        perm[1] = 1;
        perm[2] = 3;
      } else if (x4[1] <= x4[2]) {
        perm[0] = 2;
        perm[1] = 3;
        perm[2] = 1;
      } else {
        perm[0] = 3;
        perm[1] = 2;
        perm[2] = 1;
      }
      c_st.site = &ub_emlrtRSI;
      if (ib > 2147483646) {
        d_st.site = &mb_emlrtRSI;
        check_forloop_overflow_error(&d_st);
      }
      i = (uint8_T)ib;
      for (k = 0; k < i; k++) {
        idx_tmp = (wOffset_tmp - ib) + k;
        bLen = perm[k];
        idx_data[idx_tmp] = idx4[bLen - 1];
        x_data[idx_tmp] = x4[bLen - 1];
      }
    }
    i1 = bLen2 >> 1;
    c_st.site = &vb_emlrtRSI;
    for (k = 0; k < i1; k++) {
      ib = wOffset_tmp + k;
      i2 = idx_data[ib];
      idx_tmp = (idx_size - k) - 1;
      idx_data[ib] = idx_data[idx_tmp];
      idx_data[idx_tmp] = i2;
      x_data[ib] = xwork_data[idx_tmp];
      x_data[idx_tmp] = xwork_data[ib];
    }
    if ((bLen2 & 1) != 0) {
      i = wOffset_tmp + i1;
      x_data[i] = xwork_data[i];
    }
    ib = 2;
    if (wOffset_tmp > 1) {
      if (*x_size >= 256) {
        int32_T nBlocks;
        nBlocks = wOffset_tmp >> 8;
        if (nBlocks > 0) {
          b_st.site = &pb_emlrtRSI;
          for (b = 0; b < nBlocks; b++) {
            real_T xwork[256];
            int16_T iwork[256];
            b_st.site = &qb_emlrtRSI;
            i4 = (b << 8) - 1;
            for (b_b = 0; b_b < 6; b_b++) {
              bLen = 1 << (b_b + 2);
              bLen2 = bLen << 1;
              i = 256 >> (b_b + 3);
              c_st.site = &wb_emlrtRSI;
              for (k = 0; k < i; k++) {
                i2 = (i4 + k * bLen2) + 1;
                c_st.site = &xb_emlrtRSI;
                for (i1 = 0; i1 < bLen2; i1++) {
                  ib = i2 + i1;
                  iwork[i1] = (int16_T)idx_data[ib];
                  xwork[i1] = x_data[ib];
                }
                i3 = 0;
                i1 = bLen;
                ib = i2 - 1;
                int32_T exitg1;
                do {
                  exitg1 = 0;
                  ib++;
                  if (xwork[i3] <= xwork[i1]) {
                    idx_data[ib] = iwork[i3];
                    x_data[ib] = xwork[i3];
                    if (i3 + 1 < bLen) {
                      i3++;
                    } else {
                      exitg1 = 1;
                    }
                  } else {
                    idx_data[ib] = iwork[i1];
                    x_data[ib] = xwork[i1];
                    if (i1 + 1 < bLen2) {
                      i1++;
                    } else {
                      ib -= i3;
                      c_st.site = &yb_emlrtRSI;
                      for (i1 = i3 + 1; i1 <= bLen; i1++) {
                        idx_tmp = ib + i1;
                        idx_data[idx_tmp] = iwork[i1 - 1];
                        x_data[idx_tmp] = xwork[i1 - 1];
                      }
                      exitg1 = 1;
                    }
                  }
                } while (exitg1 == 0);
              }
            }
          }
          ib = nBlocks << 8;
          i1 = wOffset_tmp - ib;
          if (i1 > 0) {
            b_st.site = &rb_emlrtRSI;
            merge_block(&b_st, idx_data, x_data, ib, i1, 2, iwork_data,
                        xwork_data);
          }
          ib = 8;
        }
      }
      b_st.site = &sb_emlrtRSI;
      merge_block(&b_st, idx_data, x_data, 0, wOffset_tmp, ib, iwork_data,
                  xwork_data);
    }
  }
  return idx_size;
}

/* End of code generation (sortIdx.c) */

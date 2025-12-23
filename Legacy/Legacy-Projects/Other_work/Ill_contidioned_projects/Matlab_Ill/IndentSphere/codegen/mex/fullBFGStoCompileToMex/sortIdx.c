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
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo ob_emlrtRSI = {
    308,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo pb_emlrtRSI = {
    316,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo qb_emlrtRSI = {
    317,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo rb_emlrtRSI = {
    325,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo sb_emlrtRSI = {
    333,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo tb_emlrtRSI = {
    392,                      /* lineNo */
    "initialize_vector_sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo ub_emlrtRSI = {
    420,                      /* lineNo */
    "initialize_vector_sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo vb_emlrtRSI = {
    427,                      /* lineNo */
    "initialize_vector_sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo wb_emlrtRSI = {
    587,                /* lineNo */
    "merge_pow2_block", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo xb_emlrtRSI = {
    589,                /* lineNo */
    "merge_pow2_block", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo yb_emlrtRSI = {
    617,                /* lineNo */
    "merge_pow2_block", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo ac_emlrtRSI = {
    499,           /* lineNo */
    "merge_block", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo cc_emlrtRSI = {
    507,           /* lineNo */
    "merge_block", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo dc_emlrtRSI = {
    514,           /* lineNo */
    "merge_block", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo ec_emlrtRSI = {
    561,     /* lineNo */
    "merge", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
                                                                           */
};

static emlrtRSInfo fc_emlrtRSI = {
    530,     /* lineNo */
    "merge", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/eml/+coder/+internal/sortIdx.m" /* pathName
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
  real_T xwork_data[570];
  int32_T iwork_data[570];
  int32_T b;
  int32_T b_b;
  int32_T ib;
  int32_T idx_size;
  int32_T k;
  int32_T quartetOffset;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  d_st.prev = &c_st;
  d_st.tls = c_st.tls;
  ib = (int16_T)*x_size;
  idx_size = (int16_T)*x_size;
  if (ib - 1 >= 0) {
    memset(&idx_data[0], 0, (uint32_T)ib * sizeof(int32_T));
  }
  if (*x_size != 0) {
    real_T x4[4];
    int32_T bLen2;
    int32_T i;
    int32_T i2;
    int32_T i3;
    int32_T i4;
    int32_T idx_tmp;
    int32_T n;
    int32_T wOffset_tmp;
    int16_T idx4[4];
    st.site = &nb_emlrtRSI;
    b_st.site = &ob_emlrtRSI;
    n = *x_size;
    x4[0] = 0.0;
    idx4[0] = 0;
    x4[1] = 0.0;
    idx4[1] = 0;
    x4[2] = 0.0;
    idx4[2] = 0;
    x4[3] = 0.0;
    idx4[3] = 0;
    if (ib - 1 >= 0) {
      memset(&iwork_data[0], 0, (uint32_T)ib * sizeof(int32_T));
    }
    ib = *x_size;
    if (ib - 1 >= 0) {
      memset(&xwork_data[0], 0, (uint32_T)ib * sizeof(real_T));
    }
    bLen2 = 0;
    ib = 0;
    c_st.site = &tb_emlrtRSI;
    for (k = 0; k < n; k++) {
      if (muDoubleScalarIsNaN(x_data[k])) {
        idx_tmp = (n - bLen2) - 1;
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
          int8_T b_i2;
          int8_T b_i3;
          int8_T b_i4;
          int8_T i1;
          quartetOffset = k - bLen2;
          if (x4[0] <= x4[1]) {
            ib = 1;
            i2 = 2;
          } else {
            ib = 2;
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
          d1 = x4[ib - 1];
          if (d1 <= d) {
            d1 = x4[i2 - 1];
            if (d1 <= d) {
              i1 = (int8_T)ib;
              b_i2 = (int8_T)i2;
              b_i3 = (int8_T)i3;
              b_i4 = (int8_T)i4;
            } else if (d1 <= x4[i4 - 1]) {
              i1 = (int8_T)ib;
              b_i2 = (int8_T)i3;
              b_i3 = (int8_T)i2;
              b_i4 = (int8_T)i4;
            } else {
              i1 = (int8_T)ib;
              b_i2 = (int8_T)i3;
              b_i3 = (int8_T)i4;
              b_i4 = (int8_T)i2;
            }
          } else {
            d = x4[i4 - 1];
            if (d1 <= d) {
              if (x4[i2 - 1] <= d) {
                i1 = (int8_T)i3;
                b_i2 = (int8_T)ib;
                b_i3 = (int8_T)i2;
                b_i4 = (int8_T)i4;
              } else {
                i1 = (int8_T)i3;
                b_i2 = (int8_T)ib;
                b_i3 = (int8_T)i4;
                b_i4 = (int8_T)i2;
              }
            } else {
              i1 = (int8_T)i3;
              b_i2 = (int8_T)i4;
              b_i3 = (int8_T)ib;
              b_i4 = (int8_T)i2;
            }
          }
          idx_data[quartetOffset - 3] = idx4[i1 - 1];
          idx_data[quartetOffset - 2] = idx4[b_i2 - 1];
          idx_data[quartetOffset - 1] = idx4[b_i3 - 1];
          idx_data[quartetOffset] = idx4[b_i4 - 1];
          x_data[quartetOffset - 3] = x4[i1 - 1];
          x_data[quartetOffset - 2] = x4[b_i2 - 1];
          x_data[quartetOffset - 1] = x4[b_i3 - 1];
          x_data[quartetOffset] = x4[b_i4 - 1];
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
        idx_tmp = perm[k] - 1;
        quartetOffset = (wOffset_tmp - ib) + k;
        idx_data[quartetOffset] = idx4[idx_tmp];
        x_data[quartetOffset] = x4[idx_tmp];
      }
    }
    ib = bLen2 >> 1;
    c_st.site = &vb_emlrtRSI;
    for (k = 0; k < ib; k++) {
      quartetOffset = wOffset_tmp + k;
      i2 = idx_data[quartetOffset];
      idx_tmp = (n - k) - 1;
      idx_data[quartetOffset] = idx_data[idx_tmp];
      idx_data[idx_tmp] = i2;
      x_data[quartetOffset] = xwork_data[idx_tmp];
      x_data[idx_tmp] = xwork_data[quartetOffset];
    }
    if ((bLen2 & 1) != 0) {
      i = wOffset_tmp + ib;
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
              n = 1 << (b_b + 2);
              bLen2 = n << 1;
              i = 256 >> (b_b + 3);
              c_st.site = &wb_emlrtRSI;
              for (k = 0; k < i; k++) {
                i2 = (i4 + k * bLen2) + 1;
                c_st.site = &xb_emlrtRSI;
                for (quartetOffset = 0; quartetOffset < bLen2;
                     quartetOffset++) {
                  ib = i2 + quartetOffset;
                  iwork[quartetOffset] = (int16_T)idx_data[ib];
                  xwork[quartetOffset] = x_data[ib];
                }
                i3 = 0;
                quartetOffset = n;
                ib = i2 - 1;
                int32_T exitg1;
                do {
                  exitg1 = 0;
                  ib++;
                  if (xwork[i3] <= xwork[quartetOffset]) {
                    idx_data[ib] = iwork[i3];
                    x_data[ib] = xwork[i3];
                    if (i3 + 1 < n) {
                      i3++;
                    } else {
                      exitg1 = 1;
                    }
                  } else {
                    idx_data[ib] = iwork[quartetOffset];
                    x_data[ib] = xwork[quartetOffset];
                    if (quartetOffset + 1 < bLen2) {
                      quartetOffset++;
                    } else {
                      ib -= i3;
                      c_st.site = &yb_emlrtRSI;
                      for (quartetOffset = i3 + 1; quartetOffset <= n;
                           quartetOffset++) {
                        idx_tmp = ib + quartetOffset;
                        idx_data[idx_tmp] = iwork[quartetOffset - 1];
                        x_data[idx_tmp] = xwork[quartetOffset - 1];
                      }
                      exitg1 = 1;
                    }
                  }
                } while (exitg1 == 0);
              }
            }
          }
          ib = nBlocks << 8;
          quartetOffset = wOffset_tmp - ib;
          if (quartetOffset > 0) {
            b_st.site = &rb_emlrtRSI;
            merge_block(&b_st, idx_data, x_data, ib, quartetOffset, 2,
                        iwork_data, xwork_data);
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

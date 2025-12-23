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
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/sortIdx.m" /* pathName */
};

static emlrtRSInfo ob_emlrtRSI = {
    308,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/sortIdx.m" /* pathName */
};

static emlrtRSInfo pb_emlrtRSI = {
    316,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/sortIdx.m" /* pathName */
};

static emlrtRSInfo qb_emlrtRSI = {
    317,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/sortIdx.m" /* pathName */
};

static emlrtRSInfo rb_emlrtRSI = {
    325,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/sortIdx.m" /* pathName */
};

static emlrtRSInfo sb_emlrtRSI = {
    333,                /* lineNo */
    "block_merge_sort", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/sortIdx.m" /* pathName */
};

static emlrtRSInfo tb_emlrtRSI = {
    392,                      /* lineNo */
    "initialize_vector_sort", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/sortIdx.m" /* pathName */
};

static emlrtRSInfo ub_emlrtRSI = {
    420,                      /* lineNo */
    "initialize_vector_sort", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/sortIdx.m" /* pathName */
};

static emlrtRSInfo vb_emlrtRSI = {
    427,                      /* lineNo */
    "initialize_vector_sort", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/sortIdx.m" /* pathName */
};

static emlrtRSInfo ac_emlrtRSI = {
    499,           /* lineNo */
    "merge_block", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/sortIdx.m" /* pathName */
};

static emlrtRSInfo cc_emlrtRSI = {
    507,           /* lineNo */
    "merge_block", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/sortIdx.m" /* pathName */
};

static emlrtRSInfo dc_emlrtRSI = {
    514,           /* lineNo */
    "merge_block", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/sortIdx.m" /* pathName */
};

static emlrtRSInfo ec_emlrtRSI = {
    561,     /* lineNo */
    "merge", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/sortIdx.m" /* pathName */
};

static emlrtRSInfo fc_emlrtRSI = {
    530,     /* lineNo */
    "merge", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/eml/+coder/+internal/sortIdx.m" /* pathName */
};

/* Function Declarations */
static void merge(const emlrtStack *sp, int32_T idx_data[], real_T x_data[],
                  int32_T offset, int32_T np, int32_T nq, int32_T iwork_data[],
                  real_T xwork_data[]);

static void merge_block(const emlrtStack *sp, int32_T idx_data[],
                        real_T x_data[], int32_T offset, int32_T n,
                        int32_T preSortLevel, int32_T iwork_data[],
                        real_T xwork_data[]);

static void merge_pow2_block(int32_T idx_data[], real_T x_data[],
                             int32_T offset);

/* Function Definitions */
static void merge(const emlrtStack *sp, int32_T idx_data[], real_T x_data[],
                  int32_T offset, int32_T np, int32_T nq, int32_T iwork_data[],
                  real_T xwork_data[])
{
  emlrtStack b_st;
  emlrtStack st;
  int32_T exitg1;
  int32_T iout;
  int32_T j;
  int32_T n_tmp;
  int32_T p;
  int32_T q;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  if (nq != 0) {
    n_tmp = np + nq;
    st.site = &fc_emlrtRSI;
    if ((1 <= n_tmp) && (n_tmp > 2147483646)) {
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
  int32_T tailOffset;
  st.prev = sp;
  st.tls = sp->tls;
  nPairs = n >> preSortLevel;
  bLen = 1 << preSortLevel;
  while (nPairs > 1) {
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

static void merge_pow2_block(int32_T idx_data[], real_T x_data[],
                             int32_T offset)
{
  real_T xwork[256];
  int32_T iwork[256];
  int32_T b;
  int32_T bLen;
  int32_T bLen2;
  int32_T blockOffset;
  int32_T exitg1;
  int32_T iout;
  int32_T j;
  int32_T k;
  int32_T nPairs;
  int32_T p;
  int32_T q;
  for (b = 0; b < 6; b++) {
    bLen = 1 << (b + 2);
    bLen2 = bLen << 1;
    nPairs = 256 >> (b + 3);
    for (k = 0; k < nPairs; k++) {
      blockOffset = offset + k * bLen2;
      for (j = 0; j < bLen2; j++) {
        iout = blockOffset + j;
        iwork[j] = idx_data[iout];
        xwork[j] = x_data[iout];
      }
      p = 0;
      q = bLen;
      iout = blockOffset - 1;
      do {
        exitg1 = 0;
        iout++;
        if (xwork[p] <= xwork[q]) {
          idx_data[iout] = iwork[p];
          x_data[iout] = xwork[p];
          if (p + 1 < bLen) {
            p++;
          } else {
            exitg1 = 1;
          }
        } else {
          idx_data[iout] = iwork[q];
          x_data[iout] = xwork[q];
          if (q + 1 < bLen2) {
            q++;
          } else {
            iout -= p;
            for (j = p + 1; j <= bLen; j++) {
              q = iout + j;
              idx_data[q] = iwork[j - 1];
              x_data[q] = xwork[j - 1];
            }
            exitg1 = 1;
          }
        }
      } while (exitg1 == 0);
    }
  }
}

void sortIdx(const emlrtStack *sp, real_T x_data[], const int32_T *x_size,
             int32_T idx_data[], int32_T *idx_size)
{
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack st;
  real_T xwork_data[729];
  real_T x4[4];
  real_T d;
  real_T d1;
  int32_T iwork_data[729];
  int32_T i2;
  int32_T i3;
  int32_T i4;
  int32_T ib;
  int32_T k;
  int32_T n;
  int32_T nNaNs;
  int32_T quartetOffset;
  int16_T idx4[4];
  int8_T perm[4];
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  d_st.prev = &c_st;
  d_st.tls = c_st.tls;
  ib = (int16_T)*x_size;
  *idx_size = (int16_T)*x_size;
  if (0 <= ib - 1) {
    memset(&idx_data[0], 0, ib * sizeof(int32_T));
  }
  if (*x_size != 0) {
    st.site = &nb_emlrtRSI;
    n = *x_size;
    b_st.site = &ob_emlrtRSI;
    x4[0] = 0.0;
    idx4[0] = 0;
    x4[1] = 0.0;
    idx4[1] = 0;
    x4[2] = 0.0;
    idx4[2] = 0;
    x4[3] = 0.0;
    idx4[3] = 0;
    if (0 <= ib - 1) {
      memset(&iwork_data[0], 0, ib * sizeof(int32_T));
    }
    ib = *x_size;
    if (0 <= ib - 1) {
      memset(&xwork_data[0], 0, ib * sizeof(real_T));
    }
    nNaNs = 0;
    ib = -1;
    c_st.site = &tb_emlrtRSI;
    for (k = 0; k < n; k++) {
      if (muDoubleScalarIsNaN(x_data[k])) {
        i3 = (n - nNaNs) - 1;
        idx_data[i3] = k + 1;
        xwork_data[i3] = x_data[k];
        nNaNs++;
      } else {
        ib++;
        idx4[ib] = (int16_T)(k + 1);
        x4[ib] = x_data[k];
        if (ib + 1 == 4) {
          quartetOffset = k - nNaNs;
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
          d = x4[ib - 1];
          d1 = x4[i3 - 1];
          if (d <= d1) {
            d = x4[i2 - 1];
            if (d <= d1) {
              perm[0] = (int8_T)ib;
              perm[1] = (int8_T)i2;
              perm[2] = (int8_T)i3;
              perm[3] = (int8_T)i4;
            } else if (d <= x4[i4 - 1]) {
              perm[0] = (int8_T)ib;
              perm[1] = (int8_T)i3;
              perm[2] = (int8_T)i2;
              perm[3] = (int8_T)i4;
            } else {
              perm[0] = (int8_T)ib;
              perm[1] = (int8_T)i3;
              perm[2] = (int8_T)i4;
              perm[3] = (int8_T)i2;
            }
          } else {
            d1 = x4[i4 - 1];
            if (d <= d1) {
              if (x4[i2 - 1] <= d1) {
                perm[0] = (int8_T)i3;
                perm[1] = (int8_T)ib;
                perm[2] = (int8_T)i2;
                perm[3] = (int8_T)i4;
              } else {
                perm[0] = (int8_T)i3;
                perm[1] = (int8_T)ib;
                perm[2] = (int8_T)i4;
                perm[3] = (int8_T)i2;
              }
            } else {
              perm[0] = (int8_T)i3;
              perm[1] = (int8_T)i4;
              perm[2] = (int8_T)ib;
              perm[3] = (int8_T)i2;
            }
          }
          idx_data[quartetOffset - 3] = idx4[perm[0] - 1];
          idx_data[quartetOffset - 2] = idx4[perm[1] - 1];
          idx_data[quartetOffset - 1] = idx4[perm[2] - 1];
          idx_data[quartetOffset] = idx4[perm[3] - 1];
          x_data[quartetOffset - 3] = x4[perm[0] - 1];
          x_data[quartetOffset - 2] = x4[perm[1] - 1];
          x_data[quartetOffset - 1] = x4[perm[2] - 1];
          x_data[quartetOffset] = x4[perm[3] - 1];
          ib = -1;
        }
      }
    }
    i4 = (n - nNaNs) - 1;
    if (ib + 1 > 0) {
      perm[1] = 0;
      perm[2] = 0;
      perm[3] = 0;
      if (ib + 1 == 1) {
        perm[0] = 1;
      } else if (ib + 1 == 2) {
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
      if (ib + 1 > 2147483646) {
        d_st.site = &mb_emlrtRSI;
        check_forloop_overflow_error(&d_st);
      }
      for (k = 0; k <= ib; k++) {
        i3 = perm[k] - 1;
        quartetOffset = (i4 - ib) + k;
        idx_data[quartetOffset] = idx4[i3];
        x_data[quartetOffset] = x4[i3];
      }
    }
    ib = (nNaNs >> 1) + 1;
    c_st.site = &vb_emlrtRSI;
    for (k = 0; k <= ib - 2; k++) {
      quartetOffset = (i4 + k) + 1;
      i2 = idx_data[quartetOffset];
      i3 = (n - k) - 1;
      idx_data[quartetOffset] = idx_data[i3];
      idx_data[i3] = i2;
      x_data[quartetOffset] = xwork_data[i3];
      x_data[i3] = xwork_data[quartetOffset];
    }
    if ((nNaNs & 1) != 0) {
      ib += i4;
      x_data[ib] = xwork_data[ib];
    }
    i2 = n - nNaNs;
    ib = 2;
    if (i2 > 1) {
      if (n >= 256) {
        quartetOffset = i2 >> 8;
        if (quartetOffset > 0) {
          b_st.site = &pb_emlrtRSI;
          for (ib = 0; ib < quartetOffset; ib++) {
            b_st.site = &qb_emlrtRSI;
            merge_pow2_block(idx_data, x_data, ib << 8);
          }
          ib = quartetOffset << 8;
          quartetOffset = i2 - ib;
          if (quartetOffset > 0) {
            b_st.site = &rb_emlrtRSI;
            merge_block(&b_st, idx_data, x_data, ib, quartetOffset, 2,
                        iwork_data, xwork_data);
          }
          ib = 8;
        }
      }
      b_st.site = &sb_emlrtRSI;
      merge_block(&b_st, idx_data, x_data, 0, i2, ib, iwork_data, xwork_data);
    }
  }
}

/* End of code generation (sortIdx.c) */

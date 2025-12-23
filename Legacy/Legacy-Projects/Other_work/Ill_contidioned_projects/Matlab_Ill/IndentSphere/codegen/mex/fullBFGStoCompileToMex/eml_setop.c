/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * eml_setop.c
 *
 * Code generation for function 'eml_setop'
 *
 */

/* Include files */
#include "eml_setop.h"
#include "rt_nonfinite.h"
#include "mwmathutil.h"

/* Variable Definitions */
static emlrtRTEInfo emlrtRTEI =
    {
        239,          /* lineNo */
        13,           /* colNo */
        "do_vectors", /* fName */
        "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/ops/private/"
        "eml_setop.m" /* pName */
};

static emlrtRTEInfo b_emlrtRTEI =
    {
        242,          /* lineNo */
        13,           /* colNo */
        "do_vectors", /* fName */
        "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/ops/private/"
        "eml_setop.m" /* pName */
};

/* Function Definitions */
int32_T do_vectors(const emlrtStack *sp, const real_T b_data[], int32_T b_size,
                   real_T c_data[], int32_T c_size[2], int32_T ia_data[],
                   int32_T *ib_size)
{
  real_T bk;
  int32_T b_ialast;
  int32_T dim;
  int32_T ia_size;
  int32_T ialast;
  int32_T iblast;
  int32_T nc;
  int32_T nia;
  boolean_T exitg1;
  boolean_T y;
  c_size[0] = 1;
  *ib_size = 0;
  y = true;
  nia = 0;
  exitg1 = false;
  while ((!exitg1) && (nia < 569)) {
    if (nia + 1 > nia + 2) {
      y = false;
      exitg1 = true;
    } else {
      nia++;
    }
  }
  if (!y) {
    emlrtErrorWithMessageIdR2018a(sp, &emlrtRTEI,
                                  "Coder:toolbox:eml_setop_unsortedA",
                                  "Coder:toolbox:eml_setop_unsortedA", 0);
  }
  y = true;
  dim = 2;
  if (b_size != 1) {
    dim = 1;
  }
  if (b_size != 0) {
    if (dim <= 1) {
      nia = b_size;
    } else {
      nia = 1;
    }
    if (nia != 1) {
      if (dim == 2) {
        ialast = -1;
      } else {
        ialast = 0;
      }
      nia = 0;
      exitg1 = false;
      while ((!exitg1) && (nia <= ialast)) {
        boolean_T exitg2;
        if (dim == 1) {
          iblast = b_size - 1;
        } else {
          iblast = b_size;
        }
        nia = 0;
        exitg2 = false;
        while ((!exitg2) && (nia <= iblast - 1)) {
          int16_T subs[2];
          subs[0] = (int16_T)(nia + 1);
          subs[1] = 1;
          subs[dim - 1]++;
          bk = b_data[subs[0] - 1];
          if ((b_data[nia] <= bk) || muDoubleScalarIsNaN(bk)) {
            nia++;
          } else {
            y = false;
            exitg2 = true;
          }
        }
        if (!y) {
          exitg1 = true;
        } else {
          nia = 1;
        }
      }
    }
  }
  if (!y) {
    emlrtErrorWithMessageIdR2018a(sp, &b_emlrtRTEI,
                                  "Coder:toolbox:eml_setop_unsortedB",
                                  "Coder:toolbox:eml_setop_unsortedB", 0);
  }
  nc = 0;
  nia = -1;
  dim = 0;
  ialast = 1;
  iblast = 1;
  while ((ialast <= 570) && (iblast <= b_size)) {
    int32_T ak;
    b_ialast = ialast;
    ak = ialast;
    while ((b_ialast < 570) && (b_ialast + 1 == ialast)) {
      b_ialast++;
    }
    ialast = b_ialast;
    bk = b_data[iblast - 1];
    while ((iblast < b_size) && (b_data[iblast] == bk)) {
      iblast++;
    }
    if (ak == bk) {
      ialast = b_ialast + 1;
      dim = b_ialast;
      iblast++;
    } else if (muDoubleScalarIsNaN(bk) || (ak < bk)) {
      nc++;
      nia++;
      c_data[nc - 1] = ak;
      ia_data[nia] = dim + 1;
      ialast = b_ialast + 1;
      dim = b_ialast;
    } else {
      iblast++;
    }
  }
  while (ialast <= 570) {
    b_ialast = ialast;
    while ((b_ialast < 570) && (b_ialast + 1 == ialast)) {
      b_ialast++;
    }
    nc++;
    nia++;
    c_data[nc - 1] = ((real_T)ialast - 1.0) + 1.0;
    ia_data[nia] = dim + 1;
    ialast = b_ialast + 1;
    dim = b_ialast;
  }
  if (nia + 1 < 1) {
    nia = -1;
  }
  ia_size = nia + 1;
  if (nc < 1) {
    c_size[1] = 0;
  } else {
    c_size[1] = nc;
  }
  return ia_size;
}

/* End of code generation (eml_setop.c) */

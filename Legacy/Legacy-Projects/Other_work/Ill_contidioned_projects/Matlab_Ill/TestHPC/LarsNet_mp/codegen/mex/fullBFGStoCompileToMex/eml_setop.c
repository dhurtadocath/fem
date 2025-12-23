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
#include <math.h>

/* Variable Definitions */
static emlrtRTEInfo b_emlrtRTEI = {
    216,          /* lineNo */
    13,           /* colNo */
    "do_vectors", /* fName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/ops/private/eml_setop.m" /* pName */
};

static emlrtRTEInfo c_emlrtRTEI = {
    219,          /* lineNo */
    13,           /* colNo */
    "do_vectors", /* fName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/ops/private/eml_setop.m" /* pName */
};

/* Function Declarations */
static real_T skip_to_last_equal_value(int32_T *k, const real_T x[729]);

/* Function Definitions */
static real_T skip_to_last_equal_value(int32_T *k, const real_T x[729])
{
  real_T absx;
  real_T xk;
  int32_T exponent;
  boolean_T exitg1;
  boolean_T p;
  xk = x[*k - 1];
  exitg1 = false;
  while ((!exitg1) && (*k < 729)) {
    absx = muDoubleScalarAbs(xk / 2.0);
    if ((!muDoubleScalarIsInf(absx)) && (!muDoubleScalarIsNaN(absx))) {
      if (absx <= 2.2250738585072014E-308) {
        absx = 4.94065645841247E-324;
      } else {
        frexp(absx, &exponent);
        absx = ldexp(1.0, exponent - 53);
      }
    } else {
      absx = rtNaN;
    }
    if ((muDoubleScalarAbs(xk - x[*k]) < absx) ||
        (muDoubleScalarIsInf(x[*k]) && muDoubleScalarIsInf(xk) &&
         ((x[*k] > 0.0) == (xk > 0.0)))) {
      p = true;
    } else {
      p = false;
    }
    if (p) {
      (*k)++;
    } else {
      exitg1 = true;
    }
  }
  return xk;
}

void do_vectors(const emlrtStack *sp, const real_T b_data[], int32_T b_size,
                real_T c_data[], int32_T c_size[2], int32_T ia_data[],
                int32_T *ia_size, int32_T *ib_size)
{
  real_T dv[729];
  real_T absx;
  real_T ak;
  real_T bk;
  int32_T b_exponent;
  int32_T b_n;
  int32_T dim;
  int32_T exponent;
  int32_T i;
  int32_T ialast;
  int32_T iblast;
  int32_T n;
  int32_T nc;
  int16_T subs[2];
  boolean_T exitg1;
  boolean_T exitg2;
  boolean_T y;
  c_size[0] = 1;
  *ib_size = 0;
  y = true;
  iblast = 0;
  exitg1 = false;
  while ((!exitg1) && (iblast < 728)) {
    y = ((iblast + 1 <= iblast + 2) && y);
    if (!y) {
      exitg1 = true;
    } else {
      iblast++;
    }
  }
  if (!y) {
    emlrtErrorWithMessageIdR2018a(sp, &b_emlrtRTEI,
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
      iblast = b_size;
    } else {
      iblast = 1;
    }
    if (iblast != 1) {
      if (dim == 2) {
        n = -1;
      } else {
        n = 0;
      }
      iblast = 0;
      exitg1 = false;
      while ((!exitg1) && (iblast <= n)) {
        if (dim == 1) {
          b_n = b_size - 1;
        } else {
          b_n = b_size;
        }
        iblast = 0;
        exitg2 = false;
        while ((!exitg2) && (iblast <= b_n - 1)) {
          subs[0] = (int16_T)(iblast + 1);
          subs[1] = 1;
          subs[dim - 1]++;
          absx = b_data[subs[0] - 1];
          if ((b_data[iblast] <= absx) || muDoubleScalarIsNaN(absx)) {
            iblast++;
          } else {
            y = false;
            exitg2 = true;
          }
        }
        if (!y) {
          exitg1 = true;
        } else {
          iblast = 1;
        }
      }
    }
  }
  if (!y) {
    emlrtErrorWithMessageIdR2018a(sp, &c_emlrtRTEI,
                                  "Coder:toolbox:eml_setop_unsortedB",
                                  "Coder:toolbox:eml_setop_unsortedB", 0);
  }
  nc = 0;
  n = -1;
  b_n = 0;
  ialast = 1;
  iblast = 1;
  while ((ialast <= 729) && (iblast <= b_size)) {
    dim = ialast;
    for (i = 0; i < 729; i++) {
      dv[i] = (real_T)i + 1.0;
    }
    ak = skip_to_last_equal_value(&dim, dv);
    ialast = dim;
    bk = b_data[iblast - 1];
    exitg1 = false;
    while ((!exitg1) && (iblast < b_size)) {
      absx = muDoubleScalarAbs(bk / 2.0);
      if ((!muDoubleScalarIsInf(absx)) && (!muDoubleScalarIsNaN(absx))) {
        if (absx <= 2.2250738585072014E-308) {
          absx = 4.94065645841247E-324;
        } else {
          frexp(absx, &exponent);
          absx = ldexp(1.0, exponent - 53);
        }
      } else {
        absx = rtNaN;
      }
      if ((muDoubleScalarAbs(bk - b_data[iblast]) < absx) ||
          (muDoubleScalarIsInf(b_data[iblast]) && muDoubleScalarIsInf(bk) &&
           ((b_data[iblast] > 0.0) == (bk > 0.0)))) {
        y = true;
      } else {
        y = false;
      }
      if (y) {
        iblast++;
      } else {
        exitg1 = true;
      }
    }
    absx = muDoubleScalarAbs(bk / 2.0);
    if ((!muDoubleScalarIsInf(absx)) && (!muDoubleScalarIsNaN(absx))) {
      if (absx <= 2.2250738585072014E-308) {
        absx = 4.94065645841247E-324;
      } else {
        frexp(absx, &b_exponent);
        absx = ldexp(1.0, b_exponent - 53);
      }
    } else {
      absx = rtNaN;
    }
    if ((muDoubleScalarAbs(bk - ak) < absx) ||
        (muDoubleScalarIsInf(ak) && muDoubleScalarIsInf(bk) &&
         ((ak > 0.0) == (bk > 0.0)))) {
      y = true;
    } else {
      y = false;
    }
    if (y) {
      ialast = dim + 1;
      b_n = dim;
      iblast++;
    } else {
      if (muDoubleScalarIsNaN(bk)) {
        y = !muDoubleScalarIsNaN(ak);
      } else if (muDoubleScalarIsNaN(ak)) {
        y = false;
      } else {
        y = (ak < bk);
      }
      if (y) {
        nc++;
        n++;
        c_data[nc - 1] = ak;
        ia_data[n] = b_n + 1;
        ialast = dim + 1;
        b_n = dim;
      } else {
        iblast++;
      }
    }
  }
  if (ialast <= 729) {
    for (i = 0; i < 729; i++) {
      dv[i] = (real_T)i + 1.0;
    }
  }
  while (ialast <= 729) {
    iblast = ialast;
    ak = skip_to_last_equal_value(&iblast, dv);
    nc++;
    n++;
    c_data[nc - 1] = ak;
    ia_data[n] = b_n + 1;
    ialast = iblast + 1;
    b_n = iblast;
  }
  if (1 > n + 1) {
    n = -1;
  }
  *ia_size = n + 1;
  if (1 > nc) {
    c_size[1] = 0;
  } else {
    c_size[1] = nc;
  }
}

/* End of code generation (eml_setop.c) */

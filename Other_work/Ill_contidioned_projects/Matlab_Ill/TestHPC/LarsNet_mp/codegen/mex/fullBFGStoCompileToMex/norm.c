/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * norm.c
 *
 * Code generation for function 'norm'
 *
 */

/* Include files */
#include "norm.h"
#include "fullBFGStoCompileToMex_types.h"
#include "rt_nonfinite.h"
#include "blas.h"
#include "mwmathutil.h"
#include <stddef.h>

/* Function Definitions */
real_T b_norm(const emxArray_real_T *x)
{
  real_T s;
  real_T y;
  int32_T b_i;
  int32_T i;
  int32_T j;
  boolean_T exitg1;
  if ((x->size[0] == 0) || (x->size[1] == 0)) {
    y = 0.0;
  } else if ((x->size[0] == 1) || (x->size[1] == 1)) {
    y = 0.0;
    i = x->size[0] * x->size[1];
    for (j = 0; j < i; j++) {
      y += muDoubleScalarAbs(x->data[j]);
    }
  } else {
    y = 0.0;
    j = 0;
    exitg1 = false;
    while ((!exitg1) && (j <= x->size[1] - 1)) {
      s = 0.0;
      i = x->size[0];
      for (b_i = 0; b_i < i; b_i++) {
        s += muDoubleScalarAbs(x->data[b_i + x->size[0] * j]);
      }
      if (muDoubleScalarIsNaN(s)) {
        y = rtNaN;
        exitg1 = true;
      } else {
        if (s > y) {
          y = s;
        }
        j++;
      }
    }
  }
  return y;
}

real_T c_norm(const real_T x_data[], int32_T x_size)
{
  ptrdiff_t incx_t;
  ptrdiff_t n_t;
  real_T y;
  if (x_size == 0) {
    y = 0.0;
  } else {
    n_t = (ptrdiff_t)x_size;
    incx_t = (ptrdiff_t)1;
    y = dnrm2(&n_t, &x_data[0], &incx_t);
  }
  return y;
}

/* End of code generation (norm.c) */

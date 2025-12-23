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
  const real_T *x_data;
  real_T y;
  int32_T b_i;
  int32_T k;
  boolean_T MATRIX_INPUT_AND_P_IS_ONE;
  boolean_T VECTOR_INPUT_AND_P_IS_NUMERIC;
  x_data = x->data;
  VECTOR_INPUT_AND_P_IS_NUMERIC = false;
  MATRIX_INPUT_AND_P_IS_ONE = false;
  if ((x->size[0] == 1) || (x->size[1] == 1)) {
    VECTOR_INPUT_AND_P_IS_NUMERIC = true;
  } else {
    MATRIX_INPUT_AND_P_IS_ONE = true;
  }
  if ((x->size[0] == 0) || (x->size[1] == 0)) {
    y = 0.0;
  } else if (MATRIX_INPUT_AND_P_IS_ONE) {
    boolean_T exitg1;
    y = 0.0;
    k = 0;
    exitg1 = false;
    while ((!exitg1) && (k <= x->size[1] - 1)) {
      real_T s;
      int32_T i;
      s = 0.0;
      i = x->size[0];
      for (b_i = 0; b_i < i; b_i++) {
        s += muDoubleScalarAbs(x_data[b_i + x->size[0] * k]);
      }
      if (muDoubleScalarIsNaN(s)) {
        y = rtNaN;
        exitg1 = true;
      } else {
        if (s > y) {
          y = s;
        }
        k++;
      }
    }
  } else if (VECTOR_INPUT_AND_P_IS_NUMERIC) {
    int32_T i;
    y = 0.0;
    i = x->size[0] * x->size[1];
    for (k = 0; k < i; k++) {
      y += muDoubleScalarAbs(x_data[k]);
    }
  } else {
    y = rtNaN;
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
    y = dnrm2(&n_t, (real_T *)&x_data[0], &incx_t);
  }
  return y;
}

/* End of code generation (norm.c) */

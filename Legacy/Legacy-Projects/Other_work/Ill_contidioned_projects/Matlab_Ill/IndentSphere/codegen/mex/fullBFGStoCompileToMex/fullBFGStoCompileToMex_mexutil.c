/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * fullBFGStoCompileToMex_mexutil.c
 *
 * Code generation for function 'fullBFGStoCompileToMex_mexutil'
 *
 */

/* Include files */
#include "fullBFGStoCompileToMex_mexutil.h"
#include "rt_nonfinite.h"

/* Function Definitions */
const mxArray *emlrt_marshallOut(const real_T u)
{
  const mxArray *m;
  const mxArray *y;
  y = NULL;
  m = emlrtCreateDoubleScalar(u);
  emlrtAssign(&y, m);
  return y;
}

/* End of code generation (fullBFGStoCompileToMex_mexutil.c) */

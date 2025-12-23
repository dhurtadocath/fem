/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * fullBFGStoCompileToMex_initialize.c
 *
 * Code generation for function 'fullBFGStoCompileToMex_initialize'
 *
 */

/* Include files */
#include "fullBFGStoCompileToMex_initialize.h"
#include "_coder_fullBFGStoCompileToMex_mex.h"
#include "fullBFGStoCompileToMex_data.h"
#include "rt_nonfinite.h"

/* Function Declarations */
static void fullBFGStoCompileToMex_once(void);

/* Function Definitions */
static void fullBFGStoCompileToMex_once(void)
{
  mex_InitInfAndNan();
}

void fullBFGStoCompileToMex_initialize(void)
{
  emlrtStack st = {
      NULL, /* site */
      NULL, /* tls */
      NULL  /* prev */
  };
  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtBreakCheckR2012bFlagVar = emlrtGetBreakCheckFlagAddressR2022b(&st);
  emlrtClearAllocCountR2012b(&st, false, 0U, NULL);
  emlrtEnterRtStackR2012b(&st);
  if (emlrtFirstTimeR2012b(emlrtRootTLSGlobal)) {
    fullBFGStoCompileToMex_once();
  }
}

/* End of code generation (fullBFGStoCompileToMex_initialize.c) */

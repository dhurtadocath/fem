/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * fullBFGStoCompileToMex_terminate.c
 *
 * Code generation for function 'fullBFGStoCompileToMex_terminate'
 *
 */

/* Include files */
#include "fullBFGStoCompileToMex_terminate.h"
#include "_coder_fullBFGStoCompileToMex_mex.h"
#include "fullBFGStoCompileToMex_data.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void fullBFGStoCompileToMex_atexit(void)
{
  emlrtStack st = {
      NULL, /* site */
      NULL, /* tls */
      NULL  /* prev */
  };
  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtEnterRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  emlrtExitTimeCleanup(&emlrtContextGlobal);
}

void fullBFGStoCompileToMex_terminate(void)
{
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

/* End of code generation (fullBFGStoCompileToMex_terminate.c) */

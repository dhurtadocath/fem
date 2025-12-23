/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_fullBFGStoCompileToMex_mex.c
 *
 * Code generation for function '_coder_fullBFGStoCompileToMex_mex'
 *
 */

/* Include files */
#include "_coder_fullBFGStoCompileToMex_mex.h"
#include "_coder_fullBFGStoCompileToMex_api.h"
#include "fullBFGStoCompileToMex_data.h"
#include "fullBFGStoCompileToMex_initialize.h"
#include "fullBFGStoCompileToMex_terminate.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void fullBFGStoCompileToMex_mexFunction(int32_T nlhs, mxArray *plhs[3],
                                        int32_T nrhs, const mxArray *prhs[16])
{
  emlrtStack st = {
      NULL, /* site */
      NULL, /* tls */
      NULL  /* prev */
  };
  const mxArray *b_prhs[16];
  const mxArray *outputs[3];
  int32_T i;
  int32_T i1;
  st.tls = emlrtRootTLSGlobal;
  /* Check for proper number of arguments. */
  if (nrhs != 16) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:WrongNumberOfInputs", 5, 12, 16, 4,
                        22, "fullBFGStoCompileToMex");
  }
  if (nlhs > 3) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:TooManyOutputArguments", 3, 4, 22,
                        "fullBFGStoCompileToMex");
  }
  /* Call the function. */
  for (i = 0; i < 16; i++) {
    b_prhs[i] = prhs[i];
  }
  fullBFGStoCompileToMex_api(b_prhs, nlhs, outputs);
  /* Copy over outputs to the caller. */
  if (nlhs < 1) {
    i1 = 1;
  } else {
    i1 = nlhs;
  }
  emlrtReturnArrays(i1, &plhs[0], &outputs[0]);
}

void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs,
                 const mxArray *prhs[])
{
  mexAtExit(&fullBFGStoCompileToMex_atexit);
  /* Module initialization. */
  fullBFGStoCompileToMex_initialize();
  /* Dispatch the entry-point. */
  fullBFGStoCompileToMex_mexFunction(nlhs, plhs, nrhs, prhs);
  /* Module termination. */
  fullBFGStoCompileToMex_terminate();
}

emlrtCTX mexFunctionCreateRootTLS(void)
{
  emlrtCreateRootTLSR2022a(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1,
                           NULL, "UTF-8", true);
  return emlrtRootTLSGlobal;
}

/* End of code generation (_coder_fullBFGStoCompileToMex_mex.c) */

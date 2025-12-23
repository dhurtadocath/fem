/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * fullBFGStoCompileToMex.h
 *
 * Code generation for function 'fullBFGStoCompileToMex'
 *
 */

#pragma once

/* Include files */
#include "rtwtypes.h"
#include "emlrt.h"
#include "mex.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Function Declarations */
void fullBFGStoCompileToMex(const emlrtStack *sp, real_T u[729],
                            const real_T X[729], const real_T conn[740],
                            const boolean_T is_free[729],
                            const boolean_T actives[222], real_T mu,
                            real_T ratio, real_T k_pen, real_T Cx, real_T Cy,
                            real_T Cz, real_T R, const real_T slaves[222],
                            const real_T dofs[729], const real_T s[444],
                            const real_T sh[444], real_T *m_new, real_T *iter);

/* End of code generation (fullBFGStoCompileToMex.h) */

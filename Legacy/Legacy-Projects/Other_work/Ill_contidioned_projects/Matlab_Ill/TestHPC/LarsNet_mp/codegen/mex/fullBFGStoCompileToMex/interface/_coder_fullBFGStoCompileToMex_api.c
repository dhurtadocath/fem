/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_fullBFGStoCompileToMex_api.c
 *
 * Code generation for function '_coder_fullBFGStoCompileToMex_api'
 *
 */

/* Include files */
#include "_coder_fullBFGStoCompileToMex_api.h"
#include "fullBFGStoCompileToMex.h"
#include "fullBFGStoCompileToMex_data.h"
#include "fullBFGStoCompileToMex_mexutil.h"
#include "rt_nonfinite.h"

/* Function Declarations */
static real_T (*ab_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                    const emlrtMsgIdentifier *msgId))[222];

static void b_emlrt_marshallOut(const real_T u[729], const mxArray *y);

static real_T (*bb_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                    const emlrtMsgIdentifier *msgId))[444];

static real_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const char_T *identifier))[729];

static real_T (*d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[729];

static real_T (*e_emlrt_marshallIn(const emlrtStack *sp, const mxArray *X,
                                   const char_T *identifier))[729];

static real_T (*f_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[729];

static real_T (*g_emlrt_marshallIn(const emlrtStack *sp, const mxArray *conn,
                                   const char_T *identifier))[740];

static real_T (*h_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[740];

static boolean_T (*i_emlrt_marshallIn(const emlrtStack *sp,
                                      const mxArray *is_free,
                                      const char_T *identifier))[729];

static boolean_T (*j_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                      const emlrtMsgIdentifier *parentId))[729];

static boolean_T (*k_emlrt_marshallIn(const emlrtStack *sp,
                                      const mxArray *actives,
                                      const char_T *identifier))[222];

static boolean_T (*l_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                      const emlrtMsgIdentifier *parentId))[222];

static real_T m_emlrt_marshallIn(const emlrtStack *sp, const mxArray *mu,
                                 const char_T *identifier);

static real_T n_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                 const emlrtMsgIdentifier *parentId);

static real_T (*o_emlrt_marshallIn(const emlrtStack *sp, const mxArray *slaves,
                                   const char_T *identifier))[222];

static real_T (*p_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[222];

static real_T (*q_emlrt_marshallIn(const emlrtStack *sp, const mxArray *s,
                                   const char_T *identifier))[444];

static real_T (*r_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[444];

static real_T (*t_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[729];

static real_T (*u_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[729];

static real_T (*v_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[740];

static boolean_T (*w_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                      const emlrtMsgIdentifier *msgId))[729];

static boolean_T (*x_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                      const emlrtMsgIdentifier *msgId))[222];

static real_T y_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                 const emlrtMsgIdentifier *msgId);

/* Function Definitions */
static real_T (*ab_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                    const emlrtMsgIdentifier *msgId))[222]
{
  static const int32_T dims[2] = {1, 222};
  real_T(*ret)[222];
  emlrtCheckBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"double",
                          false, 2U, (void *)&dims[0]);
  ret = (real_T(*)[222])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static void b_emlrt_marshallOut(const real_T u[729], const mxArray *y)
{
  static const int32_T i = 729;
  emlrtMxSetData((mxArray *)y, (void *)&u[0]);
  emlrtSetDimensions((mxArray *)y, &i, 1);
}

static real_T (*bb_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                    const emlrtMsgIdentifier *msgId))[444]
{
  static const int32_T dims[2] = {222, 2};
  real_T(*ret)[444];
  emlrtCheckBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"double",
                          false, 2U, (void *)&dims[0]);
  ret = (real_T(*)[444])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static real_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const char_T *identifier))[729]
{
  emlrtMsgIdentifier thisId;
  real_T(*y)[729];
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = d_emlrt_marshallIn(sp, emlrtAlias(u), &thisId);
  emlrtDestroyArray(&u);
  return y;
}

static real_T (*d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[729]
{
  real_T(*y)[729];
  y = t_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static real_T (*e_emlrt_marshallIn(const emlrtStack *sp, const mxArray *X,
                                   const char_T *identifier))[729]
{
  emlrtMsgIdentifier thisId;
  real_T(*y)[729];
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = f_emlrt_marshallIn(sp, emlrtAlias(X), &thisId);
  emlrtDestroyArray(&X);
  return y;
}

static real_T (*f_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[729]
{
  real_T(*y)[729];
  y = u_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static real_T (*g_emlrt_marshallIn(const emlrtStack *sp, const mxArray *conn,
                                   const char_T *identifier))[740]
{
  emlrtMsgIdentifier thisId;
  real_T(*y)[740];
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = h_emlrt_marshallIn(sp, emlrtAlias(conn), &thisId);
  emlrtDestroyArray(&conn);
  return y;
}

static real_T (*h_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[740]
{
  real_T(*y)[740];
  y = v_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static boolean_T (*i_emlrt_marshallIn(const emlrtStack *sp,
                                      const mxArray *is_free,
                                      const char_T *identifier))[729]
{
  emlrtMsgIdentifier thisId;
  boolean_T(*y)[729];
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = j_emlrt_marshallIn(sp, emlrtAlias(is_free), &thisId);
  emlrtDestroyArray(&is_free);
  return y;
}

static boolean_T (*j_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                      const emlrtMsgIdentifier *parentId))[729]
{
  boolean_T(*y)[729];
  y = w_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static boolean_T (*k_emlrt_marshallIn(const emlrtStack *sp,
                                      const mxArray *actives,
                                      const char_T *identifier))[222]
{
  emlrtMsgIdentifier thisId;
  boolean_T(*y)[222];
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = l_emlrt_marshallIn(sp, emlrtAlias(actives), &thisId);
  emlrtDestroyArray(&actives);
  return y;
}

static boolean_T (*l_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                      const emlrtMsgIdentifier *parentId))[222]
{
  boolean_T(*y)[222];
  y = x_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static real_T m_emlrt_marshallIn(const emlrtStack *sp, const mxArray *mu,
                                 const char_T *identifier)
{
  emlrtMsgIdentifier thisId;
  real_T y;
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = n_emlrt_marshallIn(sp, emlrtAlias(mu), &thisId);
  emlrtDestroyArray(&mu);
  return y;
}

static real_T n_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                 const emlrtMsgIdentifier *parentId)
{
  real_T y;
  y = y_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static real_T (*o_emlrt_marshallIn(const emlrtStack *sp, const mxArray *slaves,
                                   const char_T *identifier))[222]
{
  emlrtMsgIdentifier thisId;
  real_T(*y)[222];
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = p_emlrt_marshallIn(sp, emlrtAlias(slaves), &thisId);
  emlrtDestroyArray(&slaves);
  return y;
}

static real_T (*p_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[222]
{
  real_T(*y)[222];
  y = ab_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static real_T (*q_emlrt_marshallIn(const emlrtStack *sp, const mxArray *s,
                                   const char_T *identifier))[444]
{
  emlrtMsgIdentifier thisId;
  real_T(*y)[444];
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = r_emlrt_marshallIn(sp, emlrtAlias(s), &thisId);
  emlrtDestroyArray(&s);
  return y;
}

static real_T (*r_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[444]
{
  real_T(*y)[444];
  y = bb_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static real_T (*t_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[729]
{
  static const int32_T dims = 729;
  real_T(*ret)[729];
  emlrtCheckBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"double",
                          false, 1U, (void *)&dims);
  ret = (real_T(*)[729])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static real_T (*u_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[729]
{
  static const int32_T dims[2] = {243, 3};
  real_T(*ret)[729];
  emlrtCheckBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"double",
                          false, 2U, (void *)&dims[0]);
  ret = (real_T(*)[729])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static real_T (*v_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[740]
{
  static const int32_T dims[2] = {370, 2};
  real_T(*ret)[740];
  emlrtCheckBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"double",
                          false, 2U, (void *)&dims[0]);
  ret = (real_T(*)[740])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static boolean_T (*w_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                      const emlrtMsgIdentifier *msgId))[729]
{
  static const int32_T dims[2] = {243, 3};
  boolean_T(*ret)[729];
  emlrtCheckBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"logical",
                          false, 2U, (void *)&dims[0]);
  ret = (boolean_T(*)[729])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static boolean_T (*x_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                      const emlrtMsgIdentifier *msgId))[222]
{
  static const int32_T dims[2] = {1, 222};
  boolean_T(*ret)[222];
  emlrtCheckBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"logical",
                          false, 2U, (void *)&dims[0]);
  ret = (boolean_T(*)[222])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static real_T y_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                 const emlrtMsgIdentifier *msgId)
{
  static const int32_T dims = 0;
  real_T ret;
  emlrtCheckBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"double",
                          false, 0U, (void *)&dims);
  ret = *(real_T *)emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

void fullBFGStoCompileToMex_api(const mxArray *const prhs[16], int32_T nlhs,
                                const mxArray *plhs[3])
{
  emlrtStack st = {
      NULL, /* site */
      NULL, /* tls */
      NULL  /* prev */
  };
  const mxArray *prhs_copy_idx_0;
  real_T(*conn)[740];
  real_T(*X)[729];
  real_T(*dofs)[729];
  real_T(*u)[729];
  real_T(*s)[444];
  real_T(*sh)[444];
  real_T(*slaves)[222];
  real_T Cx;
  real_T Cy;
  real_T Cz;
  real_T R;
  real_T iter;
  real_T k_pen;
  real_T m_new;
  real_T mu;
  real_T ratio;
  boolean_T(*is_free)[729];
  boolean_T(*actives)[222];
  st.tls = emlrtRootTLSGlobal;
  prhs_copy_idx_0 = emlrtProtectR2012b(prhs[0], 0, true, -1);
  /* Marshall function inputs */
  u = c_emlrt_marshallIn(&st, emlrtAlias(prhs_copy_idx_0), "u");
  X = e_emlrt_marshallIn(&st, emlrtAlias(prhs[1]), "X");
  conn = g_emlrt_marshallIn(&st, emlrtAlias(prhs[2]), "conn");
  is_free = i_emlrt_marshallIn(&st, emlrtAlias(prhs[3]), "is_free");
  actives = k_emlrt_marshallIn(&st, emlrtAlias(prhs[4]), "actives");
  mu = m_emlrt_marshallIn(&st, emlrtAliasP(prhs[5]), "mu");
  ratio = m_emlrt_marshallIn(&st, emlrtAliasP(prhs[6]), "ratio");
  k_pen = m_emlrt_marshallIn(&st, emlrtAliasP(prhs[7]), "k_pen");
  Cx = m_emlrt_marshallIn(&st, emlrtAliasP(prhs[8]), "Cx");
  Cy = m_emlrt_marshallIn(&st, emlrtAliasP(prhs[9]), "Cy");
  Cz = m_emlrt_marshallIn(&st, emlrtAliasP(prhs[10]), "Cz");
  R = m_emlrt_marshallIn(&st, emlrtAliasP(prhs[11]), "R");
  slaves = o_emlrt_marshallIn(&st, emlrtAlias(prhs[12]), "slaves");
  dofs = e_emlrt_marshallIn(&st, emlrtAlias(prhs[13]), "dofs");
  s = q_emlrt_marshallIn(&st, emlrtAlias(prhs[14]), "s");
  sh = q_emlrt_marshallIn(&st, emlrtAlias(prhs[15]), "sh");
  /* Invoke the target function */
  fullBFGStoCompileToMex(&st, *u, *X, *conn, *is_free, *actives, mu, ratio,
                         k_pen, Cx, Cy, Cz, R, *slaves, *dofs, *s, *sh, &m_new,
                         &iter);
  /* Marshall function outputs */
  b_emlrt_marshallOut(*u, prhs_copy_idx_0);
  plhs[0] = prhs_copy_idx_0;
  if (nlhs > 1) {
    plhs[1] = emlrt_marshallOut(m_new);
  }
  if (nlhs > 2) {
    plhs[2] = emlrt_marshallOut(iter);
  }
}

/* End of code generation (_coder_fullBFGStoCompileToMex_api.c) */

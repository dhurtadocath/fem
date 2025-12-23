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
                                    const emlrtMsgIdentifier *msgId))[152];

static void b_emlrt_marshallOut(const real_T u[570], const mxArray *y);

static real_T (*bb_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                    const emlrtMsgIdentifier *msgId))[304];

static real_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const char_T *identifier))[570];

static real_T (*d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[570];

static real_T (*e_emlrt_marshallIn(const emlrtStack *sp, const mxArray *X,
                                   const char_T *identifier))[570];

static real_T (*f_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[570];

static real_T (*g_emlrt_marshallIn(const emlrtStack *sp, const mxArray *conn,
                                   const char_T *identifier))[532];

static real_T (*h_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[532];

static boolean_T (*i_emlrt_marshallIn(const emlrtStack *sp,
                                      const mxArray *is_free,
                                      const char_T *identifier))[570];

static boolean_T (*j_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                      const emlrtMsgIdentifier *parentId))[570];

static boolean_T (*k_emlrt_marshallIn(const emlrtStack *sp,
                                      const mxArray *actives,
                                      const char_T *identifier))[152];

static boolean_T (*l_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                      const emlrtMsgIdentifier *parentId))[152];

static real_T m_emlrt_marshallIn(const emlrtStack *sp, const mxArray *mu,
                                 const char_T *identifier);

static real_T n_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                 const emlrtMsgIdentifier *parentId);

static real_T (*o_emlrt_marshallIn(const emlrtStack *sp, const mxArray *slaves,
                                   const char_T *identifier))[152];

static real_T (*p_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[152];

static real_T (*q_emlrt_marshallIn(const emlrtStack *sp, const mxArray *s,
                                   const char_T *identifier))[304];

static real_T (*r_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[304];

static real_T (*t_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[570];

static real_T (*u_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[570];

static real_T (*v_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[532];

static boolean_T (*w_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                      const emlrtMsgIdentifier *msgId))[570];

static boolean_T (*x_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                      const emlrtMsgIdentifier *msgId))[152];

static real_T y_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                 const emlrtMsgIdentifier *msgId);

/* Function Definitions */
static real_T (*ab_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                    const emlrtMsgIdentifier *msgId))[152]
{
  static const int32_T dims[2] = {1, 152};
  real_T(*ret)[152];
  int32_T iv[2];
  boolean_T bv[2] = {false, false};
  emlrtCheckVsBuiltInR2012b((emlrtConstCTX)sp, msgId, src, "double", false, 2U,
                            (const void *)&dims[0], &bv[0], &iv[0]);
  ret = (real_T(*)[152])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static void b_emlrt_marshallOut(const real_T u[570], const mxArray *y)
{
  static const int32_T i = 570;
  emlrtMxSetData((mxArray *)y, (void *)&u[0]);
  emlrtSetDimensions((mxArray *)y, &i, 1);
}

static real_T (*bb_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                    const emlrtMsgIdentifier *msgId))[304]
{
  static const int32_T dims[2] = {152, 2};
  real_T(*ret)[304];
  int32_T iv[2];
  boolean_T bv[2] = {false, false};
  emlrtCheckVsBuiltInR2012b((emlrtConstCTX)sp, msgId, src, "double", false, 2U,
                            (const void *)&dims[0], &bv[0], &iv[0]);
  ret = (real_T(*)[304])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static real_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const char_T *identifier))[570]
{
  emlrtMsgIdentifier thisId;
  real_T(*y)[570];
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = d_emlrt_marshallIn(sp, emlrtAlias(u), &thisId);
  emlrtDestroyArray(&u);
  return y;
}

static real_T (*d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[570]
{
  real_T(*y)[570];
  y = t_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static real_T (*e_emlrt_marshallIn(const emlrtStack *sp, const mxArray *X,
                                   const char_T *identifier))[570]
{
  emlrtMsgIdentifier thisId;
  real_T(*y)[570];
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = f_emlrt_marshallIn(sp, emlrtAlias(X), &thisId);
  emlrtDestroyArray(&X);
  return y;
}

static real_T (*f_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[570]
{
  real_T(*y)[570];
  y = u_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static real_T (*g_emlrt_marshallIn(const emlrtStack *sp, const mxArray *conn,
                                   const char_T *identifier))[532]
{
  emlrtMsgIdentifier thisId;
  real_T(*y)[532];
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = h_emlrt_marshallIn(sp, emlrtAlias(conn), &thisId);
  emlrtDestroyArray(&conn);
  return y;
}

static real_T (*h_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[532]
{
  real_T(*y)[532];
  y = v_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static boolean_T (*i_emlrt_marshallIn(const emlrtStack *sp,
                                      const mxArray *is_free,
                                      const char_T *identifier))[570]
{
  emlrtMsgIdentifier thisId;
  boolean_T(*y)[570];
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = j_emlrt_marshallIn(sp, emlrtAlias(is_free), &thisId);
  emlrtDestroyArray(&is_free);
  return y;
}

static boolean_T (*j_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                      const emlrtMsgIdentifier *parentId))[570]
{
  boolean_T(*y)[570];
  y = w_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static boolean_T (*k_emlrt_marshallIn(const emlrtStack *sp,
                                      const mxArray *actives,
                                      const char_T *identifier))[152]
{
  emlrtMsgIdentifier thisId;
  boolean_T(*y)[152];
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = l_emlrt_marshallIn(sp, emlrtAlias(actives), &thisId);
  emlrtDestroyArray(&actives);
  return y;
}

static boolean_T (*l_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                      const emlrtMsgIdentifier *parentId))[152]
{
  boolean_T(*y)[152];
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
                                   const char_T *identifier))[152]
{
  emlrtMsgIdentifier thisId;
  real_T(*y)[152];
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = p_emlrt_marshallIn(sp, emlrtAlias(slaves), &thisId);
  emlrtDestroyArray(&slaves);
  return y;
}

static real_T (*p_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[152]
{
  real_T(*y)[152];
  y = ab_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static real_T (*q_emlrt_marshallIn(const emlrtStack *sp, const mxArray *s,
                                   const char_T *identifier))[304]
{
  emlrtMsgIdentifier thisId;
  real_T(*y)[304];
  thisId.fIdentifier = (const char_T *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = r_emlrt_marshallIn(sp, emlrtAlias(s), &thisId);
  emlrtDestroyArray(&s);
  return y;
}

static real_T (*r_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[304]
{
  real_T(*y)[304];
  y = bb_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static real_T (*t_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[570]
{
  static const int32_T dims = 570;
  real_T(*ret)[570];
  int32_T i;
  boolean_T b = false;
  emlrtCheckVsBuiltInR2012b((emlrtConstCTX)sp, msgId, src, "double", false, 1U,
                            (const void *)&dims, &b, &i);
  ret = (real_T(*)[570])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static real_T (*u_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[570]
{
  static const int32_T dims[2] = {190, 3};
  real_T(*ret)[570];
  int32_T iv[2];
  boolean_T bv[2] = {false, false};
  emlrtCheckVsBuiltInR2012b((emlrtConstCTX)sp, msgId, src, "double", false, 2U,
                            (const void *)&dims[0], &bv[0], &iv[0]);
  ret = (real_T(*)[570])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static real_T (*v_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[532]
{
  static const int32_T dims[2] = {266, 2};
  real_T(*ret)[532];
  int32_T iv[2];
  boolean_T bv[2] = {false, false};
  emlrtCheckVsBuiltInR2012b((emlrtConstCTX)sp, msgId, src, "double", false, 2U,
                            (const void *)&dims[0], &bv[0], &iv[0]);
  ret = (real_T(*)[532])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static boolean_T (*w_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                      const emlrtMsgIdentifier *msgId))[570]
{
  static const int32_T dims[2] = {190, 3};
  int32_T iv[2];
  boolean_T(*ret)[570];
  boolean_T bv[2] = {false, false};
  emlrtCheckVsBuiltInR2012b((emlrtConstCTX)sp, msgId, src, "logical", false, 2U,
                            (const void *)&dims[0], &bv[0], &iv[0]);
  ret = (boolean_T(*)[570])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static boolean_T (*x_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                      const emlrtMsgIdentifier *msgId))[152]
{
  static const int32_T dims[2] = {1, 152};
  int32_T iv[2];
  boolean_T(*ret)[152];
  boolean_T bv[2] = {false, false};
  emlrtCheckVsBuiltInR2012b((emlrtConstCTX)sp, msgId, src, "logical", false, 2U,
                            (const void *)&dims[0], &bv[0], &iv[0]);
  ret = (boolean_T(*)[152])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static real_T y_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                 const emlrtMsgIdentifier *msgId)
{
  static const int32_T dims = 0;
  real_T ret;
  emlrtCheckBuiltInR2012b((emlrtConstCTX)sp, msgId, src, "double", false, 0U,
                          (const void *)&dims);
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
  real_T(*X)[570];
  real_T(*dofs)[570];
  real_T(*u)[570];
  real_T(*conn)[532];
  real_T(*s)[304];
  real_T(*sh)[304];
  real_T(*slaves)[152];
  real_T Cx;
  real_T Cy;
  real_T Cz;
  real_T R;
  real_T iter;
  real_T k_pen;
  real_T m_new;
  real_T mu;
  real_T ratio;
  boolean_T(*is_free)[570];
  boolean_T(*actives)[152];
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

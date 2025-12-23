/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * fullBFGStoCompileToMex.c
 *
 * Code generation for function 'fullBFGStoCompileToMex'
 *
 */

/* Include files */
#include "fullBFGStoCompileToMex.h"
#include "eml_mtimes_helper.h"
#include "eml_setop.h"
#include "eye.h"
#include "find.h"
#include "fullBFGStoCompileToMex_data.h"
#include "fullBFGStoCompileToMex_emxutil.h"
#include "fullBFGStoCompileToMex_types.h"
#include "inv.h"
#include "mtimes.h"
#include "norm.h"
#include "rt_nonfinite.h"
#include "sort.h"
#include "mwmathutil.h"
#include <emmintrin.h>
#include <string.h>

/* Variable Definitions */
static emlrtRSInfo emlrtRSI =
    {
        22,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo b_emlrtRSI =
    {
        24,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo c_emlrtRSI =
    {
        51,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo d_emlrtRSI =
    {
        55,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo e_emlrtRSI =
    {
        56,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo f_emlrtRSI =
    {
        66,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo g_emlrtRSI =
    {
        67,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo h_emlrtRSI =
    {
        68,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo i_emlrtRSI =
    {
        71,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo j_emlrtRSI =
    {
        78,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo k_emlrtRSI =
    {
        85,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo l_emlrtRSI =
    {
        89,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo m_emlrtRSI =
    {
        94,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo n_emlrtRSI =
    {
        100,                      /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo o_emlrtRSI =
    {
        105,                      /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo p_emlrtRSI =
    {
        119,                      /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo q_emlrtRSI =
    {
        134,                      /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo r_emlrtRSI =
    {
        144,                      /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo s_emlrtRSI =
    {
        148,                      /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo t_emlrtRSI =
    {
        155,                      /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo u_emlrtRSI =
    {
        158,                      /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo v_emlrtRSI =
    {
        168,                      /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo w_emlrtRSI =
    {
        243,              /* lineNo */
        "mf_with_constr", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo x_emlrtRSI =
    {
        249,              /* lineNo */
        "mf_with_constr", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo y_emlrtRSI =
    {
        254,              /* lineNo */
        "mf_with_constr", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo ab_emlrtRSI =
    {
        256,              /* lineNo */
        "mf_with_constr", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo bb_emlrtRSI =
    {
        299,              /* lineNo */
        "mf_with_constr", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo cb_emlrtRSI = {
    44,       /* lineNo */
    "mpower", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/matfun/mpower.m" /* pathName
                                                                       */
};

static emlrtRSInfo eb_emlrtRSI = {
    42,     /* lineNo */
    "sort", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/datafun/sort.m" /* pathName
                                                                      */
};

static emlrtRSInfo jc_emlrtRSI = {
    19,        /* lineNo */
    "setdiff", /* fcnName */
    "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/ops/setdiff.m" /* pathName
                                                                     */
};

static emlrtRSInfo kc_emlrtRSI =
    {
        94,          /* lineNo */
        "eml_setop", /* fcnName */
        "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/ops/private/"
        "eml_setop.m" /* pathName */
};

static emlrtRSInfo fd_emlrtRSI =
    {
        94,                  /* lineNo */
        "eml_mtimes_helper", /* fcnName */
        "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/ops/"
        "eml_mtimes_helper.m" /* pathName */
};

static emlrtRSInfo gd_emlrtRSI =
    {
        69,                  /* lineNo */
        "eml_mtimes_helper", /* fcnName */
        "/usr/local/MATLAB/R2023a/toolbox/eml/lib/matlab/ops/"
        "eml_mtimes_helper.m" /* pathName */
};

static emlrtDCInfo emlrtDCI =
    {
        224,              /* lineNo */
        16,               /* colNo */
        "mf_with_constr", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        1                           /* checkKind */
};

static emlrtBCInfo emlrtBCI =
    {
        1,                /* iFirst */
        190,              /* iLast */
        224,              /* lineNo */
        16,               /* colNo */
        "X",              /* aName */
        "mf_with_constr", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        0                           /* checkKind */
};

static emlrtDCInfo b_emlrtDCI =
    {
        225,              /* lineNo */
        16,               /* colNo */
        "mf_with_constr", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        1                           /* checkKind */
};

static emlrtBCInfo b_emlrtBCI =
    {
        1,                /* iFirst */
        190,              /* iLast */
        225,              /* lineNo */
        16,               /* colNo */
        "X",              /* aName */
        "mf_with_constr", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        0                           /* checkKind */
};

static emlrtBCInfo c_emlrtBCI =
    {
        -1,                /* iFirst */
        -1,                /* iLast */
        265,               /* lineNo */
        31,                /* colNo */
        "old_constraints", /* aName */
        "mf_with_constr",  /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        0                           /* checkKind */
};

static emlrtBCInfo d_emlrtBCI =
    {
        1,                /* iFirst */
        190,              /* iLast */
        260,              /* lineNo */
        12,               /* colNo */
        "X",              /* aName */
        "mf_with_constr", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        0                           /* checkKind */
};

static emlrtDCInfo c_emlrtDCI =
    {
        260,              /* lineNo */
        12,               /* colNo */
        "mf_with_constr", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        1                           /* checkKind */
};

static emlrtBCInfo e_emlrtBCI =
    {
        1,                /* iFirst */
        570,              /* iLast */
        260,              /* lineNo */
        26,               /* colNo */
        "u",              /* aName */
        "mf_with_constr", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        0                           /* checkKind */
};

static emlrtDCInfo d_emlrtDCI =
    {
        260,              /* lineNo */
        26,               /* colNo */
        "mf_with_constr", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        1                           /* checkKind */
};

static emlrtBCInfo f_emlrtBCI =
    {
        1,                /* iFirst */
        570,              /* iLast */
        249,              /* lineNo */
        9,                /* colNo */
        "f",              /* aName */
        "mf_with_constr", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        3                           /* checkKind */
};

static emlrtDCInfo e_emlrtDCI =
    {
        249,              /* lineNo */
        9,                /* colNo */
        "mf_with_constr", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        1                           /* checkKind */
};

static emlrtBCInfo g_emlrtBCI =
    {
        1,                /* iFirst */
        570,              /* iLast */
        297,              /* lineNo */
        9,                /* colNo */
        "f",              /* aName */
        "mf_with_constr", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        3                           /* checkKind */
};

static emlrtDCInfo f_emlrtDCI =
    {
        297,              /* lineNo */
        9,                /* colNo */
        "mf_with_constr", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        1                           /* checkKind */
};

static emlrtECInfo emlrtECI =
    {
        -1,                       /* nDims */
        55,                       /* lineNo */
        9,                        /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo b_emlrtECI =
    {
        -1,                       /* nDims */
        56,                       /* lineNo */
        9,                        /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo c_emlrtECI =
    {
        1,                        /* nDims */
        72,                       /* lineNo */
        18,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo d_emlrtECI =
    {
        2,                        /* nDims */
        72,                       /* lineNo */
        18,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo e_emlrtECI =
    {
        1,                        /* nDims */
        75,                       /* lineNo */
        21,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo f_emlrtECI =
    {
        2,                        /* nDims */
        75,                       /* lineNo */
        21,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo g_emlrtECI =
    {
        1,                        /* nDims */
        84,                       /* lineNo */
        18,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo h_emlrtECI =
    {
        -1,                       /* nDims */
        84,                       /* lineNo */
        5,                        /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo i_emlrtECI =
    {
        1,                        /* nDims */
        99,                       /* lineNo */
        22,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo j_emlrtECI =
    {
        -1,                       /* nDims */
        99,                       /* lineNo */
        9,                        /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo k_emlrtECI =
    {
        1,                        /* nDims */
        118,                      /* lineNo */
        22,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo l_emlrtECI =
    {
        -1,                       /* nDims */
        118,                      /* lineNo */
        9,                        /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo m_emlrtECI =
    {
        1,                        /* nDims */
        143,                      /* lineNo */
        30,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo n_emlrtECI =
    {
        -1,                       /* nDims */
        143,                      /* lineNo */
        17,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo o_emlrtECI =
    {
        1,                        /* nDims */
        154,                      /* lineNo */
        34,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo p_emlrtECI =
    {
        -1,                       /* nDims */
        154,                      /* lineNo */
        21,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo q_emlrtECI =
    {
        1,                        /* nDims */
        167,                      /* lineNo */
        34,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo r_emlrtECI =
    {
        -1,                       /* nDims */
        167,                      /* lineNo */
        21,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtBCInfo h_emlrtBCI =
    {
        1,                        /* iFirst */
        570,                      /* iLast */
        23,                       /* lineNo */
        7,                        /* colNo */
        "f",                      /* aName */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        0                           /* checkKind */
};

static emlrtDCInfo g_emlrtDCI =
    {
        23,                       /* lineNo */
        7,                        /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m", /* pName */
        1                           /* checkKind */
};

static emlrtRTEInfo q_emlrtRTEI =
    {
        31,                       /* lineNo */
        1,                        /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo r_emlrtRTEI =
    {
        51,                       /* lineNo */
        15,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo s_emlrtRTEI =
    {
        55,                       /* lineNo */
        22,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo t_emlrtRTEI =
    {
        179,                      /* lineNo */
        5,                        /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo u_emlrtRTEI =
    {
        78,                       /* lineNo */
        15,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo v_emlrtRTEI =
    {
        24,                       /* lineNo */
        1,                        /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo w_emlrtRTEI =
    {
        66,                       /* lineNo */
        9,                        /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo x_emlrtRTEI =
    {
        67,                       /* lineNo */
        9,                        /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo y_emlrtRTEI =
    {
        68,                       /* lineNo */
        9,                        /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo ab_emlrtRTEI =
    {
        75,                       /* lineNo */
        21,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo bb_emlrtRTEI =
    {
        72,                       /* lineNo */
        17,                       /* colNo */
        "fullBFGStoCompileToMex", /* fName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRSInfo ld_emlrtRSI =
    {
        75,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo md_emlrtRSI =
    {
        72,                       /* lineNo */
        "fullBFGStoCompileToMex", /* fcnName */
        "/home/diego/fem/Matlab_Ill/IndentSphere/fnctns/"
        "fullBFGStoCompileToMex.m" /* pathName */
};

/* Function Declarations */
static int32_T b_binary_expand_op(real_T in1_data[], const real_T in2[570],
                                  const int32_T in3_data[],
                                  const int32_T *in3_size,
                                  const real_T in4_data[],
                                  const int32_T *in4_size);

static int32_T binary_expand_op(real_T in1_data[], const real_T in2[570],
                                const real_T in3[570], const int16_T in4_data[],
                                const int32_T *in4_size,
                                const real_T in5_data[],
                                const int32_T *in5_size);

static void c_binary_expand_op(const emlrtStack *sp, emxArray_real_T *in1,
                               const emxArray_real_T *in2, real_T in3);

static real_T
mf_with_constr(const emlrtStack *sp, const real_T u[570], const real_T X[570],
               const real_T conn[532], const real_T free_ind_data[],
               int32_T free_ind_size, const real_T old_constraints_data[],
               const int32_T old_constraints_size[2], real_T ratio,
               real_T k_pen, real_T Cx, real_T Cy, real_T Cz, real_T R,
               const real_T slaves[152], const real_T dofs[570],
               const real_T s[304], const real_T sh[304], real_T f[570]);

static void minus(const emlrtStack *sp, emxArray_real_T *in1,
                  const emxArray_real_T *in2);

static void plus(const emlrtStack *sp, emxArray_real_T *in1,
                 const emxArray_real_T *in2);

/* Function Definitions */
static int32_T b_binary_expand_op(real_T in1_data[], const real_T in2[570],
                                  const int32_T in3_data[],
                                  const int32_T *in3_size,
                                  const real_T in4_data[],
                                  const int32_T *in4_size)
{
  int32_T i;
  int32_T in1_size;
  int32_T stride_0_0;
  int32_T stride_1_0;
  if (*in4_size == 1) {
    in1_size = *in3_size;
  } else {
    in1_size = *in4_size;
  }
  stride_0_0 = (*in3_size != 1);
  stride_1_0 = (*in4_size != 1);
  for (i = 0; i < in1_size; i++) {
    in1_data[i] = in2[in3_data[i * stride_0_0] - 1] + in4_data[i * stride_1_0];
  }
  return in1_size;
}

static int32_T binary_expand_op(real_T in1_data[], const real_T in2[570],
                                const real_T in3[570], const int16_T in4_data[],
                                const int32_T *in4_size,
                                const real_T in5_data[],
                                const int32_T *in5_size)
{
  int32_T i;
  int32_T in1_size;
  int32_T stride_0_0;
  int32_T stride_1_0;
  if (*in5_size == 1) {
    in1_size = *in4_size;
  } else {
    in1_size = *in5_size;
  }
  stride_0_0 = (*in4_size != 1);
  stride_1_0 = (*in5_size != 1);
  for (i = 0; i < in1_size; i++) {
    in1_data[i] = in2[(int32_T)in3[in4_data[i * stride_0_0]] - 1] +
                  in5_data[i * stride_1_0];
  }
  return in1_size;
}

static void c_binary_expand_op(const emlrtStack *sp, emxArray_real_T *in1,
                               const emxArray_real_T *in2, real_T in3)
{
  emxArray_real_T *b_in1;
  const real_T *in2_data;
  real_T *b_in1_data;
  real_T *in1_data;
  int32_T aux_0_1;
  int32_T aux_1_1;
  int32_T b_loop_ub;
  int32_T i;
  int32_T i1;
  int32_T loop_ub;
  int32_T stride_0_0;
  int32_T stride_0_1;
  int32_T stride_1_0;
  int32_T stride_1_1;
  in2_data = in2->data;
  in1_data = in1->data;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtConstCTX)sp);
  emxInit_real_T(sp, &b_in1, &bb_emlrtRTEI);
  if (in2->size[0] == 1) {
    loop_ub = in1->size[0];
  } else {
    loop_ub = in2->size[0];
  }
  i = b_in1->size[0] * b_in1->size[1];
  b_in1->size[0] = loop_ub;
  if (in2->size[1] == 1) {
    b_loop_ub = in1->size[1];
  } else {
    b_loop_ub = in2->size[1];
  }
  b_in1->size[1] = b_loop_ub;
  emxEnsureCapacity_real_T(sp, b_in1, i, &bb_emlrtRTEI);
  b_in1_data = b_in1->data;
  stride_0_0 = (in1->size[0] != 1);
  stride_0_1 = (in1->size[1] != 1);
  stride_1_0 = (in2->size[0] != 1);
  stride_1_1 = (in2->size[1] != 1);
  aux_0_1 = 0;
  aux_1_1 = 0;
  for (i = 0; i < b_loop_ub; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_in1_data[i1 + b_in1->size[0] * i] =
          (in1_data[i1 * stride_0_0 + in1->size[0] * aux_0_1] +
           in2_data[i1 * stride_1_0 + in2->size[0] * aux_1_1]) /
          in3;
    }
    aux_1_1 += stride_1_1;
    aux_0_1 += stride_0_1;
  }
  i = in1->size[0] * in1->size[1];
  in1->size[0] = b_in1->size[0];
  in1->size[1] = b_in1->size[1];
  emxEnsureCapacity_real_T(sp, in1, i, &bb_emlrtRTEI);
  in1_data = in1->data;
  loop_ub = b_in1->size[1];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = b_in1->size[0];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      in1_data[i1 + in1->size[0] * i] = b_in1_data[i1 + b_in1->size[0] * i];
    }
  }
  emxFree_real_T(sp, &b_in1);
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtConstCTX)sp);
}

static real_T
mf_with_constr(const emlrtStack *sp, const real_T u[570], const real_T X[570],
               const real_T conn[532], const real_T free_ind_data[],
               int32_T free_ind_size, const real_T old_constraints_data[],
               const int32_T old_constraints_size[2], real_T ratio,
               real_T k_pen, real_T Cx, real_T Cy, real_T Cz, real_T R,
               const real_T slaves[152], const real_T dofs[570],
               const real_T s[304], const real_T sh[304], real_T f[570])
{
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack st;
  real_T c_data[570];
  real_T free_ind_sorted_data[570];
  real_T xs[456];
  real_T x1[3];
  real_T absxk;
  real_T b_scale;
  real_T d;
  real_T dofi_tmp_tmp;
  real_T k;
  real_T m;
  real_T phi;
  real_T scale;
  real_T t;
  real_T x2_idx_1;
  real_T x2_idx_2;
  int32_T ia_data[570];
  int32_T c_size[2];
  int32_T b_i;
  int32_T i;
  int32_T i_cnstr;
  int32_T ib_size;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  /* UNTITLED Summary of this function goes here */
  /*    Detailed explanation goes here */
  /*  m=mp('0');  % it will adapt to datatype in the 'for' loop */
  m = 0.0;
  /*  it will adapt to datatype in the 'for' loop */
  /*  f=zeros(len_u,1,'like',u);   */
  /*  f=zeros(len_u,1,'mp'); */
  memset(&f[0], 0, 570U * sizeof(real_T));
  for (i = 0; i < 266; i++) {
    real_T b_f[6];
    real_T dofi[6];
    real_T Cxy_idx_0;
    real_T Cxy_idx_1;
    real_T L;
    real_T L0;
    real_T b_dofi_tmp;
    real_T d1;
    real_T dofi_tmp;
    real_T x2_idx_0;
    d = conn[i];
    dofi_tmp = d * 3.0 - 2.0;
    dofi[0] = dofi_tmp;
    dofi[1] = d * 3.0 - 1.0;
    dofi[2] = d * 3.0;
    dofi_tmp_tmp = conn[i + 266];
    b_dofi_tmp = dofi_tmp_tmp * 3.0;
    dofi[3] = b_dofi_tmp - 2.0;
    dofi[4] = b_dofi_tmp - 1.0;
    dofi[5] = b_dofi_tmp;
    if (d != (int32_T)muDoubleScalarFloor(d)) {
      emlrtIntegerCheckR2012b(d, &emlrtDCI, (emlrtConstCTX)sp);
    }
    if (((int32_T)d < 1) || ((int32_T)d > 190)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)d, 1, 190, &emlrtBCI,
                                    (emlrtConstCTX)sp);
    }
    k = X[(int32_T)d - 1];
    x1[0] = k + u[(int32_T)dofi_tmp - 1];
    phi = X[(int32_T)d + 189];
    x1[1] = phi + u[(int32_T)(dofi_tmp + 1.0) - 1];
    x2_idx_2 = X[(int32_T)d + 379];
    x1[2] = x2_idx_2 + u[(int32_T)(dofi_tmp + 2.0) - 1];
    if (dofi_tmp_tmp != (int32_T)muDoubleScalarFloor(dofi_tmp_tmp)) {
      emlrtIntegerCheckR2012b(dofi_tmp_tmp, &b_emlrtDCI, (emlrtConstCTX)sp);
    }
    if (((int32_T)dofi_tmp_tmp < 1) || ((int32_T)dofi_tmp_tmp > 190)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)dofi_tmp_tmp, 1, 190, &b_emlrtBCI,
                                    (emlrtConstCTX)sp);
    }
    scale = 3.3121686421112381E-170;
    b_scale = 3.3121686421112381E-170;
    d = X[(int32_T)dofi_tmp_tmp - 1];
    d1 = d + u[(int32_T)(b_dofi_tmp - 2.0) - 1];
    x2_idx_0 = d1;
    absxk = muDoubleScalarAbs(k - d);
    if (absxk > 3.3121686421112381E-170) {
      L0 = 1.0;
      scale = absxk;
    } else {
      t = absxk / 3.3121686421112381E-170;
      L0 = t * t;
    }
    d = x1[0] - d1;
    Cxy_idx_0 = d;
    absxk = muDoubleScalarAbs(d);
    if (absxk > 3.3121686421112381E-170) {
      L = 1.0;
      b_scale = absxk;
    } else {
      t = absxk / 3.3121686421112381E-170;
      L = t * t;
    }
    d = X[(int32_T)dofi_tmp_tmp + 189];
    d1 = d + u[(int32_T)((b_dofi_tmp - 2.0) + 1.0) - 1];
    x2_idx_1 = d1;
    absxk = muDoubleScalarAbs(phi - d);
    if (absxk > scale) {
      t = scale / absxk;
      L0 = L0 * t * t + 1.0;
      scale = absxk;
    } else {
      t = absxk / scale;
      L0 += t * t;
    }
    d = x1[1] - d1;
    Cxy_idx_1 = d;
    absxk = muDoubleScalarAbs(d);
    if (absxk > b_scale) {
      t = b_scale / absxk;
      L = L * t * t + 1.0;
      b_scale = absxk;
    } else {
      t = absxk / b_scale;
      L += t * t;
    }
    d = X[(int32_T)dofi_tmp_tmp + 379];
    d1 = d + u[(int32_T)((b_dofi_tmp - 2.0) + 2.0) - 1];
    absxk = muDoubleScalarAbs(x2_idx_2 - d);
    if (absxk > scale) {
      t = scale / absxk;
      L0 = L0 * t * t + 1.0;
      scale = absxk;
    } else {
      t = absxk / scale;
      L0 += t * t;
    }
    d = x1[2] - d1;
    absxk = muDoubleScalarAbs(d);
    if (absxk > b_scale) {
      t = b_scale / absxk;
      L = L * t * t + 1.0;
      b_scale = absxk;
    } else {
      t = absxk / b_scale;
      L += t * t;
    }
    L0 = scale * muDoubleScalarSqrt(L0);
    L = b_scale * muDoubleScalarSqrt(L);
    if (L > L0) {
      k = 1.0;
    } else {
      k = ratio;
    }
    st.site = &w_emlrtRSI;
    b_st.site = &w_emlrtRSI;
    phi = muDoubleScalarLog(L / L0);
    b_st.site = &cb_emlrtRSI;
    m += k * L0 * (0.5 * (phi * phi));
    st.site = &x_emlrtRSI;
    absxk = k * (L0 / L) * phi;
    b_f[0] = f[(int16_T)dofi_tmp - 1] + absxk * (Cxy_idx_0 / L);
    b_f[3] =
        f[(int16_T)(b_dofi_tmp - 2.0) - 1] + absxk * ((x2_idx_0 - x1[0]) / L);
    b_f[1] = f[(int16_T)dofi[1] - 1] + absxk * (Cxy_idx_1 / L);
    b_f[4] =
        f[(int16_T)(b_dofi_tmp - 1.0) - 1] + absxk * ((x2_idx_1 - x1[1]) / L);
    b_f[2] = f[(int16_T)dofi[2] - 1] + absxk * (d / L);
    b_f[5] = f[(int16_T)b_dofi_tmp - 1] + absxk * ((d1 - x1[2]) / L);
    for (b_i = 0; b_i < 6; b_i++) {
      d = dofi[b_i];
      if (d != (int32_T)muDoubleScalarFloor(d)) {
        emlrtIntegerCheckR2012b(d, &e_emlrtDCI, (emlrtConstCTX)sp);
      }
      if (((int32_T)d < 1) || ((int32_T)d > 570)) {
        emlrtDynamicBoundsCheckR2012b((int32_T)d, 1, 570, &f_emlrtBCI,
                                      (emlrtConstCTX)sp);
      }
      f[(int32_T)d - 1] = b_f[b_i];
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtConstCTX)sp);
    }
  }
  /*  Sort 'free_ind' in ascending order */
  st.site = &y_emlrtRSI;
  if (free_ind_size - 1 >= 0) {
    memcpy(&free_ind_sorted_data[0], &free_ind_data[0],
           (uint32_T)free_ind_size * sizeof(real_T));
  }
  b_st.site = &eb_emlrtRSI;
  sort(&b_st, free_ind_sorted_data, &free_ind_size);
  /*  Use the sorted array with 'setdiff' */
  st.site = &ab_emlrtRSI;
  b_st.site = &jc_emlrtRSI;
  c_st.site = &kc_emlrtRSI;
  do_vectors(&c_st, free_ind_sorted_data, free_ind_size, c_data, c_size,
             ia_data, &ib_size);
  i = c_size[1];
  for (b_i = 0; b_i < i; b_i++) {
    f[(int32_T)c_data[b_i] - 1] = 0.0;
  }
  /*  f(setdiff(1:numel(f),free_ind))=0; */
  for (b_i = 0; b_i < 3; b_i++) {
    for (ib_size = 0; ib_size < 152; ib_size++) {
      d = slaves[ib_size];
      if (d != (int32_T)muDoubleScalarFloor(d)) {
        emlrtIntegerCheckR2012b(d, &c_emlrtDCI, (emlrtConstCTX)sp);
      }
      if (((int32_T)d < 1) || ((int32_T)d > 190)) {
        emlrtDynamicBoundsCheckR2012b((int32_T)d, 1, 190, &d_emlrtBCI,
                                      (emlrtConstCTX)sp);
      }
      i = ((int32_T)d + 190 * b_i) - 1;
      d = dofs[i];
      if (d != (int32_T)muDoubleScalarFloor(d)) {
        emlrtIntegerCheckR2012b(d, &d_emlrtDCI, (emlrtConstCTX)sp);
      }
      if (((int32_T)d < 1) || ((int32_T)d > 570)) {
        emlrtDynamicBoundsCheckR2012b((int32_T)d, 1, 570, &e_emlrtBCI,
                                      (emlrtConstCTX)sp);
      }
      xs[ib_size + 152 * b_i] = X[i] + u[(int32_T)d - 1];
    }
  }
  b_i = old_constraints_size[1];
  for (i_cnstr = 0; i_cnstr < b_i; i_cnstr++) {
    boolean_T x[2];
    boolean_T exitg1;
    boolean_T y;
    if (i_cnstr + 1 > old_constraints_size[1]) {
      emlrtDynamicBoundsCheckR2012b(i_cnstr + 1, 1, old_constraints_size[1],
                                    &c_emlrtBCI, (emlrtConstCTX)sp);
    }
    /*  % Fless case */
    /*  g =norm(xsi - Cxy) - R; */
    /*  dgdu = (xsi-Cxy)/norm(xsi-Cxy); */
    /*   */
    i = (int32_T)old_constraints_data[i_cnstr];
    x[0] = (sh[i - 1] == -1.0);
    x[1] = (sh[i + 151] == -1.0);
    y = true;
    ib_size = 0;
    exitg1 = false;
    while ((!exitg1) && (ib_size < 2)) {
      if (!x[ib_size]) {
        y = false;
        exitg1 = true;
      } else {
        ib_size++;
      }
    }
    if (y) {
      __m128d r;
      scale = 3.3121686421112381E-170;
      d = xs[i - 1] - Cx;
      x1[0] = d;
      absxk = muDoubleScalarAbs(d);
      if (absxk > 3.3121686421112381E-170) {
        k = 1.0;
        scale = absxk;
      } else {
        t = absxk / 3.3121686421112381E-170;
        k = t * t;
      }
      d = xs[i + 151] - Cy;
      x1[1] = d;
      absxk = muDoubleScalarAbs(d);
      if (absxk > scale) {
        t = scale / absxk;
        k = k * t * t + 1.0;
        scale = absxk;
      } else {
        t = absxk / scale;
        k += t * t;
      }
      d = xs[i + 303] - Cz;
      x1[2] = d;
      absxk = muDoubleScalarAbs(d);
      if (absxk > scale) {
        t = scale / absxk;
        k = k * t * t + 1.0;
        scale = absxk;
      } else {
        t = absxk / scale;
        k += t * t;
      }
      k = scale * muDoubleScalarSqrt(k);
      b_scale = k - R;
      r = _mm_loadu_pd(&x1[0]);
      _mm_storeu_pd(&x1[0], _mm_div_pd(r, _mm_set1_pd(k)));
      x1[2] /= k;
    } else {
      __m128d r;
      /*  theta = pi*(1-s(cnt_act,2)); */
      /*  phi = pi*s(cnt_act,1); */
      k = 3.1415926535897931 * (1.0 - s[i + 151]);
      phi = 3.1415926535897931 * s[i - 1];
      dofi_tmp_tmp = muDoubleScalarSin(k);
      x1[0] = Cx + -(muDoubleScalarCos(phi) * dofi_tmp_tmp) * R;
      x1[1] = Cy + -muDoubleScalarCos(k) * R;
      x1[2] = Cz + -(dofi_tmp_tmp * muDoubleScalarSin(phi)) * R;
      scale = 3.3121686421112381E-170;
      d = xs[i - 1] - x1[0];
      x1[0] = d;
      absxk = muDoubleScalarAbs(d);
      if (absxk > 3.3121686421112381E-170) {
        b_scale = 1.0;
        scale = absxk;
      } else {
        t = absxk / 3.3121686421112381E-170;
        b_scale = t * t;
      }
      d = xs[i + 151] - x1[1];
      x1[1] = d;
      absxk = muDoubleScalarAbs(d);
      if (absxk > scale) {
        t = scale / absxk;
        b_scale = b_scale * t * t + 1.0;
        scale = absxk;
      } else {
        t = absxk / scale;
        b_scale += t * t;
      }
      d = xs[i + 303] - x1[2];
      x1[2] = d;
      absxk = muDoubleScalarAbs(d);
      if (absxk > scale) {
        t = scale / absxk;
        b_scale = b_scale * t * t + 1.0;
        scale = absxk;
      } else {
        t = absxk / scale;
        b_scale += t * t;
      }
      b_scale = scale * muDoubleScalarSqrt(b_scale);
      r = _mm_loadu_pd(&x1[0]);
      _mm_storeu_pd(&x1[0], _mm_div_pd(r, _mm_set1_pd(b_scale)));
      x1[2] /= b_scale;
    }
    /*  f(a)=f(a)+[dEdu1;dEdv1;dEdw1]; */
    absxk = k_pen * b_scale;
    i = (int32_T)slaves[i - 1];
    k = dofs[i - 1];
    phi = dofs[i + 189];
    x2_idx_1 = f[(int32_T)phi - 1] + absxk * x1[1];
    dofi_tmp_tmp = dofs[i + 379];
    x2_idx_2 = f[(int32_T)dofi_tmp_tmp - 1] + absxk * x1[2];
    if (k != (int32_T)muDoubleScalarFloor(k)) {
      emlrtIntegerCheckR2012b(k, &f_emlrtDCI, (emlrtConstCTX)sp);
    }
    if (((int32_T)k < 1) || ((int32_T)k > 570)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)k, 1, 570, &g_emlrtBCI,
                                    (emlrtConstCTX)sp);
    }
    f[(int32_T)k - 1] += absxk * x1[0];
    if (phi != (int32_T)muDoubleScalarFloor(phi)) {
      emlrtIntegerCheckR2012b(phi, &f_emlrtDCI, (emlrtConstCTX)sp);
    }
    if (((int32_T)phi < 1) || ((int32_T)phi > 570)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)phi, 1, 570, &g_emlrtBCI,
                                    (emlrtConstCTX)sp);
    }
    f[(int32_T)phi - 1] = x2_idx_1;
    if (dofi_tmp_tmp != (int32_T)muDoubleScalarFloor(dofi_tmp_tmp)) {
      emlrtIntegerCheckR2012b(dofi_tmp_tmp, &f_emlrtDCI, (emlrtConstCTX)sp);
    }
    if (((int32_T)dofi_tmp_tmp < 1) || ((int32_T)dofi_tmp_tmp > 570)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)dofi_tmp_tmp, 1, 570, &g_emlrtBCI,
                                    (emlrtConstCTX)sp);
    }
    f[(int32_T)dofi_tmp_tmp - 1] = x2_idx_2;
    st.site = &bb_emlrtRSI;
    b_st.site = &cb_emlrtRSI;
    m += 0.5 * k_pen * (b_scale * b_scale);
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtConstCTX)sp);
    }
  }
  return m;
}

static void minus(const emlrtStack *sp, emxArray_real_T *in1,
                  const emxArray_real_T *in2)
{
  emxArray_real_T *b_in1;
  const real_T *in2_data;
  real_T *b_in1_data;
  real_T *in1_data;
  int32_T aux_0_1;
  int32_T aux_1_1;
  int32_T b_loop_ub;
  int32_T i;
  int32_T i1;
  int32_T loop_ub;
  int32_T stride_0_0;
  int32_T stride_0_1;
  int32_T stride_1_0;
  int32_T stride_1_1;
  in2_data = in2->data;
  in1_data = in1->data;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtConstCTX)sp);
  emxInit_real_T(sp, &b_in1, &ab_emlrtRTEI);
  if (in2->size[0] == 1) {
    loop_ub = in1->size[0];
  } else {
    loop_ub = in2->size[0];
  }
  i = b_in1->size[0] * b_in1->size[1];
  b_in1->size[0] = loop_ub;
  if (in2->size[1] == 1) {
    b_loop_ub = in1->size[1];
  } else {
    b_loop_ub = in2->size[1];
  }
  b_in1->size[1] = b_loop_ub;
  emxEnsureCapacity_real_T(sp, b_in1, i, &ab_emlrtRTEI);
  b_in1_data = b_in1->data;
  stride_0_0 = (in1->size[0] != 1);
  stride_0_1 = (in1->size[1] != 1);
  stride_1_0 = (in2->size[0] != 1);
  stride_1_1 = (in2->size[1] != 1);
  aux_0_1 = 0;
  aux_1_1 = 0;
  for (i = 0; i < b_loop_ub; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_in1_data[i1 + b_in1->size[0] * i] =
          in1_data[i1 * stride_0_0 + in1->size[0] * aux_0_1] -
          in2_data[i1 * stride_1_0 + in2->size[0] * aux_1_1];
    }
    aux_1_1 += stride_1_1;
    aux_0_1 += stride_0_1;
  }
  i = in1->size[0] * in1->size[1];
  in1->size[0] = b_in1->size[0];
  in1->size[1] = b_in1->size[1];
  emxEnsureCapacity_real_T(sp, in1, i, &ab_emlrtRTEI);
  in1_data = in1->data;
  loop_ub = b_in1->size[1];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = b_in1->size[0];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      in1_data[i1 + in1->size[0] * i] = b_in1_data[i1 + b_in1->size[0] * i];
    }
  }
  emxFree_real_T(sp, &b_in1);
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtConstCTX)sp);
}

static void plus(const emlrtStack *sp, emxArray_real_T *in1,
                 const emxArray_real_T *in2)
{
  emxArray_real_T *b_in1;
  const real_T *in2_data;
  real_T *b_in1_data;
  real_T *in1_data;
  int32_T aux_0_1;
  int32_T aux_1_1;
  int32_T b_loop_ub;
  int32_T i;
  int32_T i1;
  int32_T loop_ub;
  int32_T stride_0_0;
  int32_T stride_0_1;
  int32_T stride_1_0;
  int32_T stride_1_1;
  in2_data = in2->data;
  in1_data = in1->data;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtConstCTX)sp);
  emxInit_real_T(sp, &b_in1, &ab_emlrtRTEI);
  if (in2->size[0] == 1) {
    loop_ub = in1->size[0];
  } else {
    loop_ub = in2->size[0];
  }
  i = b_in1->size[0] * b_in1->size[1];
  b_in1->size[0] = loop_ub;
  if (in2->size[1] == 1) {
    b_loop_ub = in1->size[1];
  } else {
    b_loop_ub = in2->size[1];
  }
  b_in1->size[1] = b_loop_ub;
  emxEnsureCapacity_real_T(sp, b_in1, i, &ab_emlrtRTEI);
  b_in1_data = b_in1->data;
  stride_0_0 = (in1->size[0] != 1);
  stride_0_1 = (in1->size[1] != 1);
  stride_1_0 = (in2->size[0] != 1);
  stride_1_1 = (in2->size[1] != 1);
  aux_0_1 = 0;
  aux_1_1 = 0;
  for (i = 0; i < b_loop_ub; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_in1_data[i1 + b_in1->size[0] * i] =
          in1_data[i1 * stride_0_0 + in1->size[0] * aux_0_1] +
          in2_data[i1 * stride_1_0 + in2->size[0] * aux_1_1];
    }
    aux_1_1 += stride_1_1;
    aux_0_1 += stride_0_1;
  }
  i = in1->size[0] * in1->size[1];
  in1->size[0] = b_in1->size[0];
  in1->size[1] = b_in1->size[1];
  emxEnsureCapacity_real_T(sp, in1, i, &ab_emlrtRTEI);
  in1_data = in1->data;
  loop_ub = b_in1->size[1];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = b_in1->size[0];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      in1_data[i1 + in1->size[0] * i] = b_in1_data[i1 + b_in1->size[0] * i];
    }
  }
  emxFree_real_T(sp, &b_in1);
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtConstCTX)sp);
}

void fullBFGStoCompileToMex(const emlrtStack *sp, real_T u[570],
                            const real_T X[570], const real_T conn[532],
                            const boolean_T is_free[570],
                            const boolean_T actives[152], real_T mu,
                            real_T ratio, real_T k_pen, real_T Cx, real_T Cy,
                            real_T Cz, real_T R, const real_T slaves[152],
                            const real_T dofs[570], const real_T s[304],
                            const real_T sh[304], real_T *m_new, real_T *iter)
{
  emlrtStack b_st;
  emlrtStack st;
  emxArray_real_T b_f_old_data;
  emxArray_real_T c_f_old_data;
  emxArray_real_T d_f_old_data;
  emxArray_real_T e_f_old_data;
  emxArray_real_T f_f_old_data;
  emxArray_real_T g_f_old_data;
  emxArray_real_T h_f_old_data;
  emxArray_real_T i_f_old_data;
  emxArray_real_T j_f_old_data;
  emxArray_real_T k_f_old_data;
  emxArray_real_T l_f_old_data;
  emxArray_real_T m_f_old_data;
  emxArray_real_T *K_new_inv;
  emxArray_real_T *delta_u;
  emxArray_real_T *product1;
  emxArray_real_T *product2;
  emxArray_real_T *product3;
  emxArray_real_T *y;
  real_T d_tmp_data[570];
  real_T e_tmp_data[570];
  real_T f[570];
  real_T f_2_data[570];
  real_T f_3_data[570];
  real_T f_new_data[570];
  real_T f_old_data[570];
  real_T free_ind_data[570];
  real_T ux[570];
  real_T old_constraints_data[152];
  real_T m_2;
  real_T scalar2;
  real_T *K_new_inv_data;
  real_T *delta_u_data;
  real_T *product1_data;
  real_T *product3_data;
  int32_T tmp_data[152];
  int32_T f_old[2];
  int32_T old_constraints_size[2];
  int32_T tmp_size[2];
  int32_T y_size[2];
  int32_T b_i;
  int32_T f_old_size;
  int32_T i;
  int32_T i1;
  int32_T partialTrueCount;
  int32_T signal1;
  int32_T trueCount;
  int16_T b_tmp_data[570];
  boolean_T b_actives[152];
  (void)mu;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtConstCTX)sp);
  /* Here is a fully expanded version of the BFGS algorithm that solves the */
  /* mechanical subproblem with a specific list of constrained slave nodes */
  /*  old_constraints = slaves(actives==true); */
  memcpy(&b_actives[0], &actives[0], 152U * sizeof(boolean_T));
  eml_find(b_actives, tmp_data, tmp_size);
  old_constraints_size[0] = 1;
  old_constraints_size[1] = tmp_size[1];
  signal1 = tmp_size[1];
  for (i = 0; i < signal1; i++) {
    old_constraints_data[i] = tmp_data[i];
  }
  trueCount = 0;
  partialTrueCount = 0;
  for (b_i = 0; b_i < 570; b_i++) {
    if (is_free[b_i]) {
      trueCount++;
      b_tmp_data[partialTrueCount] = (int16_T)b_i;
      partialTrueCount++;
    }
  }
  for (i = 0; i < trueCount; i++) {
    free_ind_data[i] = dofs[b_tmp_data[i]];
  }
  st.site = &emlrtRSI;
  *m_new = mf_with_constr(&st, u, X, conn, free_ind_data, trueCount,
                          old_constraints_data, old_constraints_size, ratio,
                          k_pen, Cx, Cy, Cz, R, slaves, dofs, s, sh, f);
  for (i = 0; i < trueCount; i++) {
    scalar2 = dofs[b_tmp_data[i]];
    if (scalar2 != (int32_T)muDoubleScalarFloor(scalar2)) {
      emlrtIntegerCheckR2012b(scalar2, &g_emlrtDCI, (emlrtConstCTX)sp);
    }
    if (((int32_T)scalar2 < 1) || ((int32_T)scalar2 > 570)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)scalar2, 1, 570, &h_emlrtBCI,
                                    (emlrtConstCTX)sp);
    }
    f_2_data[i] = f[(int32_T)scalar2 - 1];
  }
  emxInit_real_T(sp, &product3, &y_emlrtRTEI);
  st.site = &b_emlrtRSI;
  eye(&st, trueCount, product3);
  emxInit_real_T(sp, &K_new_inv, &v_emlrtRTEI);
  st.site = &b_emlrtRSI;
  inv(&st, product3, K_new_inv);
  K_new_inv_data = K_new_inv->data;
  if (trueCount - 1 >= 0) {
    memset(&f_new_data[0], 0, (uint32_T)trueCount * sizeof(real_T));
  }
  *iter = 0.0;
  emxInit_real_T(sp, &delta_u, &q_emlrtRTEI);
  i = delta_u->size[0] * delta_u->size[1];
  delta_u->size[0] = trueCount;
  delta_u->size[1] = trueCount;
  emxEnsureCapacity_real_T(sp, delta_u, i, &q_emlrtRTEI);
  delta_u_data = delta_u->data;
  signal1 = trueCount * trueCount;
  for (i = 0; i < signal1; i++) {
    delta_u_data[i] = 0.0;
  }
  m_2 = 0.0;
  emxInit_real_T(sp, &product1, &w_emlrtRTEI);
  emxInit_real_T(sp, &product2, &x_emlrtRTEI);
  emxInit_real_T(sp, &y, &s_emlrtRTEI);
  int32_T exitg1;
  do {
    exitg1 = 0;
    partialTrueCount = (trueCount != 1);
    b_i = (trueCount != 1);
    for (i = 0; i < trueCount; i++) {
      f_old_data[i] = f_2_data[i * partialTrueCount] - f_new_data[i * b_i];
    }
    if ((c_norm(f_old_data, trueCount) > 0.0) &&
        (c_norm(f_2_data, trueCount) > 0.0)) {
      __m128d r;
      __m128d r1;
      real_T alpha3;
      real_T m_3;
      int32_T c_tmp_data[570];
      int32_T signal2;
      int32_T tmp_size_idx_0;
      (*iter)++;
      f_old_size = trueCount;
      if (trueCount - 1 >= 0) {
        memcpy(&f_old_data[0], &f_new_data[0],
               (uint32_T)trueCount * sizeof(real_T));
      }
      if (trueCount - 1 >= 0) {
        memcpy(&f_new_data[0], &f_2_data[0],
               (uint32_T)trueCount * sizeof(real_T));
      }
      partialTrueCount = (trueCount / 2) << 1;
      b_i = partialTrueCount - 2;
      for (i = 0; i <= b_i; i += 2) {
        r = _mm_loadu_pd(&f_2_data[i]);
        r1 = _mm_loadu_pd(&f_old_data[i]);
        _mm_storeu_pd(&f_old_data[i], _mm_sub_pd(r, r1));
      }
      for (i = partialTrueCount; i < trueCount; i++) {
        f_old_data[i] = f_2_data[i] - f_old_data[i];
      }
      if (*iter == 1.0) {
        st.site = &c_emlrtRSI;
        i = product3->size[0] * product3->size[1];
        product3->size[0] = K_new_inv->size[0];
        product3->size[1] = K_new_inv->size[1];
        emxEnsureCapacity_real_T(&st, product3, i, &r_emlrtRTEI);
        product3_data = product3->data;
        signal1 = K_new_inv->size[0] * K_new_inv->size[1];
        partialTrueCount = (signal1 / 2) << 1;
        b_i = partialTrueCount - 2;
        for (i = 0; i <= b_i; i += 2) {
          r = _mm_loadu_pd(&K_new_inv_data[i]);
          _mm_storeu_pd(&product3_data[i], _mm_mul_pd(r, _mm_set1_pd(-1.0)));
        }
        for (i = partialTrueCount; i < signal1; i++) {
          product3_data[i] = -K_new_inv_data[i];
        }
        b_st.site = &gd_emlrtRSI;
        dynamic_size_checks(&b_st, product3, trueCount, product3->size[1],
                            trueCount);
        f_old_size = mtimes(product3, f_2_data, trueCount, f_old_data);
      } else {
        /*  K_new_inv=K_old_inv+(delta_u'*delta_f+delta_f'*K_old_inv*delta_f)*(delta_u*delta_u')/(delta_u'*delta_f)^2-(K_old_inv*delta_f*delta_u'+delta_u*delta_f'*K_old_inv)/(delta_u'*delta_f);
         */
        st.site = &d_emlrtRSI;
        f_old[0] = f_old_size;
        f_old[1] = 1;
        b_f_old_data.data = &f_old_data[0];
        b_f_old_data.size = &f_old[0];
        b_f_old_data.allocatedSize = 570;
        b_f_old_data.numDimensions = 2;
        b_f_old_data.canFreeData = false;
        b_st.site = &gd_emlrtRSI;
        b_dynamic_size_checks(&b_st, delta_u, &b_f_old_data, delta_u->size[0],
                              f_old_size);
        f_old[0] = f_old_size;
        f_old[1] = 1;
        c_f_old_data.data = &f_old_data[0];
        c_f_old_data.size = &f_old[0];
        c_f_old_data.allocatedSize = 570;
        c_f_old_data.numDimensions = 2;
        c_f_old_data.canFreeData = false;
        b_st.site = &fd_emlrtRSI;
        b_mtimes(&b_st, delta_u, &c_f_old_data, y);
        delta_u_data = y->data;
        st.site = &d_emlrtRSI;
        f_old[0] = f_old_size;
        f_old[1] = 1;
        d_f_old_data.data = &f_old_data[0];
        d_f_old_data.size = &f_old[0];
        d_f_old_data.allocatedSize = 570;
        d_f_old_data.numDimensions = 2;
        d_f_old_data.canFreeData = false;
        b_st.site = &gd_emlrtRSI;
        b_dynamic_size_checks(&b_st, &d_f_old_data, K_new_inv, f_old_size,
                              K_new_inv->size[0]);
        f_old[0] = f_old_size;
        f_old[1] = 1;
        e_f_old_data.data = &f_old_data[0];
        e_f_old_data.size = &f_old[0];
        e_f_old_data.allocatedSize = 570;
        e_f_old_data.numDimensions = 2;
        e_f_old_data.canFreeData = false;
        b_st.site = &fd_emlrtRSI;
        b_mtimes(&b_st, &e_f_old_data, K_new_inv, product3);
        st.site = &d_emlrtRSI;
        f_old[0] = f_old_size;
        f_old[1] = 1;
        f_f_old_data.data = &f_old_data[0];
        f_f_old_data.size = &f_old[0];
        f_f_old_data.allocatedSize = 570;
        f_f_old_data.numDimensions = 2;
        f_f_old_data.canFreeData = false;
        b_st.site = &gd_emlrtRSI;
        b_dynamic_size_checks(&b_st, product3, &f_f_old_data, product3->size[1],
                              f_old_size);
        f_old[0] = f_old_size;
        f_old[1] = 1;
        g_f_old_data.data = &f_old_data[0];
        g_f_old_data.size = &f_old[0];
        g_f_old_data.allocatedSize = 570;
        g_f_old_data.numDimensions = 2;
        g_f_old_data.canFreeData = false;
        b_st.site = &fd_emlrtRSI;
        c_mtimes(&b_st, product3, &g_f_old_data, product1);
        product1_data = product1->data;
        tmp_size_idx_0 = y->size[0];
        signal1 = y->size[1];
        for (i = 0; i < signal1; i++) {
          signal2 = y->size[0];
          partialTrueCount = (y->size[0] / 2) << 1;
          b_i = partialTrueCount - 2;
          for (i1 = 0; i1 <= b_i; i1 += 2) {
            r = _mm_loadu_pd(&delta_u_data[i1 + y->size[0] * i]);
            _mm_storeu_pd(
                &f[i1 + tmp_size_idx_0 * i],
                _mm_add_pd(r,
                           _mm_set1_pd(product1_data[product1->size[0] * i])));
          }
          for (i1 = partialTrueCount; i1 < signal2; i1++) {
            f[i1 + tmp_size_idx_0 * i] = delta_u_data[i1 + y->size[0] * i] +
                                         product1_data[product1->size[0] * i];
          }
        }
        if (tmp_size_idx_0 != 1) {
          emlrtSubAssignSizeCheck1dR2017a(1, tmp_size_idx_0, &emlrtECI,
                                          (emlrtConstCTX)sp);
        }
        st.site = &e_emlrtRSI;
        f_old[0] = f_old_size;
        f_old[1] = 1;
        h_f_old_data.data = &f_old_data[0];
        h_f_old_data.size = &f_old[0];
        h_f_old_data.allocatedSize = 570;
        h_f_old_data.numDimensions = 2;
        h_f_old_data.canFreeData = false;
        b_st.site = &gd_emlrtRSI;
        b_dynamic_size_checks(&b_st, delta_u, &h_f_old_data, delta_u->size[0],
                              f_old_size);
        f_old[0] = f_old_size;
        f_old[1] = 1;
        i_f_old_data.data = &f_old_data[0];
        i_f_old_data.size = &f_old[0];
        i_f_old_data.allocatedSize = 570;
        i_f_old_data.numDimensions = 2;
        i_f_old_data.canFreeData = false;
        b_st.site = &fd_emlrtRSI;
        b_mtimes(&b_st, delta_u, &i_f_old_data, y);
        delta_u_data = y->data;
        if (y->size[0] != 1) {
          emlrtSubAssignSizeCheck1dR2017a(1, y->size[0], &b_emlrtECI,
                                          (emlrtConstCTX)sp);
        }
        scalar2 = delta_u_data[0];
        /*  scalar1=double(scalar1); */
        /*  scalar2=double(scalar2); */
        /*  K_new_inv=K_old_inv+scalar1*(delta_u*delta_u')/scalar2^2-(K_old_inv*delta_f*delta_u'+delta_u*delta_f'*K_old_inv)/scalar2;
         */
        /*  Step 1: Calculate matrix products separately */
        st.site = &f_emlrtRSI;
        b_st.site = &gd_emlrtRSI;
        b_dynamic_size_checks(&b_st, delta_u, delta_u, delta_u->size[1],
                              delta_u->size[1]);
        b_st.site = &fd_emlrtRSI;
        d_mtimes(&b_st, delta_u, delta_u, product1);
        product1_data = product1->data;
        st.site = &g_emlrtRSI;
        f_old[0] = f_old_size;
        f_old[1] = 1;
        j_f_old_data.data = &f_old_data[0];
        j_f_old_data.size = &f_old[0];
        j_f_old_data.allocatedSize = 570;
        j_f_old_data.numDimensions = 2;
        j_f_old_data.canFreeData = false;
        b_st.site = &gd_emlrtRSI;
        b_dynamic_size_checks(&b_st, K_new_inv, &j_f_old_data,
                              K_new_inv->size[1], f_old_size);
        f_old[0] = f_old_size;
        f_old[1] = 1;
        k_f_old_data.data = &f_old_data[0];
        k_f_old_data.size = &f_old[0];
        k_f_old_data.allocatedSize = 570;
        k_f_old_data.numDimensions = 2;
        k_f_old_data.canFreeData = false;
        b_st.site = &fd_emlrtRSI;
        c_mtimes(&b_st, K_new_inv, &k_f_old_data, y);
        st.site = &g_emlrtRSI;
        b_st.site = &gd_emlrtRSI;
        b_dynamic_size_checks(&b_st, y, delta_u, y->size[1], delta_u->size[1]);
        b_st.site = &fd_emlrtRSI;
        d_mtimes(&b_st, y, delta_u, product2);
        delta_u_data = product2->data;
        st.site = &h_emlrtRSI;
        f_old[0] = f_old_size;
        f_old[1] = 1;
        l_f_old_data.data = &f_old_data[0];
        l_f_old_data.size = &f_old[0];
        l_f_old_data.allocatedSize = 570;
        l_f_old_data.numDimensions = 2;
        l_f_old_data.canFreeData = false;
        b_st.site = &gd_emlrtRSI;
        b_dynamic_size_checks(&b_st, delta_u, &l_f_old_data, delta_u->size[1],
                              1);
        f_old[0] = f_old_size;
        f_old[1] = 1;
        m_f_old_data.data = &f_old_data[0];
        m_f_old_data.size = &f_old[0];
        m_f_old_data.allocatedSize = 570;
        m_f_old_data.numDimensions = 2;
        m_f_old_data.canFreeData = false;
        b_st.site = &fd_emlrtRSI;
        d_mtimes(&b_st, delta_u, &m_f_old_data, y);
        st.site = &h_emlrtRSI;
        b_st.site = &gd_emlrtRSI;
        b_dynamic_size_checks(&b_st, y, K_new_inv, y->size[1],
                              K_new_inv->size[0]);
        b_st.site = &fd_emlrtRSI;
        c_mtimes(&b_st, y, K_new_inv, product3);
        product3_data = product3->data;
        /*  Step 2: Perform operations with scalars */
        st.site = &i_emlrtRSI;
        alpha3 = f[0] / (scalar2 * scalar2);
        signal1 = product1->size[0] * product1->size[1];
        partialTrueCount = (signal1 / 2) << 1;
        b_i = partialTrueCount - 2;
        for (i = 0; i <= b_i; i += 2) {
          r = _mm_loadu_pd(&product1_data[i]);
          _mm_storeu_pd(&product1_data[i], _mm_mul_pd(_mm_set1_pd(alpha3), r));
        }
        for (i = partialTrueCount; i < signal1; i++) {
          product1_data[i] *= alpha3;
        }
        if ((product2->size[0] != product3->size[0]) &&
            ((product2->size[0] != 1) && (product3->size[0] != 1))) {
          emlrtDimSizeImpxCheckR2021b(product2->size[0], product3->size[0],
                                      &c_emlrtECI, (emlrtConstCTX)sp);
        }
        if ((product2->size[1] != product3->size[1]) &&
            ((product2->size[1] != 1) && (product3->size[1] != 1))) {
          emlrtDimSizeImpxCheckR2021b(product2->size[1], product3->size[1],
                                      &d_emlrtECI, (emlrtConstCTX)sp);
        }
        if ((product2->size[0] == product3->size[0]) &&
            (product2->size[1] == product3->size[1])) {
          signal1 = product2->size[0] * product2->size[1];
          partialTrueCount = (signal1 / 2) << 1;
          b_i = partialTrueCount - 2;
          for (i = 0; i <= b_i; i += 2) {
            r = _mm_loadu_pd(&delta_u_data[i]);
            r1 = _mm_loadu_pd(&product3_data[i]);
            _mm_storeu_pd(&delta_u_data[i],
                          _mm_div_pd(_mm_add_pd(r, r1), _mm_set1_pd(scalar2)));
          }
          for (i = partialTrueCount; i < signal1; i++) {
            delta_u_data[i] = (delta_u_data[i] + product3_data[i]) / scalar2;
          }
        } else {
          st.site = &md_emlrtRSI;
          c_binary_expand_op(&st, product2, product3, scalar2);
          delta_u_data = product2->data;
        }
        /*  Step 3: Combine the parts */
        if ((K_new_inv->size[0] != product1->size[0]) &&
            ((K_new_inv->size[0] != 1) && (product1->size[0] != 1))) {
          emlrtDimSizeImpxCheckR2021b(K_new_inv->size[0], product1->size[0],
                                      &e_emlrtECI, (emlrtConstCTX)sp);
        }
        if ((K_new_inv->size[1] != product1->size[1]) &&
            ((K_new_inv->size[1] != 1) && (product1->size[1] != 1))) {
          emlrtDimSizeImpxCheckR2021b(K_new_inv->size[1], product1->size[1],
                                      &f_emlrtECI, (emlrtConstCTX)sp);
        }
        if ((K_new_inv->size[0] == product1->size[0]) &&
            (K_new_inv->size[1] == product1->size[1])) {
          signal1 = K_new_inv->size[0] * K_new_inv->size[1];
          partialTrueCount = (signal1 / 2) << 1;
          b_i = partialTrueCount - 2;
          for (i = 0; i <= b_i; i += 2) {
            r = _mm_loadu_pd(&K_new_inv_data[i]);
            r1 = _mm_loadu_pd(&product1_data[i]);
            _mm_storeu_pd(&K_new_inv_data[i], _mm_add_pd(r, r1));
          }
          for (i = partialTrueCount; i < signal1; i++) {
            K_new_inv_data[i] += product1_data[i];
          }
        } else {
          st.site = &ld_emlrtRSI;
          plus(&st, K_new_inv, product1);
          K_new_inv_data = K_new_inv->data;
        }
        if ((K_new_inv->size[0] != product2->size[0]) &&
            ((K_new_inv->size[0] != 1) && (product2->size[0] != 1))) {
          emlrtDimSizeImpxCheckR2021b(K_new_inv->size[0], product2->size[0],
                                      &e_emlrtECI, (emlrtConstCTX)sp);
        }
        if ((K_new_inv->size[1] != product2->size[1]) &&
            ((K_new_inv->size[1] != 1) && (product2->size[1] != 1))) {
          emlrtDimSizeImpxCheckR2021b(K_new_inv->size[1], product2->size[1],
                                      &f_emlrtECI, (emlrtConstCTX)sp);
        }
        if ((K_new_inv->size[0] == product2->size[0]) &&
            (K_new_inv->size[1] == product2->size[1])) {
          signal1 = K_new_inv->size[0] * K_new_inv->size[1];
          partialTrueCount = (signal1 / 2) << 1;
          b_i = partialTrueCount - 2;
          for (i = 0; i <= b_i; i += 2) {
            r = _mm_loadu_pd(&K_new_inv_data[i]);
            r1 = _mm_loadu_pd(&delta_u_data[i]);
            _mm_storeu_pd(&K_new_inv_data[i], _mm_sub_pd(r, r1));
          }
          for (i = partialTrueCount; i < signal1; i++) {
            K_new_inv_data[i] -= delta_u_data[i];
          }
        } else {
          st.site = &ld_emlrtRSI;
          minus(&st, K_new_inv, product2);
          K_new_inv_data = K_new_inv->data;
        }
        st.site = &j_emlrtRSI;
        i = product3->size[0] * product3->size[1];
        product3->size[0] = K_new_inv->size[0];
        product3->size[1] = K_new_inv->size[1];
        emxEnsureCapacity_real_T(&st, product3, i, &u_emlrtRTEI);
        product3_data = product3->data;
        signal1 = K_new_inv->size[0] * K_new_inv->size[1];
        partialTrueCount = (signal1 / 2) << 1;
        b_i = partialTrueCount - 2;
        for (i = 0; i <= b_i; i += 2) {
          r = _mm_loadu_pd(&K_new_inv_data[i]);
          _mm_storeu_pd(&product3_data[i], _mm_mul_pd(r, _mm_set1_pd(-1.0)));
        }
        for (i = partialTrueCount; i < signal1; i++) {
          product3_data[i] = -K_new_inv_data[i];
        }
        b_st.site = &gd_emlrtRSI;
        dynamic_size_checks(&b_st, product3, trueCount, product3->size[1],
                            trueCount);
        f_old_size = mtimes(product3, f_2_data, trueCount, f_old_data);
      }
      alpha3 = 1.0;
      memcpy(&ux[0], &u[0], 570U * sizeof(real_T));
      for (i = 0; i < trueCount; i++) {
        c_tmp_data[i] = (int32_T)dofs[b_tmp_data[i]];
      }
      if ((trueCount != f_old_size) &&
          ((trueCount != 1) && (f_old_size != 1))) {
        emlrtDimSizeImpxCheckR2021b(trueCount, f_old_size, &g_emlrtECI,
                                    (emlrtConstCTX)sp);
      }
      if (trueCount == f_old_size) {
        partialTrueCount = trueCount;
        for (i = 0; i < trueCount; i++) {
          d_tmp_data[i] = u[c_tmp_data[i] - 1] + f_old_data[i];
        }
      } else {
        partialTrueCount = b_binary_expand_op(
            d_tmp_data, u, c_tmp_data, &trueCount, f_old_data, &f_old_size);
      }
      if (trueCount != partialTrueCount) {
        emlrtSubAssignSizeCheck1dR2017a(trueCount, partialTrueCount,
                                        &h_emlrtECI, (emlrtConstCTX)sp);
      }
      for (i = 0; i < partialTrueCount; i++) {
        ux[c_tmp_data[i] - 1] = d_tmp_data[i];
      }
      st.site = &k_emlrtRSI;
      m_3 = mf_with_constr(&st, ux, X, conn, free_ind_data, trueCount,
                           old_constraints_data, old_constraints_size, ratio,
                           k_pen, Cx, Cy, Cz, R, slaves, dofs, s, sh, f);
      for (i = 0; i < trueCount; i++) {
        f_3_data[i] = f[(int32_T)dofs[b_tmp_data[i]] - 1];
      }
      signal1 = 0;
      y_size[0] = 1;
      y_size[1] = f_old_size;
      b_i = (f_old_size / 2) << 1;
      tmp_size_idx_0 = b_i - 2;
      for (i = 0; i <= tmp_size_idx_0; i += 2) {
        r = _mm_loadu_pd(&f_old_data[i]);
        _mm_storeu_pd(&f[i], _mm_mul_pd(_mm_set1_pd(0.0001), r));
      }
      for (i = b_i; i < f_old_size; i++) {
        f[i] = 0.0001 * f_old_data[i];
      }
      st.site = &l_emlrtRSI;
      b_st.site = &gd_emlrtRSI;
      c_dynamic_size_checks(&b_st, y_size, trueCount, f_old_size, trueCount);
      if (m_3 <= *m_new + e_mtimes(f, y_size, f_2_data)) {
        st.site = &l_emlrtRSI;
        b_st.site = &gd_emlrtRSI;
        d_dynamic_size_checks(&b_st, f_old_size, trueCount, f_old_size,
                              trueCount);
        st.site = &l_emlrtRSI;
        b_st.site = &gd_emlrtRSI;
        d_dynamic_size_checks(&b_st, f_old_size, trueCount, f_old_size,
                              trueCount);
        if (muDoubleScalarAbs(f_mtimes(f_old_data, f_old_size, f_3_data)) <=
            0.9 *
                muDoubleScalarAbs(f_mtimes(f_old_data, f_old_size, f_2_data))) {
          signal1 = 1;
        }
      }
      int32_T exitg2;
      do {
        exitg2 = 0;
        scalar2 = 0.0001 * alpha3;
        y_size[0] = 1;
        y_size[1] = f_old_size;
        for (i = 0; i <= tmp_size_idx_0; i += 2) {
          r = _mm_loadu_pd(&f_old_data[i]);
          _mm_storeu_pd(&f[i], _mm_mul_pd(_mm_set1_pd(scalar2), r));
        }
        for (i = b_i; i < f_old_size; i++) {
          f[i] = scalar2 * f_old_data[i];
        }
        st.site = &m_emlrtRSI;
        b_st.site = &gd_emlrtRSI;
        c_dynamic_size_checks(&b_st, y_size, trueCount, f_old_size, trueCount);
        if (m_3 < *m_new + e_mtimes(f, y_size, f_2_data)) {
          st.site = &m_emlrtRSI;
          b_st.site = &gd_emlrtRSI;
          d_dynamic_size_checks(&b_st, f_old_size, trueCount, f_old_size,
                                trueCount);
          y_size[0] = 1;
          y_size[1] = f_old_size;
          for (i = 0; i <= tmp_size_idx_0; i += 2) {
            r = _mm_loadu_pd(&f_old_data[i]);
            _mm_storeu_pd(&f[i], _mm_mul_pd(_mm_set1_pd(-0.9), r));
          }
          for (i = b_i; i < f_old_size; i++) {
            f[i] = -0.9 * f_old_data[i];
          }
          st.site = &m_emlrtRSI;
          b_st.site = &gd_emlrtRSI;
          c_dynamic_size_checks(&b_st, y_size, trueCount, f_old_size,
                                trueCount);
          if ((f_mtimes(f_old_data, f_old_size, f_3_data) <
               e_mtimes(f, y_size, f_2_data)) &&
              (signal1 == 0)) {
            alpha3 /= 0.5;
            memcpy(&ux[0], &u[0], 570U * sizeof(real_T));
            for (i = 0; i <= tmp_size_idx_0; i += 2) {
              r = _mm_loadu_pd(&f_old_data[i]);
              _mm_storeu_pd(&d_tmp_data[i], _mm_mul_pd(_mm_set1_pd(alpha3), r));
            }
            for (i = b_i; i < f_old_size; i++) {
              d_tmp_data[i] = alpha3 * f_old_data[i];
            }
            if ((trueCount != f_old_size) &&
                ((trueCount != 1) && (f_old_size != 1))) {
              emlrtDimSizeImpxCheckR2021b(trueCount, f_old_size, &i_emlrtECI,
                                          (emlrtConstCTX)sp);
            }
            if (trueCount == f_old_size) {
              partialTrueCount = trueCount;
              for (i = 0; i < trueCount; i++) {
                e_tmp_data[i] =
                    u[(int32_T)dofs[b_tmp_data[i]] - 1] + d_tmp_data[i];
              }
            } else {
              partialTrueCount =
                  binary_expand_op(e_tmp_data, u, dofs, b_tmp_data, &trueCount,
                                   d_tmp_data, &f_old_size);
            }
            if (trueCount != partialTrueCount) {
              emlrtSubAssignSizeCheck1dR2017a(trueCount, partialTrueCount,
                                              &j_emlrtECI, (emlrtConstCTX)sp);
            }
            for (i = 0; i < partialTrueCount; i++) {
              ux[c_tmp_data[i] - 1] = e_tmp_data[i];
            }
            st.site = &n_emlrtRSI;
            m_3 = mf_with_constr(&st, ux, X, conn, free_ind_data, trueCount,
                                 old_constraints_data, old_constraints_size,
                                 ratio, k_pen, Cx, Cy, Cz, R, slaves, dofs, s,
                                 sh, f);
            for (i = 0; i < trueCount; i++) {
              f_3_data[i] = f[(int32_T)dofs[b_tmp_data[i]] - 1];
            }
            scalar2 = 0.0001 * alpha3;
            y_size[0] = 1;
            y_size[1] = f_old_size;
            for (i = 0; i <= tmp_size_idx_0; i += 2) {
              r = _mm_loadu_pd(&f_old_data[i]);
              _mm_storeu_pd(&f[i], _mm_mul_pd(_mm_set1_pd(scalar2), r));
            }
            for (i = b_i; i < f_old_size; i++) {
              f[i] = scalar2 * f_old_data[i];
            }
            st.site = &o_emlrtRSI;
            b_st.site = &gd_emlrtRSI;
            c_dynamic_size_checks(&b_st, y_size, trueCount, f_old_size,
                                  trueCount);
            if (m_3 <= *m_new + e_mtimes(f, y_size, f_2_data)) {
              st.site = &o_emlrtRSI;
              b_st.site = &gd_emlrtRSI;
              d_dynamic_size_checks(&b_st, f_old_size, trueCount, f_old_size,
                                    trueCount);
              st.site = &o_emlrtRSI;
              b_st.site = &gd_emlrtRSI;
              d_dynamic_size_checks(&b_st, f_old_size, trueCount, f_old_size,
                                    trueCount);
              if (muDoubleScalarAbs(
                      f_mtimes(f_old_data, f_old_size, f_3_data)) <=
                  0.9 * muDoubleScalarAbs(
                            f_mtimes(f_old_data, f_old_size, f_2_data))) {
                signal1 = 1;
              }
            }
            if (*emlrtBreakCheckR2012bFlagVar != 0) {
              emlrtBreakCheckR2012b((emlrtConstCTX)sp);
            }
          } else {
            exitg2 = 1;
          }
        } else {
          exitg2 = 1;
        }
      } while (exitg2 == 0);
      if (signal1 == 0) {
        real_T alpha1;
        real_T alpha2;
        alpha1 = 0.0;
        alpha2 = alpha3 / 2.0;
        memcpy(&ux[0], &u[0], 570U * sizeof(real_T));
        for (i = 0; i <= tmp_size_idx_0; i += 2) {
          r = _mm_loadu_pd(&f_old_data[i]);
          _mm_storeu_pd(&d_tmp_data[i], _mm_mul_pd(_mm_set1_pd(alpha2), r));
        }
        for (i = b_i; i < f_old_size; i++) {
          d_tmp_data[i] = alpha2 * f_old_data[i];
        }
        if ((trueCount != f_old_size) &&
            ((trueCount != 1) && (f_old_size != 1))) {
          emlrtDimSizeImpxCheckR2021b(trueCount, f_old_size, &k_emlrtECI,
                                      (emlrtConstCTX)sp);
        }
        if (trueCount == f_old_size) {
          partialTrueCount = trueCount;
          for (i = 0; i < trueCount; i++) {
            e_tmp_data[i] = u[(int32_T)dofs[b_tmp_data[i]] - 1] + d_tmp_data[i];
          }
        } else {
          partialTrueCount =
              binary_expand_op(e_tmp_data, u, dofs, b_tmp_data, &trueCount,
                               d_tmp_data, &f_old_size);
        }
        if (trueCount != partialTrueCount) {
          emlrtSubAssignSizeCheck1dR2017a(trueCount, partialTrueCount,
                                          &l_emlrtECI, (emlrtConstCTX)sp);
        }
        for (i = 0; i < partialTrueCount; i++) {
          ux[c_tmp_data[i] - 1] = e_tmp_data[i];
        }
        st.site = &p_emlrtRSI;
        m_2 = mf_with_constr(&st, ux, X, conn, free_ind_data, trueCount,
                             old_constraints_data, old_constraints_size, ratio,
                             k_pen, Cx, Cy, Cz, R, slaves, dofs, s, sh, f);
        for (i = 0; i < trueCount; i++) {
          f_2_data[i] = f[(int32_T)dofs[b_tmp_data[i]] - 1];
        }
        signal2 = 0;
        while (signal2 == 0) {
          if (alpha3 - alpha1 < 1.0E-15) {
            signal2 = 1;
            m_2 = *m_new;
            if (trueCount - 1 >= 0) {
              memcpy(&f_2_data[0], &f_new_data[0],
                     (uint32_T)trueCount * sizeof(real_T));
            }
          } else {
            scalar2 = 0.0001 * alpha2;
            y_size[0] = 1;
            y_size[1] = f_old_size;
            for (i = 0; i <= tmp_size_idx_0; i += 2) {
              r = _mm_loadu_pd(&f_old_data[i]);
              _mm_storeu_pd(&f[i], _mm_mul_pd(_mm_set1_pd(scalar2), r));
            }
            for (i = b_i; i < f_old_size; i++) {
              f[i] = scalar2 * f_old_data[i];
            }
            st.site = &q_emlrtRSI;
            b_st.site = &gd_emlrtRSI;
            c_dynamic_size_checks(&b_st, y_size, trueCount, f_old_size,
                                  trueCount);
            if (m_2 > *m_new + e_mtimes(f, y_size, f_new_data)) {
              alpha3 = alpha2;
              m_3 = m_2;
              if (trueCount - 1 >= 0) {
                memcpy(&f_3_data[0], &f_2_data[0],
                       (uint32_T)trueCount * sizeof(real_T));
              }
              alpha2 = 0.5 * (alpha1 + alpha2);
              memcpy(&ux[0], &u[0], 570U * sizeof(real_T));
              for (i = 0; i <= tmp_size_idx_0; i += 2) {
                r = _mm_loadu_pd(&f_old_data[i]);
                _mm_storeu_pd(&d_tmp_data[i],
                              _mm_mul_pd(_mm_set1_pd(alpha2), r));
              }
              for (i = b_i; i < f_old_size; i++) {
                d_tmp_data[i] = alpha2 * f_old_data[i];
              }
              if ((trueCount != f_old_size) &&
                  ((trueCount != 1) && (f_old_size != 1))) {
                emlrtDimSizeImpxCheckR2021b(trueCount, f_old_size, &m_emlrtECI,
                                            (emlrtConstCTX)sp);
              }
              if (trueCount == f_old_size) {
                partialTrueCount = trueCount;
                for (i = 0; i < trueCount; i++) {
                  e_tmp_data[i] =
                      u[(int32_T)dofs[b_tmp_data[i]] - 1] + d_tmp_data[i];
                }
              } else {
                partialTrueCount =
                    binary_expand_op(e_tmp_data, u, dofs, b_tmp_data,
                                     &trueCount, d_tmp_data, &f_old_size);
              }
              if (trueCount != partialTrueCount) {
                emlrtSubAssignSizeCheck1dR2017a(trueCount, partialTrueCount,
                                                &n_emlrtECI, (emlrtConstCTX)sp);
              }
              for (i = 0; i < partialTrueCount; i++) {
                ux[c_tmp_data[i] - 1] = e_tmp_data[i];
              }
              st.site = &r_emlrtRSI;
              m_2 = mf_with_constr(&st, ux, X, conn, free_ind_data, trueCount,
                                   old_constraints_data, old_constraints_size,
                                   ratio, k_pen, Cx, Cy, Cz, R, slaves, dofs, s,
                                   sh, f);
              for (i = 0; i < trueCount; i++) {
                f_2_data[i] = f[(int32_T)dofs[b_tmp_data[i]] - 1];
              }
            } else {
              st.site = &s_emlrtRSI;
              b_st.site = &gd_emlrtRSI;
              d_dynamic_size_checks(&b_st, f_old_size, trueCount, f_old_size,
                                    trueCount);
              scalar2 = f_mtimes(f_old_data, f_old_size, f_2_data);
              y_size[0] = 1;
              y_size[1] = f_old_size;
              for (i = 0; i <= tmp_size_idx_0; i += 2) {
                r = _mm_loadu_pd(&f_old_data[i]);
                _mm_storeu_pd(&f[i], _mm_mul_pd(_mm_set1_pd(0.9), r));
              }
              for (i = b_i; i < f_old_size; i++) {
                f[i] = 0.9 * f_old_data[i];
              }
              st.site = &s_emlrtRSI;
              b_st.site = &gd_emlrtRSI;
              c_dynamic_size_checks(&b_st, y_size, trueCount, f_old_size,
                                    trueCount);
              if (scalar2 < e_mtimes(f, y_size, f_new_data)) {
                alpha1 = alpha2;
                alpha2 = 0.5 * (alpha2 + alpha3);
                memcpy(&ux[0], &u[0], 570U * sizeof(real_T));
                for (i = 0; i <= tmp_size_idx_0; i += 2) {
                  r = _mm_loadu_pd(&f_old_data[i]);
                  _mm_storeu_pd(&d_tmp_data[i],
                                _mm_mul_pd(_mm_set1_pd(alpha2), r));
                }
                for (i = b_i; i < f_old_size; i++) {
                  d_tmp_data[i] = alpha2 * f_old_data[i];
                }
                if ((trueCount != f_old_size) &&
                    ((trueCount != 1) && (f_old_size != 1))) {
                  emlrtDimSizeImpxCheckR2021b(trueCount, f_old_size,
                                              &o_emlrtECI, (emlrtConstCTX)sp);
                }
                if (trueCount == f_old_size) {
                  partialTrueCount = trueCount;
                  for (i = 0; i < trueCount; i++) {
                    e_tmp_data[i] =
                        u[(int32_T)dofs[b_tmp_data[i]] - 1] + d_tmp_data[i];
                  }
                } else {
                  partialTrueCount =
                      binary_expand_op(e_tmp_data, u, dofs, b_tmp_data,
                                       &trueCount, d_tmp_data, &f_old_size);
                }
                if (trueCount != partialTrueCount) {
                  emlrtSubAssignSizeCheck1dR2017a(trueCount, partialTrueCount,
                                                  &p_emlrtECI,
                                                  (emlrtConstCTX)sp);
                }
                for (i = 0; i < partialTrueCount; i++) {
                  ux[c_tmp_data[i] - 1] = e_tmp_data[i];
                }
                st.site = &t_emlrtRSI;
                m_2 = mf_with_constr(&st, ux, X, conn, free_ind_data, trueCount,
                                     old_constraints_data, old_constraints_size,
                                     ratio, k_pen, Cx, Cy, Cz, R, slaves, dofs,
                                     s, sh, f);
                for (i = 0; i < trueCount; i++) {
                  f_2_data[i] = f[(int32_T)dofs[b_tmp_data[i]] - 1];
                }
              } else {
                st.site = &u_emlrtRSI;
                b_st.site = &gd_emlrtRSI;
                d_dynamic_size_checks(&b_st, f_old_size, trueCount, f_old_size,
                                      trueCount);
                y_size[0] = 1;
                y_size[1] = f_old_size;
                for (i = 0; i <= tmp_size_idx_0; i += 2) {
                  r = _mm_loadu_pd(&f_old_data[i]);
                  _mm_storeu_pd(&f[i], _mm_mul_pd(_mm_set1_pd(-0.9), r));
                }
                for (i = b_i; i < f_old_size; i++) {
                  f[i] = -0.9 * f_old_data[i];
                }
                st.site = &u_emlrtRSI;
                b_st.site = &gd_emlrtRSI;
                c_dynamic_size_checks(&b_st, y_size, trueCount, f_old_size,
                                      trueCount);
                if (scalar2 > e_mtimes(f, y_size, f_new_data)) {
                  alpha3 = alpha2;
                  m_3 = m_2;
                  if (trueCount - 1 >= 0) {
                    memcpy(&f_3_data[0], &f_2_data[0],
                           (uint32_T)trueCount * sizeof(real_T));
                  }
                  alpha2 = 0.5 * (alpha1 + alpha2);
                  memcpy(&ux[0], &u[0], 570U * sizeof(real_T));
                  for (i = 0; i <= tmp_size_idx_0; i += 2) {
                    r = _mm_loadu_pd(&f_old_data[i]);
                    _mm_storeu_pd(&d_tmp_data[i],
                                  _mm_mul_pd(_mm_set1_pd(alpha2), r));
                  }
                  for (i = b_i; i < f_old_size; i++) {
                    d_tmp_data[i] = alpha2 * f_old_data[i];
                  }
                  if ((trueCount != f_old_size) &&
                      ((trueCount != 1) && (f_old_size != 1))) {
                    emlrtDimSizeImpxCheckR2021b(trueCount, f_old_size,
                                                &q_emlrtECI, (emlrtConstCTX)sp);
                  }
                  if (trueCount == f_old_size) {
                    partialTrueCount = trueCount;
                    for (i = 0; i < trueCount; i++) {
                      e_tmp_data[i] =
                          u[(int32_T)dofs[b_tmp_data[i]] - 1] + d_tmp_data[i];
                    }
                  } else {
                    partialTrueCount =
                        binary_expand_op(e_tmp_data, u, dofs, b_tmp_data,
                                         &trueCount, d_tmp_data, &f_old_size);
                  }
                  if (trueCount != partialTrueCount) {
                    emlrtSubAssignSizeCheck1dR2017a(trueCount, partialTrueCount,
                                                    &r_emlrtECI,
                                                    (emlrtConstCTX)sp);
                  }
                  for (i = 0; i < partialTrueCount; i++) {
                    ux[c_tmp_data[i] - 1] = e_tmp_data[i];
                  }
                  st.site = &v_emlrtRSI;
                  m_2 = mf_with_constr(&st, ux, X, conn, free_ind_data,
                                       trueCount, old_constraints_data,
                                       old_constraints_size, ratio, k_pen, Cx,
                                       Cy, Cz, R, slaves, dofs, s, sh, f);
                  for (i = 0; i < trueCount; i++) {
                    f_2_data[i] = f[(int32_T)dofs[b_tmp_data[i]] - 1];
                  }
                } else {
                  signal2 = 1;
                }
              }
            }
          }
          if (*emlrtBreakCheckR2012bFlagVar != 0) {
            emlrtBreakCheckR2012b((emlrtConstCTX)sp);
          }
        }
      }
      i = delta_u->size[0] * delta_u->size[1];
      delta_u->size[0] = trueCount;
      delta_u->size[1] = 1;
      emxEnsureCapacity_real_T(sp, delta_u, i, &t_emlrtRTEI);
      delta_u_data = delta_u->data;
      for (i = 0; i < trueCount; i++) {
        i1 = (int32_T)dofs[b_tmp_data[i]] - 1;
        delta_u_data[i] = ux[i1] - u[i1];
      }
      memcpy(&u[0], &ux[0], 570U * sizeof(real_T));
      if (signal1 == 1) {
        *m_new = m_3;
        if (trueCount - 1 >= 0) {
          memcpy(&f_2_data[0], &f_3_data[0],
                 (uint32_T)trueCount * sizeof(real_T));
        }
      } else {
        *m_new = m_2;
      }
      if (*emlrtBreakCheckR2012bFlagVar != 0) {
        emlrtBreakCheckR2012b((emlrtConstCTX)sp);
      }
    } else {
      exitg1 = 1;
    }
  } while (exitg1 == 0);
  emxFree_real_T(sp, &y);
  emxFree_real_T(sp, &product3);
  emxFree_real_T(sp, &product2);
  emxFree_real_T(sp, &product1);
  emxFree_real_T(sp, &delta_u);
  emxFree_real_T(sp, &K_new_inv);
  /*  sprintf('fullBFGS: %s',int2str(iter)) */
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtConstCTX)sp);
}

/* End of code generation (fullBFGStoCompileToMex.c) */

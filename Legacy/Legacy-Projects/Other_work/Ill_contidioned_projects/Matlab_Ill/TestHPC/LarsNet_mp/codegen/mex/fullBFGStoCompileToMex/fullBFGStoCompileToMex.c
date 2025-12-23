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
#include <string.h>

/* Variable Definitions */
static emlrtRSInfo emlrtRSI = {
    22,                       /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo b_emlrtRSI = {
    24,                       /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo c_emlrtRSI = {
    51,                       /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo d_emlrtRSI = {
    55,                       /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo e_emlrtRSI = {
    56,                       /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo f_emlrtRSI = {
    66,                       /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo g_emlrtRSI = {
    67,                       /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo h_emlrtRSI = {
    68,                       /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo i_emlrtRSI = {
    71,                       /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo j_emlrtRSI = {
    78,                       /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo k_emlrtRSI = {
    85,                       /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo l_emlrtRSI = {
    89,                       /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo m_emlrtRSI = {
    94,                       /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo n_emlrtRSI = {
    100,                      /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo o_emlrtRSI = {
    105,                      /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo p_emlrtRSI = {
    119,                      /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo q_emlrtRSI = {
    134,                      /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo r_emlrtRSI = {
    144,                      /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo s_emlrtRSI = {
    148,                      /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo t_emlrtRSI = {
    155,                      /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo u_emlrtRSI = {
    158,                      /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo v_emlrtRSI = {
    168,                      /* lineNo */
    "fullBFGStoCompileToMex", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo w_emlrtRSI = {
    243,              /* lineNo */
    "mf_with_constr", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo x_emlrtRSI = {
    249,              /* lineNo */
    "mf_with_constr", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo y_emlrtRSI = {
    254,              /* lineNo */
    "mf_with_constr", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo ab_emlrtRSI = {
    256,              /* lineNo */
    "mf_with_constr", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo bb_emlrtRSI = {
    299,              /* lineNo */
    "mf_with_constr", /* fcnName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pathName */
};

static emlrtRSInfo cb_emlrtRSI = {
    44,       /* lineNo */
    "mpower", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/ops/mpower.m" /* pathName */
};

static emlrtRSInfo eb_emlrtRSI = {
    32,     /* lineNo */
    "sort", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/datafun/sort.m" /* pathName */
};

static emlrtRSInfo mc_emlrtRSI = {
    19,        /* lineNo */
    "setdiff", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/ops/setdiff.m" /* pathName */
};

static emlrtRSInfo nc_emlrtRSI = {
    70,          /* lineNo */
    "eml_setop", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/ops/private/eml_setop.m" /* pathName */
};

static emlrtRSInfo md_emlrtRSI = {
    91,                  /* lineNo */
    "eml_mtimes_helper", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/ops/eml_mtimes_helper.m" /* pathName */
};

static emlrtRSInfo nd_emlrtRSI = {
    60,                  /* lineNo */
    "eml_mtimes_helper", /* fcnName */
    "/mnt/irisgpfs/apps/resif/aion/2020b/epyc/software/MATLAB/2021a/toolbox/"
    "eml/lib/matlab/ops/eml_mtimes_helper.m" /* pathName */
};

static emlrtDCInfo emlrtDCI = {
    224,              /* lineNo */
    16,               /* colNo */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    1                           /* checkKind */
};

static emlrtBCInfo emlrtBCI = {
    1,                /* iFirst */
    243,              /* iLast */
    224,              /* lineNo */
    16,               /* colNo */
    "X",              /* aName */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    0                           /* checkKind */
};

static emlrtDCInfo b_emlrtDCI = {
    225,              /* lineNo */
    16,               /* colNo */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    1                           /* checkKind */
};

static emlrtBCInfo b_emlrtBCI = {
    1,                /* iFirst */
    243,              /* iLast */
    225,              /* lineNo */
    16,               /* colNo */
    "X",              /* aName */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    0                           /* checkKind */
};

static emlrtBCInfo c_emlrtBCI = {
    -1,                /* iFirst */
    -1,                /* iLast */
    265,               /* lineNo */
    31,                /* colNo */
    "old_constraints", /* aName */
    "mf_with_constr",  /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    0                           /* checkKind */
};

static emlrtBCInfo d_emlrtBCI = {
    1,                /* iFirst */
    243,              /* iLast */
    260,              /* lineNo */
    12,               /* colNo */
    "X",              /* aName */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    0                           /* checkKind */
};

static emlrtDCInfo c_emlrtDCI = {
    260,              /* lineNo */
    12,               /* colNo */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    1                           /* checkKind */
};

static emlrtBCInfo e_emlrtBCI = {
    1,                /* iFirst */
    729,              /* iLast */
    260,              /* lineNo */
    26,               /* colNo */
    "u",              /* aName */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    0                           /* checkKind */
};

static emlrtDCInfo d_emlrtDCI = {
    260,              /* lineNo */
    26,               /* colNo */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    1                           /* checkKind */
};

static emlrtDCInfo e_emlrtDCI = {
    260,              /* lineNo */
    31,               /* colNo */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    1                           /* checkKind */
};

static emlrtBCInfo f_emlrtBCI = {
    1,                /* iFirst */
    729,              /* iLast */
    249,              /* lineNo */
    19,               /* colNo */
    "f",              /* aName */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    0                           /* checkKind */
};

static emlrtDCInfo f_emlrtDCI = {
    249,              /* lineNo */
    19,               /* colNo */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    1                           /* checkKind */
};

static emlrtBCInfo g_emlrtBCI = {
    1,                /* iFirst */
    729,              /* iLast */
    249,              /* lineNo */
    9,                /* colNo */
    "f",              /* aName */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    3                           /* checkKind */
};

static emlrtDCInfo g_emlrtDCI = {
    249,              /* lineNo */
    9,                /* colNo */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    1                           /* checkKind */
};

static emlrtBCInfo h_emlrtBCI = {
    1,                /* iFirst */
    729,              /* iLast */
    297,              /* lineNo */
    9,                /* colNo */
    "f",              /* aName */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    3                           /* checkKind */
};

static emlrtDCInfo h_emlrtDCI = {
    297,              /* lineNo */
    9,                /* colNo */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    1                           /* checkKind */
};

static emlrtBCInfo i_emlrtBCI = {
    1,                /* iFirst */
    729,              /* iLast */
    297,              /* lineNo */
    16,               /* colNo */
    "f",              /* aName */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    0                           /* checkKind */
};

static emlrtDCInfo i_emlrtDCI = {
    297,              /* lineNo */
    16,               /* colNo */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    1                           /* checkKind */
};

static emlrtBCInfo j_emlrtBCI = {
    1,                /* iFirst */
    729,              /* iLast */
    225,              /* lineNo */
    26,               /* colNo */
    "u",              /* aName */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    0                           /* checkKind */
};

static emlrtDCInfo j_emlrtDCI = {
    225,              /* lineNo */
    26,               /* colNo */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    1                           /* checkKind */
};

static emlrtBCInfo k_emlrtBCI = {
    1,                /* iFirst */
    729,              /* iLast */
    224,              /* lineNo */
    26,               /* colNo */
    "u",              /* aName */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    0                           /* checkKind */
};

static emlrtDCInfo k_emlrtDCI = {
    224,              /* lineNo */
    26,               /* colNo */
    "mf_with_constr", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    1                           /* checkKind */
};

static emlrtECInfo emlrtECI = {
    2,                        /* nDims */
    55,                       /* lineNo */
    22,                       /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo b_emlrtECI = {
    -1,                       /* nDims */
    55,                       /* lineNo */
    9,                        /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo c_emlrtECI = {
    -1,                       /* nDims */
    56,                       /* lineNo */
    9,                        /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo d_emlrtECI = {
    2,                        /* nDims */
    72,                       /* lineNo */
    18,                       /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo e_emlrtECI = {
    2,                        /* nDims */
    75,                       /* lineNo */
    21,                       /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo f_emlrtECI = {
    -1,                       /* nDims */
    84,                       /* lineNo */
    18,                       /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo g_emlrtECI = {
    -1,                       /* nDims */
    99,                       /* lineNo */
    22,                       /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo h_emlrtECI = {
    -1,                       /* nDims */
    118,                      /* lineNo */
    22,                       /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo i_emlrtECI = {
    -1,                       /* nDims */
    143,                      /* lineNo */
    30,                       /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo j_emlrtECI = {
    -1,                       /* nDims */
    154,                      /* lineNo */
    34,                       /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtECInfo k_emlrtECI = {
    -1,                       /* nDims */
    167,                      /* lineNo */
    34,                       /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtBCInfo l_emlrtBCI = {
    1,                        /* iFirst */
    729,                      /* iLast */
    23,                       /* lineNo */
    7,                        /* colNo */
    "f",                      /* aName */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    0                           /* checkKind */
};

static emlrtDCInfo l_emlrtDCI = {
    23,                       /* lineNo */
    7,                        /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m", /* pName */
    1                           /* checkKind */
};

static emlrtRTEInfo q_emlrtRTEI = {
    31,                       /* lineNo */
    1,                        /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo r_emlrtRTEI = {
    48,                       /* lineNo */
    5,                        /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo s_emlrtRTEI = {
    51,                       /* lineNo */
    15,                       /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo t_emlrtRTEI = {
    179,                      /* lineNo */
    5,                        /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo u_emlrtRTEI = {
    78,                       /* lineNo */
    15,                       /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo v_emlrtRTEI = {
    24,                       /* lineNo */
    1,                        /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo w_emlrtRTEI = {
    32,                       /* lineNo */
    1,                        /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo x_emlrtRTEI = {
    66,                       /* lineNo */
    9,                        /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo y_emlrtRTEI = {
    67,                       /* lineNo */
    9,                        /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo ab_emlrtRTEI = {
    55,                       /* lineNo */
    22,                       /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

static emlrtRTEInfo bb_emlrtRTEI = {
    55,                       /* lineNo */
    39,                       /* colNo */
    "fullBFGStoCompileToMex", /* fName */
    "/mnt/irisgpfs/users/dihurtado/Matlab/LarsNet_mp/fnctns/"
    "fullBFGStoCompileToMex.m" /* pName */
};

/* Function Declarations */
static void mf_with_constr(const emlrtStack *sp, const real_T u[729],
                           const real_T X[729], const real_T conn[740],
                           const real_T free_ind_data[], int32_T free_ind_size,
                           const real_T old_constraints_data[],
                           const int32_T old_constraints_size[2], real_T ratio,
                           real_T k_pen, real_T Cx, real_T Cy, real_T Cz,
                           real_T R, const real_T slaves[222],
                           const real_T dofs[729], const real_T s[444],
                           const real_T sh[444], real_T *m, real_T f[729]);

/* Function Definitions */
static void mf_with_constr(const emlrtStack *sp, const real_T u[729],
                           const real_T X[729], const real_T conn[740],
                           const real_T free_ind_data[], int32_T free_ind_size,
                           const real_T old_constraints_data[],
                           const int32_T old_constraints_size[2], real_T ratio,
                           real_T k_pen, real_T Cx, real_T Cy, real_T Cz,
                           real_T R, const real_T slaves[222],
                           const real_T dofs[729], const real_T s[444],
                           const real_T sh[444], real_T *m, real_T f[729])
{
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack st;
  real_T c_data[729];
  real_T free_ind_sorted_data[729];
  real_T xs[666];
  real_T a[6];
  real_T b_f[6];
  real_T dofi[6];
  real_T Cxy_idx_0;
  real_T Cxy_idx_1;
  real_T L;
  real_T L0;
  real_T absxk;
  real_T b_scale;
  real_T d;
  real_T k;
  real_T scale;
  real_T t;
  real_T x1_idx_0;
  real_T x1_idx_1;
  real_T x1_idx_1_tmp;
  real_T x1_idx_2;
  real_T x1_idx_2_tmp;
  real_T x2_idx_0;
  real_T x2_idx_1;
  real_T x2_idx_1_tmp;
  real_T x2_idx_2;
  real_T x2_idx_2_tmp;
  int32_T ia_data[729];
  int32_T c_size[2];
  int32_T b_i;
  int32_T i;
  int32_T i_cnstr;
  int32_T ib_size;
  boolean_T x[2];
  boolean_T exitg1;
  boolean_T y;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  /* UNTITLED Summary of this function goes here */
  /*    Detailed explanation goes here */
  /*  m=mp('0');  % it will adapt to datatype in the 'for' loop */
  *m = 0.0;
  /*  it will adapt to datatype in the 'for' loop */
  /*  f=zeros(len_u,1,'like',u);   */
  /*  f=zeros(len_u,1,'mp'); */
  memset(&f[0], 0, 729U * sizeof(real_T));
  for (i = 0; i < 370; i++) {
    d = conn[i];
    dofi[0] = d * 3.0 - 2.0;
    dofi[1] = d * 3.0 - 1.0;
    dofi[2] = d * 3.0;
    scale = conn[i + 370];
    Cxy_idx_1 = scale * 3.0;
    dofi[3] = Cxy_idx_1 - 2.0;
    dofi[4] = Cxy_idx_1 - 1.0;
    dofi[5] = Cxy_idx_1;
    if (d != (int32_T)muDoubleScalarFloor(d)) {
      emlrtIntegerCheckR2012b(d, &emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)d < 1) || ((int32_T)d > 243)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)d, 1, 243, &emlrtBCI,
                                    (emlrtCTX)sp);
    }
    L0 = 3.0 * d;
    if (L0 + -2.0 != (int32_T)muDoubleScalarFloor(L0 + -2.0)) {
      emlrtIntegerCheckR2012b(L0 + -2.0, &k_emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)(L0 + -2.0) < 1) || ((int32_T)(L0 + -2.0) > 729)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)(L0 + -2.0), 1, 729, &k_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    Cxy_idx_0 = X[(int32_T)d - 1];
    x1_idx_0 = Cxy_idx_0 + u[(int32_T)(L0 + -2.0) - 1];
    if (L0 + -1.0 != (int32_T)muDoubleScalarFloor(L0 + -1.0)) {
      emlrtIntegerCheckR2012b(L0 + -1.0, &k_emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)(L0 + -1.0) < 1) || ((int32_T)(L0 + -1.0) > 729)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)(L0 + -1.0), 1, 729, &k_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    x1_idx_1_tmp = X[(int32_T)d + 242];
    x1_idx_1 = x1_idx_1_tmp + u[(int32_T)(L0 + -1.0) - 1];
    if (L0 != (int32_T)muDoubleScalarFloor(L0)) {
      emlrtIntegerCheckR2012b(L0, &k_emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)L0 < 1) || ((int32_T)L0 > 729)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)L0, 1, 729, &k_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    x1_idx_2_tmp = X[(int32_T)d + 485];
    x1_idx_2 = x1_idx_2_tmp + u[(int32_T)L0 - 1];
    if (scale != (int32_T)muDoubleScalarFloor(scale)) {
      emlrtIntegerCheckR2012b(scale, &b_emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)scale < 1) || ((int32_T)scale > 243)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)scale, 1, 243, &b_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    if (Cxy_idx_1 + -2.0 != (int32_T)muDoubleScalarFloor(Cxy_idx_1 + -2.0)) {
      emlrtIntegerCheckR2012b(Cxy_idx_1 + -2.0, &j_emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)(Cxy_idx_1 + -2.0) < 1) ||
        ((int32_T)(Cxy_idx_1 + -2.0) > 729)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)(Cxy_idx_1 + -2.0), 1, 729,
                                    &j_emlrtBCI, (emlrtCTX)sp);
    }
    k = X[(int32_T)scale - 1];
    x2_idx_0 = k + u[(int32_T)(Cxy_idx_1 + -2.0) - 1];
    if (Cxy_idx_1 + -1.0 != (int32_T)muDoubleScalarFloor(Cxy_idx_1 + -1.0)) {
      emlrtIntegerCheckR2012b(Cxy_idx_1 + -1.0, &j_emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)(Cxy_idx_1 + -1.0) < 1) ||
        ((int32_T)(Cxy_idx_1 + -1.0) > 729)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)(Cxy_idx_1 + -1.0), 1, 729,
                                    &j_emlrtBCI, (emlrtCTX)sp);
    }
    x2_idx_1_tmp = X[(int32_T)scale + 242];
    x2_idx_1 = x2_idx_1_tmp + u[(int32_T)(Cxy_idx_1 + -1.0) - 1];
    if (Cxy_idx_1 != (int32_T)muDoubleScalarFloor(Cxy_idx_1)) {
      emlrtIntegerCheckR2012b(Cxy_idx_1, &j_emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)Cxy_idx_1 < 1) || ((int32_T)Cxy_idx_1 > 729)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)Cxy_idx_1, 1, 729, &j_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    x2_idx_2_tmp = X[(int32_T)scale + 485];
    x2_idx_2 = x2_idx_2_tmp + u[(int32_T)Cxy_idx_1 - 1];
    b_scale = 3.3121686421112381E-170;
    scale = 3.3121686421112381E-170;
    absxk = muDoubleScalarAbs(Cxy_idx_0 - k);
    if (absxk > 3.3121686421112381E-170) {
      L0 = 1.0;
      b_scale = absxk;
    } else {
      t = absxk / 3.3121686421112381E-170;
      L0 = t * t;
    }
    d = x1_idx_0 - x2_idx_0;
    Cxy_idx_0 = d;
    absxk = muDoubleScalarAbs(d);
    if (absxk > 3.3121686421112381E-170) {
      L = 1.0;
      scale = absxk;
    } else {
      t = absxk / 3.3121686421112381E-170;
      L = t * t;
    }
    absxk = muDoubleScalarAbs(x1_idx_1_tmp - x2_idx_1_tmp);
    if (absxk > b_scale) {
      t = b_scale / absxk;
      L0 = L0 * t * t + 1.0;
      b_scale = absxk;
    } else {
      t = absxk / b_scale;
      L0 += t * t;
    }
    d = x1_idx_1 - x2_idx_1;
    Cxy_idx_1 = d;
    absxk = muDoubleScalarAbs(d);
    if (absxk > scale) {
      t = scale / absxk;
      L = L * t * t + 1.0;
      scale = absxk;
    } else {
      t = absxk / scale;
      L += t * t;
    }
    absxk = muDoubleScalarAbs(x1_idx_2_tmp - x2_idx_2_tmp);
    if (absxk > b_scale) {
      t = b_scale / absxk;
      L0 = L0 * t * t + 1.0;
      b_scale = absxk;
    } else {
      t = absxk / b_scale;
      L0 += t * t;
    }
    d = x1_idx_2 - x2_idx_2;
    absxk = muDoubleScalarAbs(d);
    if (absxk > scale) {
      t = scale / absxk;
      L = L * t * t + 1.0;
      scale = absxk;
    } else {
      t = absxk / scale;
      L += t * t;
    }
    L0 = b_scale * muDoubleScalarSqrt(L0);
    L = scale * muDoubleScalarSqrt(L);
    if (L > L0) {
      k = 1.0;
    } else {
      k = ratio;
    }
    st.site = &w_emlrtRSI;
    b_st.site = &w_emlrtRSI;
    scale = muDoubleScalarLog(L / L0);
    b_st.site = &cb_emlrtRSI;
    *m += k * L0 * (0.5 * (scale * scale));
    st.site = &x_emlrtRSI;
    scale *= k * (L0 / L);
    a[0] = scale * (Cxy_idx_0 / L);
    a[3] = scale * ((x2_idx_0 - x1_idx_0) / L);
    a[1] = scale * (Cxy_idx_1 / L);
    a[4] = scale * ((x2_idx_1 - x1_idx_1) / L);
    a[2] = scale * (d / L);
    a[5] = scale * ((x2_idx_2 - x1_idx_2) / L);
    for (b_i = 0; b_i < 6; b_i++) {
      d = dofi[b_i];
      if (d != (int32_T)muDoubleScalarFloor(d)) {
        emlrtIntegerCheckR2012b(d, &f_emlrtDCI, (emlrtCTX)sp);
      }
      if (((int32_T)d < 1) || ((int32_T)d > 729)) {
        emlrtDynamicBoundsCheckR2012b((int32_T)d, 1, 729, &f_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      b_f[b_i] = f[(int32_T)d - 1] + a[b_i];
    }
    for (b_i = 0; b_i < 6; b_i++) {
      d = dofi[b_i];
      if (d != (int32_T)muDoubleScalarFloor(d)) {
        emlrtIntegerCheckR2012b(d, &g_emlrtDCI, (emlrtCTX)sp);
      }
      if (((int32_T)d < 1) || ((int32_T)d > 729)) {
        emlrtDynamicBoundsCheckR2012b((int32_T)d, 1, 729, &g_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      f[(int32_T)d - 1] = b_f[b_i];
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  /*  Sort 'free_ind' in ascending order */
  st.site = &y_emlrtRSI;
  if (0 <= free_ind_size - 1) {
    memcpy(&free_ind_sorted_data[0], &free_ind_data[0],
           free_ind_size * sizeof(real_T));
  }
  b_st.site = &eb_emlrtRSI;
  sort(&b_st, free_ind_sorted_data, &free_ind_size);
  /*  Use the sorted array with 'setdiff' */
  st.site = &ab_emlrtRSI;
  b_st.site = &mc_emlrtRSI;
  c_st.site = &nc_emlrtRSI;
  do_vectors(&c_st, free_ind_sorted_data, free_ind_size, c_data, c_size,
             ia_data, &i, &ib_size);
  i = c_size[1];
  ib_size = c_size[1];
  for (b_i = 0; b_i < ib_size; b_i++) {
    ia_data[b_i] = (int32_T)c_data[b_i];
  }
  for (b_i = 0; b_i < i; b_i++) {
    f[ia_data[b_i] - 1] = 0.0;
  }
  /*  f(setdiff(1:numel(f),free_ind))=0; */
  for (b_i = 0; b_i < 3; b_i++) {
    for (i_cnstr = 0; i_cnstr < 222; i_cnstr++) {
      d = slaves[i_cnstr];
      i = (int32_T)muDoubleScalarFloor(d);
      if (d != i) {
        emlrtIntegerCheckR2012b(d, &c_emlrtDCI, (emlrtCTX)sp);
      }
      ib_size = (int32_T)d;
      if ((ib_size < 1) || (ib_size > 243)) {
        emlrtDynamicBoundsCheckR2012b(ib_size, 1, 243, &d_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      if (ib_size != i) {
        emlrtIntegerCheckR2012b(d, &e_emlrtDCI, (emlrtCTX)sp);
      }
      i = (ib_size + 243 * b_i) - 1;
      d = dofs[i];
      if (d != (int32_T)muDoubleScalarFloor(d)) {
        emlrtIntegerCheckR2012b(d, &d_emlrtDCI, (emlrtCTX)sp);
      }
      if (((int32_T)d < 1) || ((int32_T)d > 729)) {
        emlrtDynamicBoundsCheckR2012b((int32_T)d, 1, 729, &e_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      xs[i_cnstr + 222 * b_i] = X[i] + u[(int32_T)d - 1];
    }
  }
  b_i = old_constraints_size[1];
  for (i_cnstr = 0; i_cnstr < b_i; i_cnstr++) {
    if (i_cnstr + 1 > old_constraints_size[1]) {
      emlrtDynamicBoundsCheckR2012b(i_cnstr + 1, 1, old_constraints_size[1],
                                    &c_emlrtBCI, (emlrtCTX)sp);
    }
    /*  % Fless case */
    /*  g =norm(xsi - Cxy) - R; */
    /*  dgdu = (xsi-Cxy)/norm(xsi-Cxy); */
    /*   */
    i = (int32_T)old_constraints_data[i_cnstr];
    x[0] = (sh[i - 1] == -1.0);
    x[1] = (sh[i + 221] == -1.0);
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
      b_scale = 3.3121686421112381E-170;
      d = xs[i - 1] - Cx;
      x1_idx_0 = d;
      absxk = muDoubleScalarAbs(d);
      if (absxk > 3.3121686421112381E-170) {
        k = 1.0;
        b_scale = absxk;
      } else {
        t = absxk / 3.3121686421112381E-170;
        k = t * t;
      }
      d = xs[i + 221] - Cy;
      x1_idx_1 = d;
      absxk = muDoubleScalarAbs(d);
      if (absxk > b_scale) {
        t = b_scale / absxk;
        k = k * t * t + 1.0;
        b_scale = absxk;
      } else {
        t = absxk / b_scale;
        k += t * t;
      }
      d = xs[i + 443] - Cz;
      absxk = muDoubleScalarAbs(d);
      if (absxk > b_scale) {
        t = b_scale / absxk;
        k = k * t * t + 1.0;
        b_scale = absxk;
      } else {
        t = absxk / b_scale;
        k += t * t;
      }
      k = b_scale * muDoubleScalarSqrt(k);
      Cxy_idx_0 = k - R;
      x1_idx_0 /= k;
      x1_idx_1 /= k;
      x1_idx_2 = d / k;
    } else {
      /*  theta = pi*(1-s(cnt_act,2)); */
      /*  phi = pi*s(cnt_act,1); */
      k = 3.1415926535897931 *
          (1.0 - s[(int32_T)old_constraints_data[i_cnstr] + 221]);
      scale =
          3.1415926535897931 * s[(int32_T)old_constraints_data[i_cnstr] - 1];
      Cxy_idx_1 = muDoubleScalarSin(k);
      b_scale = 3.3121686421112381E-170;
      d = xs[i - 1] - (Cx + -(muDoubleScalarCos(scale) * Cxy_idx_1) * R);
      x1_idx_0 = d;
      absxk = muDoubleScalarAbs(d);
      if (absxk > 3.3121686421112381E-170) {
        Cxy_idx_0 = 1.0;
        b_scale = absxk;
      } else {
        t = absxk / 3.3121686421112381E-170;
        Cxy_idx_0 = t * t;
      }
      d = xs[i + 221] - (Cy + -muDoubleScalarCos(k) * R);
      x1_idx_1 = d;
      absxk = muDoubleScalarAbs(d);
      if (absxk > b_scale) {
        t = b_scale / absxk;
        Cxy_idx_0 = Cxy_idx_0 * t * t + 1.0;
        b_scale = absxk;
      } else {
        t = absxk / b_scale;
        Cxy_idx_0 += t * t;
      }
      d = xs[i + 443] - (Cz + -(Cxy_idx_1 * muDoubleScalarSin(scale)) * R);
      absxk = muDoubleScalarAbs(d);
      if (absxk > b_scale) {
        t = b_scale / absxk;
        Cxy_idx_0 = Cxy_idx_0 * t * t + 1.0;
        b_scale = absxk;
      } else {
        t = absxk / b_scale;
        Cxy_idx_0 += t * t;
      }
      Cxy_idx_0 = b_scale * muDoubleScalarSqrt(Cxy_idx_0);
      x1_idx_0 /= Cxy_idx_0;
      x1_idx_1 /= Cxy_idx_0;
      x1_idx_2 = d / Cxy_idx_0;
    }
    /*  f(a)=f(a)+[dEdu1;dEdv1;dEdw1]; */
    scale = k_pen * Cxy_idx_0;
    i = (int32_T)slaves[(int32_T)old_constraints_data[i_cnstr] - 1];
    d = dofs[i - 1];
    if (d != (int32_T)muDoubleScalarFloor(d)) {
      emlrtIntegerCheckR2012b(d, &i_emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)d < 1) || ((int32_T)d > 729)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)d, 1, 729, &i_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    L0 = dofs[i + 242];
    if (L0 != (int32_T)muDoubleScalarFloor(L0)) {
      emlrtIntegerCheckR2012b(L0, &i_emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)L0 < 1) || ((int32_T)L0 > 729)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)L0, 1, 729, &i_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    x2_idx_1 = f[(int32_T)L0 - 1] + scale * x1_idx_1;
    k = dofs[i + 485];
    if (k != (int32_T)muDoubleScalarFloor(k)) {
      emlrtIntegerCheckR2012b(k, &i_emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)k < 1) || ((int32_T)k > 729)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)k, 1, 729, &i_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    x2_idx_2 = f[(int32_T)k - 1] + scale * x1_idx_2;
    if (d != (int32_T)muDoubleScalarFloor(d)) {
      emlrtIntegerCheckR2012b(d, &h_emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)d < 1) || ((int32_T)d > 729)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)d, 1, 729, &h_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    f[(int32_T)d - 1] += scale * x1_idx_0;
    if (L0 != (int32_T)muDoubleScalarFloor(L0)) {
      emlrtIntegerCheckR2012b(L0, &h_emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)L0 < 1) || ((int32_T)L0 > 729)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)L0, 1, 729, &h_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    f[(int32_T)L0 - 1] = x2_idx_1;
    if (k != (int32_T)muDoubleScalarFloor(k)) {
      emlrtIntegerCheckR2012b(k, &h_emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)k < 1) || ((int32_T)k > 729)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)k, 1, 729, &h_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    f[(int32_T)k - 1] = x2_idx_2;
    st.site = &bb_emlrtRSI;
    b_st.site = &cb_emlrtRSI;
    *m += 0.5 * k_pen * (Cxy_idx_0 * Cxy_idx_0);
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
}

void fullBFGStoCompileToMex(const emlrtStack *sp, real_T u[729],
                            const real_T X[729], const real_T conn[740],
                            const boolean_T is_free[729],
                            const boolean_T actives[222], real_T mu,
                            real_T ratio, real_T k_pen, real_T Cx, real_T Cy,
                            real_T Cz, real_T R, const real_T slaves[222],
                            const real_T dofs[729], const real_T s[444],
                            const real_T sh[444], real_T *m_new, real_T *iter)
{
  emlrtStack b_st;
  emlrtStack st;
  emxArray_real_T *K_new_inv;
  emxArray_real_T *b_y;
  emxArray_real_T *c_y;
  emxArray_real_T *delta_f;
  emxArray_real_T *delta_u;
  emxArray_real_T *product1;
  emxArray_real_T *product2;
  emxArray_real_T *y;
  real_T f[729];
  real_T f_2_data[729];
  real_T f_3_data[729];
  real_T f_new_data[729];
  real_T f_old_data[729];
  real_T free_ind_data[729];
  real_T ux[729];
  real_T old_constraints_data[222];
  real_T alpha1;
  real_T alpha2;
  real_T alpha3;
  real_T m_2;
  real_T m_3;
  real_T scalar2;
  int32_T tmp_data[222];
  int32_T iv[2];
  int32_T old_constraints_size[2];
  int32_T tmp_size[2];
  int32_T y_size[2];
  int32_T exitg1;
  int32_T exitg2;
  int32_T f_old_size;
  int32_T i;
  int32_T signal1;
  int32_T signal2;
  int32_T trueCount;
  int16_T b_tmp_data[729];
  boolean_T b_actives[222];
  (void)mu;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  /* Here is a fully expanded version of the BFGS algorithm that solves the */
  /* mechanical subproblem with a specific list of constrained slave nodes */
  /*  old_constraints = slaves(actives==true); */
  memcpy(&b_actives[0], &actives[0], 222U * sizeof(boolean_T));
  eml_find(b_actives, tmp_data, tmp_size);
  old_constraints_size[0] = 1;
  old_constraints_size[1] = tmp_size[1];
  signal1 = tmp_size[1];
  for (i = 0; i < signal1; i++) {
    old_constraints_data[i] = tmp_data[i];
  }
  trueCount = 0;
  signal1 = 0;
  for (signal2 = 0; signal2 < 729; signal2++) {
    if (is_free[signal2]) {
      trueCount++;
      free_ind_data[signal1] = dofs[signal2];
      signal1++;
    }
  }
  st.site = &emlrtRSI;
  mf_with_constr(&st, u, X, conn, free_ind_data, trueCount,
                 old_constraints_data, old_constraints_size, ratio, k_pen, Cx,
                 Cy, Cz, R, slaves, dofs, s, sh, m_new, f);
  for (i = 0; i < trueCount; i++) {
    scalar2 = free_ind_data[i];
    if (scalar2 != (int32_T)muDoubleScalarFloor(scalar2)) {
      emlrtIntegerCheckR2012b(scalar2, &l_emlrtDCI, (emlrtCTX)sp);
    }
    if (((int32_T)scalar2 < 1) || ((int32_T)scalar2 > 729)) {
      emlrtDynamicBoundsCheckR2012b((int32_T)scalar2, 1, 729, &l_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    f_2_data[i] = f[(int32_T)scalar2 - 1];
  }
  emxInit_real_T(sp, &K_new_inv, 2, &v_emlrtRTEI, true);
  emxInit_real_T(sp, &delta_u, 2, &q_emlrtRTEI, true);
  st.site = &b_emlrtRSI;
  eye(&st, trueCount, delta_u);
  st.site = &b_emlrtRSI;
  inv(&st, delta_u, K_new_inv);
  if (0 <= trueCount - 1) {
    memset(&f_new_data[0], 0, trueCount * sizeof(real_T));
  }
  *iter = 0.0;
  i = delta_u->size[0] * delta_u->size[1];
  delta_u->size[0] = trueCount;
  delta_u->size[1] = trueCount;
  emxEnsureCapacity_real_T(sp, delta_u, i, &q_emlrtRTEI);
  signal1 = trueCount * trueCount;
  for (i = 0; i < signal1; i++) {
    delta_u->data[i] = 0.0;
  }
  m_2 = 0.0;
  emxInit_real_T(sp, &delta_f, 2, &w_emlrtRTEI, true);
  emxInit_real_T(sp, &product1, 2, &x_emlrtRTEI, true);
  emxInit_real_T(sp, &product2, 2, &y_emlrtRTEI, true);
  emxInit_real_T(sp, &y, 2, &ab_emlrtRTEI, true);
  emxInit_real_T(sp, &b_y, 2, &bb_emlrtRTEI, true);
  emxInit_real_T(sp, &c_y, 2, &bb_emlrtRTEI, true);
  do {
    exitg1 = 0;
    for (i = 0; i < trueCount; i++) {
      f_old_data[i] = f_2_data[i] - f_new_data[i];
    }
    if ((c_norm(f_old_data, trueCount) > 0.0) &&
        (c_norm(f_2_data, trueCount) > 0.0)) {
      (*iter)++;
      f_old_size = trueCount;
      if (0 <= trueCount - 1) {
        memcpy(&f_old_data[0], &f_new_data[0], trueCount * sizeof(real_T));
      }
      if (0 <= trueCount - 1) {
        memcpy(&f_new_data[0], &f_2_data[0], trueCount * sizeof(real_T));
      }
      i = delta_f->size[0] * delta_f->size[1];
      delta_f->size[0] = trueCount;
      delta_f->size[1] = 1;
      emxEnsureCapacity_real_T(sp, delta_f, i, &r_emlrtRTEI);
      for (i = 0; i < trueCount; i++) {
        delta_f->data[i] = f_2_data[i] - f_old_data[i];
      }
      if (*iter == 1.0) {
        st.site = &c_emlrtRSI;
        i = delta_u->size[0] * delta_u->size[1];
        delta_u->size[0] = K_new_inv->size[0];
        delta_u->size[1] = K_new_inv->size[1];
        emxEnsureCapacity_real_T(&st, delta_u, i, &s_emlrtRTEI);
        signal1 = K_new_inv->size[0] * K_new_inv->size[1];
        for (i = 0; i < signal1; i++) {
          delta_u->data[i] = -K_new_inv->data[i];
        }
        b_st.site = &nd_emlrtRSI;
        dynamic_size_checks(&b_st, delta_u, trueCount, delta_u->size[1],
                            trueCount);
        mtimes(delta_u, f_2_data, trueCount, f_old_data, &f_old_size);
      } else {
        /*  K_new_inv=K_old_inv+(delta_u'*delta_f+delta_f'*K_old_inv*delta_f)*(delta_u*delta_u')/(delta_u'*delta_f)^2-(K_old_inv*delta_f*delta_u'+delta_u*delta_f'*K_old_inv)/(delta_u'*delta_f);
         */
        st.site = &d_emlrtRSI;
        b_st.site = &nd_emlrtRSI;
        b_dynamic_size_checks(&b_st, delta_u, delta_f, delta_u->size[0],
                              delta_f->size[0]);
        b_st.site = &md_emlrtRSI;
        b_mtimes(&b_st, delta_u, delta_f, y);
        st.site = &d_emlrtRSI;
        b_st.site = &nd_emlrtRSI;
        b_dynamic_size_checks(&b_st, delta_f, K_new_inv, delta_f->size[0],
                              K_new_inv->size[0]);
        b_st.site = &md_emlrtRSI;
        b_mtimes(&b_st, delta_f, K_new_inv, b_y);
        st.site = &d_emlrtRSI;
        b_st.site = &nd_emlrtRSI;
        b_dynamic_size_checks(&b_st, b_y, delta_f, b_y->size[1],
                              delta_f->size[0]);
        b_st.site = &md_emlrtRSI;
        c_mtimes(&b_st, b_y, delta_f, c_y);
        tmp_size[0] = (*(int32_T(*)[2])y->size)[0];
        tmp_size[1] = (*(int32_T(*)[2])y->size)[1];
        iv[0] = (*(int32_T(*)[2])c_y->size)[0];
        iv[1] = (*(int32_T(*)[2])c_y->size)[1];
        emlrtSizeEqCheckNDR2012b(&tmp_size[0], &iv[0], &emlrtECI, (emlrtCTX)sp);
        if (1 != y->size[0]) {
          emlrtSubAssignSizeCheck1dR2017a(1, y->size[0], &b_emlrtECI,
                                          (emlrtCTX)sp);
        }
        st.site = &e_emlrtRSI;
        b_st.site = &nd_emlrtRSI;
        b_dynamic_size_checks(&b_st, delta_u, delta_f, delta_u->size[0],
                              delta_f->size[0]);
        b_st.site = &md_emlrtRSI;
        b_mtimes(&b_st, delta_u, delta_f, b_y);
        if (1 != b_y->size[0]) {
          emlrtSubAssignSizeCheck1dR2017a(1, b_y->size[0], &c_emlrtECI,
                                          (emlrtCTX)sp);
        }
        scalar2 = b_y->data[0];
        /*  scalar1=double(scalar1); */
        /*  scalar2=double(scalar2); */
        /*  K_new_inv=K_old_inv+scalar1*(delta_u*delta_u')/scalar2^2-(K_old_inv*delta_f*delta_u'+delta_u*delta_f'*K_old_inv)/scalar2;
         */
        /*  Step 1: Calculate matrix products separately */
        st.site = &f_emlrtRSI;
        b_st.site = &nd_emlrtRSI;
        b_dynamic_size_checks(&b_st, delta_u, delta_u, delta_u->size[1],
                              delta_u->size[1]);
        b_st.site = &md_emlrtRSI;
        d_mtimes(&b_st, delta_u, delta_u, product1);
        st.site = &g_emlrtRSI;
        b_st.site = &nd_emlrtRSI;
        b_dynamic_size_checks(&b_st, K_new_inv, delta_f, K_new_inv->size[1],
                              delta_f->size[0]);
        b_st.site = &md_emlrtRSI;
        c_mtimes(&b_st, K_new_inv, delta_f, b_y);
        st.site = &g_emlrtRSI;
        b_st.site = &nd_emlrtRSI;
        b_dynamic_size_checks(&b_st, b_y, delta_u, b_y->size[1],
                              delta_u->size[1]);
        b_st.site = &md_emlrtRSI;
        d_mtimes(&b_st, b_y, delta_u, product2);
        st.site = &h_emlrtRSI;
        b_st.site = &nd_emlrtRSI;
        b_dynamic_size_checks(&b_st, delta_u, delta_f, delta_u->size[1], 1);
        b_st.site = &md_emlrtRSI;
        d_mtimes(&b_st, delta_u, delta_f, b_y);
        st.site = &h_emlrtRSI;
        b_st.site = &nd_emlrtRSI;
        b_dynamic_size_checks(&b_st, b_y, K_new_inv, b_y->size[1],
                              K_new_inv->size[0]);
        b_st.site = &md_emlrtRSI;
        c_mtimes(&b_st, b_y, K_new_inv, delta_u);
        /*  Step 2: Perform operations with scalars */
        st.site = &i_emlrtRSI;
        alpha3 = (y->data[0] + c_y->data[0]) / (scalar2 * scalar2);
        signal1 = product1->size[0] * product1->size[1];
        for (i = 0; i < signal1; i++) {
          product1->data[i] *= alpha3;
        }
        tmp_size[0] = (*(int32_T(*)[2])product2->size)[0];
        tmp_size[1] = (*(int32_T(*)[2])product2->size)[1];
        iv[0] = (*(int32_T(*)[2])delta_u->size)[0];
        iv[1] = (*(int32_T(*)[2])delta_u->size)[1];
        emlrtSizeEqCheckNDR2012b(&tmp_size[0], &iv[0], &d_emlrtECI,
                                 (emlrtCTX)sp);
        signal1 = product2->size[0] * product2->size[1];
        for (i = 0; i < signal1; i++) {
          product2->data[i] = (product2->data[i] + delta_u->data[i]) / scalar2;
        }
        /*  Step 3: Combine the parts */
        tmp_size[0] = (*(int32_T(*)[2])K_new_inv->size)[0];
        tmp_size[1] = (*(int32_T(*)[2])K_new_inv->size)[1];
        iv[0] = (*(int32_T(*)[2])product1->size)[0];
        iv[1] = (*(int32_T(*)[2])product1->size)[1];
        emlrtSizeEqCheckNDR2012b(&tmp_size[0], &iv[0], &e_emlrtECI,
                                 (emlrtCTX)sp);
        tmp_size[0] = (*(int32_T(*)[2])K_new_inv->size)[0];
        tmp_size[1] = (*(int32_T(*)[2])K_new_inv->size)[1];
        iv[0] = (*(int32_T(*)[2])product2->size)[0];
        iv[1] = (*(int32_T(*)[2])product2->size)[1];
        emlrtSizeEqCheckNDR2012b(&tmp_size[0], &iv[0], &e_emlrtECI,
                                 (emlrtCTX)sp);
        signal1 = K_new_inv->size[0] * K_new_inv->size[1];
        for (i = 0; i < signal1; i++) {
          K_new_inv->data[i] =
              (K_new_inv->data[i] + product1->data[i]) - product2->data[i];
        }
        st.site = &j_emlrtRSI;
        i = delta_u->size[0] * delta_u->size[1];
        delta_u->size[0] = K_new_inv->size[0];
        delta_u->size[1] = K_new_inv->size[1];
        emxEnsureCapacity_real_T(&st, delta_u, i, &u_emlrtRTEI);
        signal1 = K_new_inv->size[0] * K_new_inv->size[1];
        for (i = 0; i < signal1; i++) {
          delta_u->data[i] = -K_new_inv->data[i];
        }
        b_st.site = &nd_emlrtRSI;
        dynamic_size_checks(&b_st, delta_u, trueCount, delta_u->size[1],
                            trueCount);
        mtimes(delta_u, f_2_data, trueCount, f_old_data, &f_old_size);
      }
      alpha3 = 1.0;
      memcpy(&ux[0], &u[0], 729U * sizeof(real_T));
      for (i = 0; i < trueCount; i++) {
        b_tmp_data[i] = (int16_T)free_ind_data[i];
      }
      if (trueCount != f_old_size) {
        emlrtSizeEqCheck1DR2012b(trueCount, f_old_size, &f_emlrtECI,
                                 (emlrtCTX)sp);
      }
      for (i = 0; i < trueCount; i++) {
        signal1 = b_tmp_data[i] - 1;
        ux[signal1] = u[signal1] + f_old_data[i];
      }
      st.site = &k_emlrtRSI;
      mf_with_constr(&st, ux, X, conn, free_ind_data, trueCount,
                     old_constraints_data, old_constraints_size, ratio, k_pen,
                     Cx, Cy, Cz, R, slaves, dofs, s, sh, &m_3, f);
      for (i = 0; i < trueCount; i++) {
        f_3_data[i] = f[(int32_T)free_ind_data[i] - 1];
      }
      signal1 = 0;
      y_size[0] = 1;
      y_size[1] = f_old_size;
      for (i = 0; i < f_old_size; i++) {
        f[i] = 0.0001 * f_old_data[i];
      }
      st.site = &l_emlrtRSI;
      b_st.site = &nd_emlrtRSI;
      c_dynamic_size_checks(&b_st, y_size, trueCount, f_old_size, trueCount);
      if (m_3 <= *m_new + e_mtimes(f, y_size, f_2_data)) {
        st.site = &l_emlrtRSI;
        b_st.site = &nd_emlrtRSI;
        d_dynamic_size_checks(&b_st, f_old_size, trueCount, f_old_size,
                              trueCount);
        st.site = &l_emlrtRSI;
        b_st.site = &nd_emlrtRSI;
        d_dynamic_size_checks(&b_st, f_old_size, trueCount, f_old_size,
                              trueCount);
        if (muDoubleScalarAbs(f_mtimes(f_old_data, f_old_size, f_3_data)) <=
            0.9 *
                muDoubleScalarAbs(f_mtimes(f_old_data, f_old_size, f_2_data))) {
          signal1 = 1;
        }
      }
      do {
        exitg2 = 0;
        scalar2 = 0.0001 * alpha3;
        y_size[0] = 1;
        y_size[1] = f_old_size;
        for (i = 0; i < f_old_size; i++) {
          f[i] = scalar2 * f_old_data[i];
        }
        st.site = &m_emlrtRSI;
        b_st.site = &nd_emlrtRSI;
        c_dynamic_size_checks(&b_st, y_size, trueCount, f_old_size, trueCount);
        if (m_3 < *m_new + e_mtimes(f, y_size, f_2_data)) {
          st.site = &m_emlrtRSI;
          b_st.site = &nd_emlrtRSI;
          d_dynamic_size_checks(&b_st, f_old_size, trueCount, f_old_size,
                                trueCount);
          y_size[0] = 1;
          y_size[1] = f_old_size;
          for (i = 0; i < f_old_size; i++) {
            f[i] = -0.9 * f_old_data[i];
          }
          st.site = &m_emlrtRSI;
          b_st.site = &nd_emlrtRSI;
          c_dynamic_size_checks(&b_st, y_size, trueCount, f_old_size,
                                trueCount);
          if ((f_mtimes(f_old_data, f_old_size, f_3_data) <
               e_mtimes(f, y_size, f_2_data)) &&
              (signal1 == 0)) {
            alpha3 /= 0.5;
            memcpy(&ux[0], &u[0], 729U * sizeof(real_T));
            for (i = 0; i < f_old_size; i++) {
              f[i] = alpha3 * f_old_data[i];
            }
            if (trueCount != f_old_size) {
              emlrtSizeEqCheck1DR2012b(trueCount, f_old_size, &g_emlrtECI,
                                       (emlrtCTX)sp);
            }
            for (i = 0; i < trueCount; i++) {
              ux[b_tmp_data[i] - 1] = u[(int32_T)free_ind_data[i] - 1] + f[i];
            }
            st.site = &n_emlrtRSI;
            mf_with_constr(&st, ux, X, conn, free_ind_data, trueCount,
                           old_constraints_data, old_constraints_size, ratio,
                           k_pen, Cx, Cy, Cz, R, slaves, dofs, s, sh, &m_3, f);
            for (i = 0; i < trueCount; i++) {
              f_3_data[i] = f[(int32_T)free_ind_data[i] - 1];
            }
            scalar2 = 0.0001 * alpha3;
            y_size[0] = 1;
            y_size[1] = f_old_size;
            for (i = 0; i < f_old_size; i++) {
              f[i] = scalar2 * f_old_data[i];
            }
            st.site = &o_emlrtRSI;
            b_st.site = &nd_emlrtRSI;
            c_dynamic_size_checks(&b_st, y_size, trueCount, f_old_size,
                                  trueCount);
            if (m_3 <= *m_new + e_mtimes(f, y_size, f_2_data)) {
              st.site = &o_emlrtRSI;
              b_st.site = &nd_emlrtRSI;
              d_dynamic_size_checks(&b_st, f_old_size, trueCount, f_old_size,
                                    trueCount);
              st.site = &o_emlrtRSI;
              b_st.site = &nd_emlrtRSI;
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
              emlrtBreakCheckR2012b((emlrtCTX)sp);
            }
          } else {
            exitg2 = 1;
          }
        } else {
          exitg2 = 1;
        }
      } while (exitg2 == 0);
      if (signal1 == 0) {
        alpha1 = 0.0;
        alpha2 = alpha3 / 2.0;
        memcpy(&ux[0], &u[0], 729U * sizeof(real_T));
        for (i = 0; i < f_old_size; i++) {
          f[i] = alpha2 * f_old_data[i];
        }
        if (trueCount != f_old_size) {
          emlrtSizeEqCheck1DR2012b(trueCount, f_old_size, &h_emlrtECI,
                                   (emlrtCTX)sp);
        }
        for (i = 0; i < trueCount; i++) {
          ux[b_tmp_data[i] - 1] = u[(int32_T)free_ind_data[i] - 1] + f[i];
        }
        st.site = &p_emlrtRSI;
        mf_with_constr(&st, ux, X, conn, free_ind_data, trueCount,
                       old_constraints_data, old_constraints_size, ratio, k_pen,
                       Cx, Cy, Cz, R, slaves, dofs, s, sh, &m_2, f);
        for (i = 0; i < trueCount; i++) {
          f_2_data[i] = f[(int32_T)free_ind_data[i] - 1];
        }
        signal2 = 0;
        while (signal2 == 0) {
          if (alpha3 - alpha1 < 1.0E-15) {
            signal2 = 1;
            m_2 = *m_new;
            if (0 <= trueCount - 1) {
              memcpy(&f_2_data[0], &f_new_data[0], trueCount * sizeof(real_T));
            }
          } else {
            scalar2 = 0.0001 * alpha2;
            y_size[0] = 1;
            y_size[1] = f_old_size;
            for (i = 0; i < f_old_size; i++) {
              f[i] = scalar2 * f_old_data[i];
            }
            st.site = &q_emlrtRSI;
            b_st.site = &nd_emlrtRSI;
            c_dynamic_size_checks(&b_st, y_size, trueCount, f_old_size,
                                  trueCount);
            if (m_2 > *m_new + e_mtimes(f, y_size, f_new_data)) {
              alpha3 = alpha2;
              m_3 = m_2;
              if (0 <= trueCount - 1) {
                memcpy(&f_3_data[0], &f_2_data[0], trueCount * sizeof(real_T));
              }
              alpha2 = 0.5 * (alpha1 + alpha2);
              memcpy(&ux[0], &u[0], 729U * sizeof(real_T));
              for (i = 0; i < f_old_size; i++) {
                f[i] = alpha2 * f_old_data[i];
              }
              if (trueCount != f_old_size) {
                emlrtSizeEqCheck1DR2012b(trueCount, f_old_size, &i_emlrtECI,
                                         (emlrtCTX)sp);
              }
              for (i = 0; i < trueCount; i++) {
                ux[b_tmp_data[i] - 1] = u[(int32_T)free_ind_data[i] - 1] + f[i];
              }
              st.site = &r_emlrtRSI;
              mf_with_constr(&st, ux, X, conn, free_ind_data, trueCount,
                             old_constraints_data, old_constraints_size, ratio,
                             k_pen, Cx, Cy, Cz, R, slaves, dofs, s, sh, &m_2,
                             f);
              for (i = 0; i < trueCount; i++) {
                f_2_data[i] = f[(int32_T)free_ind_data[i] - 1];
              }
            } else {
              st.site = &s_emlrtRSI;
              b_st.site = &nd_emlrtRSI;
              d_dynamic_size_checks(&b_st, f_old_size, trueCount, f_old_size,
                                    trueCount);
              scalar2 = f_mtimes(f_old_data, f_old_size, f_2_data);
              y_size[0] = 1;
              y_size[1] = f_old_size;
              for (i = 0; i < f_old_size; i++) {
                f[i] = 0.9 * f_old_data[i];
              }
              st.site = &s_emlrtRSI;
              b_st.site = &nd_emlrtRSI;
              c_dynamic_size_checks(&b_st, y_size, trueCount, f_old_size,
                                    trueCount);
              if (scalar2 < e_mtimes(f, y_size, f_new_data)) {
                alpha1 = alpha2;
                alpha2 = 0.5 * (alpha2 + alpha3);
                memcpy(&ux[0], &u[0], 729U * sizeof(real_T));
                for (i = 0; i < f_old_size; i++) {
                  f[i] = alpha2 * f_old_data[i];
                }
                if (trueCount != f_old_size) {
                  emlrtSizeEqCheck1DR2012b(trueCount, f_old_size, &j_emlrtECI,
                                           (emlrtCTX)sp);
                }
                for (i = 0; i < trueCount; i++) {
                  ux[b_tmp_data[i] - 1] =
                      u[(int32_T)free_ind_data[i] - 1] + f[i];
                }
                st.site = &t_emlrtRSI;
                mf_with_constr(&st, ux, X, conn, free_ind_data, trueCount,
                               old_constraints_data, old_constraints_size,
                               ratio, k_pen, Cx, Cy, Cz, R, slaves, dofs, s, sh,
                               &m_2, f);
                for (i = 0; i < trueCount; i++) {
                  f_2_data[i] = f[(int32_T)free_ind_data[i] - 1];
                }
              } else {
                st.site = &u_emlrtRSI;
                b_st.site = &nd_emlrtRSI;
                d_dynamic_size_checks(&b_st, f_old_size, trueCount, f_old_size,
                                      trueCount);
                y_size[0] = 1;
                y_size[1] = f_old_size;
                for (i = 0; i < f_old_size; i++) {
                  f[i] = -0.9 * f_old_data[i];
                }
                st.site = &u_emlrtRSI;
                b_st.site = &nd_emlrtRSI;
                c_dynamic_size_checks(&b_st, y_size, trueCount, f_old_size,
                                      trueCount);
                if (scalar2 > e_mtimes(f, y_size, f_new_data)) {
                  alpha3 = alpha2;
                  m_3 = m_2;
                  if (0 <= trueCount - 1) {
                    memcpy(&f_3_data[0], &f_2_data[0],
                           trueCount * sizeof(real_T));
                  }
                  alpha2 = 0.5 * (alpha1 + alpha2);
                  memcpy(&ux[0], &u[0], 729U * sizeof(real_T));
                  for (i = 0; i < f_old_size; i++) {
                    f[i] = alpha2 * f_old_data[i];
                  }
                  if (trueCount != f_old_size) {
                    emlrtSizeEqCheck1DR2012b(trueCount, f_old_size, &k_emlrtECI,
                                             (emlrtCTX)sp);
                  }
                  for (i = 0; i < trueCount; i++) {
                    ux[b_tmp_data[i] - 1] =
                        u[(int32_T)free_ind_data[i] - 1] + f[i];
                  }
                  st.site = &v_emlrtRSI;
                  mf_with_constr(&st, ux, X, conn, free_ind_data, trueCount,
                                 old_constraints_data, old_constraints_size,
                                 ratio, k_pen, Cx, Cy, Cz, R, slaves, dofs, s,
                                 sh, &m_2, f);
                  for (i = 0; i < trueCount; i++) {
                    f_2_data[i] = f[(int32_T)free_ind_data[i] - 1];
                  }
                } else {
                  signal2 = 1;
                }
              }
            }
          }
          if (*emlrtBreakCheckR2012bFlagVar != 0) {
            emlrtBreakCheckR2012b((emlrtCTX)sp);
          }
        }
      }
      i = delta_u->size[0] * delta_u->size[1];
      delta_u->size[0] = trueCount;
      delta_u->size[1] = 1;
      emxEnsureCapacity_real_T(sp, delta_u, i, &t_emlrtRTEI);
      for (i = 0; i < trueCount; i++) {
        scalar2 = free_ind_data[i];
        delta_u->data[i] = ux[(int32_T)scalar2 - 1] - u[(int32_T)scalar2 - 1];
      }
      memcpy(&u[0], &ux[0], 729U * sizeof(real_T));
      if (signal1 == 1) {
        *m_new = m_3;
        if (0 <= trueCount - 1) {
          memcpy(&f_2_data[0], &f_3_data[0], trueCount * sizeof(real_T));
        }
      } else {
        *m_new = m_2;
      }
      if (*emlrtBreakCheckR2012bFlagVar != 0) {
        emlrtBreakCheckR2012b((emlrtCTX)sp);
      }
    } else {
      exitg1 = 1;
    }
  } while (exitg1 == 0);
  emxFree_real_T(&c_y);
  emxFree_real_T(&b_y);
  emxFree_real_T(&y);
  emxFree_real_T(&product2);
  emxFree_real_T(&product1);
  emxFree_real_T(&delta_f);
  emxFree_real_T(&delta_u);
  emxFree_real_T(&K_new_inv);
  /*  sprintf('fullBFGS: %s',int2str(iter)) */
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

/* End of code generation (fullBFGStoCompileToMex.c) */

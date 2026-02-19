"""
ContactPotato_NGSolve.py — NGSolve hyperelastic contact simulation
==================================================================
Single self-contained script simulating a hyperelastic block sliding over a
rigid "potato" body.  ALL FE computations use the NGSolve API (mesh, FE space,
hyperelastic energy via Variation, Dirichlet BCs).  Contact detection reuses
the existing Gregory-patch / trust-region projection backend.

Configuration is via variables in the "Configuration" section below.

Solvers
-------
- **newton**: Full Newton with direct UMFPACK solve and Armijo linesearch.
  Uses the material tangent from NGSolve plus contact Hessian
  kn * (n ⊗ n + g * dn/dx_s) when full_hessian=True.  Contact is
  evaluated dynamically at each iteration with cached projections
  (reuse threshold = contact_tol_reuse) for energy consistency.
- **newton-cg**: Newton-CG via scipy.optimize.minimize with Hessian-vector
  product (hessp).  Uses CG to approximately solve each Newton system.
  Supports full_hessian for curvature term without matrix singularity risk.
- **trust-constr**: Trust-region constrained minimizer (unconstrained mode).
  Uses hessp for Hessian-vector products.
- **lbfgsb**: L-BFGS-B quasi-Newton via scipy.optimize.minimize.  No Hessian
  needed; uses only gradient information with limited-memory BFGS updates.
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1.  IMPORTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
import os, sys, pickle
from contextlib import nullcontext
from datetime import datetime
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from ngsolve import *
from ngsolve.meshes import MakeStructured3DMesh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyClasses import gregory_patch_backend as gb
from PyClasses._contact_tr_multi_helpers import project_points_tr_multi_batch

# ── Configuration ─────────────────────────────────────────────────────────────
# Mesh
n           = 15                # mesh density (n x n x n hex elements)

# Material (compressible neo-Hookean)
E_val       = 0.05          # Young's modulus
nu_val      = 0.3           # Poisson ratio

# Contact
kn_factor   = 20.0          # kn = kn_factor * E / h  (Wriggers 2006)
contact_tol_reuse = 1e-5 #np.inf #0.05 #np.inf  # per-node displacement threshold for projection reuse
                            # inf  = lock projections during solve (most robust)
                            # finite (e.g. 1e-5) = re-project nodes that move beyond this

# Solver: "newton", "newton-cg", "trust-constr", "lbfgsb"
solver      = "newton"
nsteps      = 100           # number of load steps
max_iter    = 200            # max iterations per step
gtol        = 1e-12          # gradient/residual tolerance
max_cutback = 5             # max bisection levels on non-convergence (2^5 = 32x refinement)

# Hessian
full_hessian = False         # include curvature term dn/dx_s in contact Hessian

# Output
compare     = False         # write comparison outputs (u arrays, reactions CSV)
plot        = 1            # VTK export every N steps (0 = off)
vtk_fields  = "minimal"    # "minimal" | "standard" | "full" (see Section 7)

# Performance
taskmanager = "auto"       # True/False/"auto" — auto enables for n >= 30 (27k+ elements)
realcompile = "auto"       # True/False/"auto" — auto enables for nsteps >= 20
profile     = True        # built-in per-operation timing (prints breakdown per step)
linear_solver = "pardiso"  # "umfpack" | "pardiso" | "mumps" — direct solver for Newton

# Plasticity (J2 von Mises, multiplicative decomposition F = Fe·Fp)
plastic       = False                  # enable elastoplastic constitutive model
plastic_param = [0.01, 0.05, 1.0]     # [My0, H_hard, m_hard]
# My0     = initial yield stress
# H_hard  = hardening modulus
# m_hard  = hardening exponent (1.0 = linear hardening)
consistent_tangent = False   # FD-based algorithmic tangent correction for yielding GPs
                              # Marginal benefit with contact; can destabilize at large steps

# AI-enhanced contact
nn_contact       = True   # enable NN for contact detection
nn_contact_mode  = "multitask"  # "multitask" (Phase 1: NN broad + C++ refine) or "neural_pull" (Phase 2: pure NN)
nn_contact_model = "v1"     # multitask variant: "v1", "v2", "v3"
nn_contact_device = "cuda"  # "cuda" or "cpu"
nn_contact_topk  = 3        # multitask only: K candidates per node
# ──────────────────────────────────────────────────────────────────────────────

# ── Profiling instrumentation ────────────────────────────────────────────────
class PerfCounters:
    """Lightweight per-operation timing accumulator with tree display.

    Timer nesting hierarchy (children are timed inside their parent):
      step_total
      ├── apply
      ├── contact_eval
      │   ├── contact_nn_project
      │   ├── contact_full_tr
      │   └── contact_cached
      ├── contact_force_asm
      ├── assemble_lin
      ├── plastic_tangent
      ├── contact_hess_asm
      ├── linear_solve
      ├── return_mapping
      ├── linesearch
      │   ├── ls_energy_mat
      │   └── ls_energy_contact
      └── vtk_output
    """

    # Nesting: parent → list of children whose time is included in parent
    _CHILDREN = {
        "step_total": [
            "apply", "contact_eval", "contact_force_asm", "assemble_lin",
            "plastic_tangent", "contact_hess_asm", "linear_solve",
            "return_mapping", "linesearch", "vtk_output",
        ],
        "contact_eval": ["contact_nn_project", "contact_full_tr",
                         "contact_cached", "contact_neural_pull"],
        "linesearch": ["ls_energy_mat", "ls_energy_contact"],
    }
    # All children (flattened) — these are nested, not top-level
    _NESTED = {c for children in _CHILDREN.values() for c in children}

    def __init__(self):
        self.data = {}
        self.step_data = {}

    def reset_step(self):
        self.step_data = {}

    def record(self, name, duration):
        self.data.setdefault(name, []).append(duration)
        self.step_data.setdefault(name, []).append(duration)

    def _get_totals(self, source):
        """Return {name: (total_time, call_count)} from source dict."""
        return {name: (sum(times), len(times)) for name, times in source.items()}

    def _format_tree(self, totals, wall_time, indent="  "):
        """Format timers as a tree, using wall_time for percentages."""
        lines = []

        def _fmt_line(prefix, name, total, count, is_other=False):
            avg_ms = total / count * 1000 if count > 0 else 0
            pct = 100.0 * total / wall_time if wall_time > 0 else 0
            label = f"{prefix}{name}"
            if is_other:
                lines.append(f"{label:<40s}  {total:8.2f}s  ({pct:5.1f}%)")
            else:
                lines.append(
                    f"{label:<40s}  {total:8.2f}s  ({pct:5.1f}%)  "
                    f"[{count} calls, avg {avg_ms:.2f}ms]"
                )

        def _format_node(name, prefix, connector, child_prefix):
            if name not in totals:
                return
            total, count = totals[name]
            _fmt_line(prefix + connector, name, total, count)

            children = self._CHILDREN.get(name, [])
            present = [c for c in children if c in totals]
            if present:
                child_sum = sum(totals[c][0] for c in present)
                has_other = (total - child_sum) > 0.005
                for i, child in enumerate(present):
                    is_final = (i == len(present) - 1) and not has_other
                    conn = "└── " if is_final else "├── "
                    ext = "    " if is_final else "│   "
                    _format_node(child, child_prefix, conn, child_prefix + ext)
                if has_other:
                    _fmt_line(child_prefix + "└── ", "(other)",
                              total - child_sum, 0, is_other=True)

        # Start with step_total if present, then any top-level extras
        if "step_total" in totals:
            _format_node("step_total", indent, "", indent)
        # Show any timers not in the tree (shouldn't happen, but safety net)
        all_in_tree = {"step_total"} | self._NESTED
        extras = {k for k in totals if k not in all_in_tree}
        for name in sorted(extras, key=lambda k: -totals[k][0]):
            total, count = totals[name]
            _fmt_line(indent, name, total, count)

        return lines

    def step_summary(self, step):
        if not self.step_data:
            return ""
        totals = self._get_totals(self.step_data)
        wall = totals.get("step_total", (0, 0))[0]
        lines = [f"  [PROFILE] Step {step}:"]
        lines.extend(self._format_tree(totals, wall))
        return "\n".join(lines)

    def final_summary(self):
        if not self.data:
            return ""
        totals = self._get_totals(self.data)
        wall = totals.get("step_total", (0, 0))[0]
        lines = [
            "\n" + "=" * 80,
            "  PROFILING SUMMARY (all steps)",
            "=" * 80,
        ]
        lines.extend(self._format_tree(totals, wall, indent="  "))
        lines.append("  " + "-" * 78)
        lines.append(f"  {'WALL TIME':40s}  {wall:8.2f}s")
        return "\n".join(lines)

perf = PerfCounters() if profile else None
# ─────────────────────────────────────────────────────────────────────────────

# Resolve "auto" performance settings based on problem size.
# TaskManager: thread dispatch overhead dominates for small meshes (~250 el/thread
# at n=20 with 32 threads).  Profiling shows 2-3x slowdown on Python-touching
# operations due to GIL contention.  Only beneficial for n>=30 (27k+ elements).
# realcompile: 5-10s JIT startup cost amortizes over 20+ steps.
if taskmanager == "auto":
    taskmanager = (n**3 >= 27000)
if realcompile == "auto":
    realcompile = (nsteps >= 20)

h_contact = 4.0 / n              # element edge length at contact surface ([-2,2]^3 block)
kn      = kn_factor * E_val / h_contact   # Wriggers (2006): kn ~ alpha * E / h

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d%H%M")
out_dir = f"OUTPUT_{timestamp}_NGSolve_ContactPotato_{n}"
os.makedirs(out_dir, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  NGSOLVE MESH + HYPERELASTIC ENERGY
# ══════════════════════════════════════════════════════════════════════════════
# Block: [-2,2]^3 translated by z+3.5  (same as ContactPotato_Ex1.py)
mesh = MakeStructured3DMesh(
    hexes=True, nx=n, ny=n, nz=n,
    mapping=lambda x, y, z: (-2 + 4*x, -2 + 4*y, -2 + 4*z + 3.5)
)

fes = VectorH1(mesh, order=1, dirichlet="top")
gfu = GridFunction(fes)
nv  = mesh.nv                       # number of vertices
ndof = fes.ndof                     # = 3 * nv for order=1

# --- Material parameters (compressible neo-Hookean) -----------------------
# W(F) = c10*(I1 - 3 - 2 ln J) + d1*(ln J)^2
#   c10 = E/(4(1+nu)),   d1 = E*nu/(2(1+nu)(1-2*nu))
c10 = E_val / (4 * (1 + nu_val))
d1  = E_val * nu_val / (2 * (1 + nu_val) * (1 - 2*nu_val))

# Symbolic energy (uses trial function u for Variation / Apply)
u_trial, v_test = fes.TnT()
F_sym  = Id(3) + Grad(u_trial)

# --- Plasticity: IntegrationRuleSpace for Fp at Gauss points -------------
if plastic:
    from ngsolve.comp import IntegrationRuleSpace
    My0, H_hard, m_hard = plastic_param

    # Integration rule space: order=1 → 2×2×2 = 8 GP per hex (matches reference code)
    fes_ir = IntegrationRuleSpace(mesh, order=1)
    irs_dx = dx(intrules=fes_ir.GetIntegrationRules())
    n_ip = fes_ir.ndof    # total integration points across all elements

    # Plastic deformation gradient Fp (3×3 per GP)
    fes_Fp = MatrixValued(fes_ir, dim=3)
    gf_Fp = GridFunction(fes_Fp)
    gf_Fp.Interpolate(Id(3))  # virgin material: Fp = I

    # F evaluator: reads deformation gradient at all GPs
    fes_F_ir = MatrixValued(fes_ir, dim=3)
    gf_F_ir = GridFunction(fes_F_ir)

    # History arrays (numpy, updated ONLY on successful load step convergence)
    _Fp_conv   = np.tile(np.eye(3).ravel(), n_ip)   # (n_ip*9,) flattened
    _epcum_conv = np.zeros(n_ip)                     # cumulative plastic strain per GP

    # Working arrays (recomputed by return mapping at each Newton iteration)
    _Fp_temp     = _Fp_conv.copy()
    _delta_epcum = np.zeros(n_ip)

    # Energy: W(Fe) where Fe = F · Fp^{-1}
    Fe_sym  = F_sym * Inv(gf_Fp)
    J_sym   = Det(Fe_sym)
    I1_sym  = Trace(Fe_sym.trans * Fe_sym)
    psi_sym = c10 * (I1_sym - 3 - 2*log(J_sym)) + d1 * log(J_sym)**2

    a_form = BilinearForm(fes)
    a_form += Variation(psi_sym.Compile(realcompile=realcompile, wait=True) * irs_dx)

    print(f"  Plasticity IRS: {n_ip} integration points "
          f"({n_ip // mesh.ne} per element)")

    # --- Consistent tangent infrastructure (element-level data) --------
    _gps_per_elem = n_ip // mesh.ne   # should be 8 for order=1 hex

    # Element connectivity: vertex numbers for each element
    _elem_verts = np.array(
        [[v.nr for v in el.vertices] for el in mesh.Elements(VOL)],
        dtype=np.int32)  # (n_elem, 8)

    # Shape function gradients at all GPs (constant in total Lagrangian).
    # For the uniform structured hex mesh, J = diag(h, h, h) for all elements,
    # so dN/dX = dN/dξ / h and detJ = h^3.  We still compute per-element
    # for generality (handles non-uniform meshes without code change).
    #
    # GP positions in [0,1]^3 reference element (2-point Gauss-Legendre per axis):
    _gp_lo = 0.5 - 0.5 / np.sqrt(3.0)   # ≈ 0.211325
    _gp_hi = 0.5 + 0.5 / np.sqrt(3.0)   # ≈ 0.788675
    _gp_1d = np.array([_gp_lo, _gp_hi])
    # All 8 GP positions: ordered as NGSolve IRS (ζ fastest, then η, then ξ — per intrule.cpp:3088)
    _gp_ref = np.array([[xi, eta, zeta]
                         for xi in _gp_1d for eta in _gp_1d for zeta in _gp_1d])
    _gp_weight = 0.125   # = 0.5^3 (product rule on [0,1]^3)

    def _hex8_dNdxi(xi, eta, zeta):
        """Parametric gradients of 8-node hex shape functions in [0,1]^3.

        Vertex ordering matches NGSolve MakeStructured3DMesh(hexes=True):
          0: (0,0,0)  1: (0,0,1)  2: (0,1,1)  3: (0,1,0)
          4: (1,0,0)  5: (1,0,1)  6: (1,1,1)  7: (1,1,0)
        Returns (8, 3) array: dN_I/dξ_j.
        """
        x, y, z = xi, eta, zeta
        return np.array([
            [-(1-y)*(1-z), -(1-x)*(1-z), -(1-x)*(1-y)],  # N0 = (1-x)(1-y)(1-z)
            [-(1-y)*z,     -(1-x)*z,       (1-x)*(1-y)],  # N1 = (1-x)(1-y)(z)
            [-(y)*z,        (1-x)*z,        (1-x)*(y)  ],  # N2 = (1-x)(y)(z)
            [-(y)*(1-z),    (1-x)*(1-z),   -(1-x)*(y)  ],  # N3 = (1-x)(y)(1-z)
            [ (1-y)*(1-z), -(x)*(1-z),     -(x)*(1-y)  ],  # N4 = (x)(1-y)(1-z)
            [ (1-y)*z,     -(x)*z,          (x)*(1-y)  ],  # N5 = (x)(1-y)(z)
            [ (y)*z,        (x)*z,          (x)*(y)    ],  # N6 = (x)(y)(z)
            [ (y)*(1-z),    (x)*(1-z),     -(x)*(y)    ],  # N7 = (x)(y)(1-z)
        ])

    # Precompute dN/dX and detJ at all (element, GP) pairs
    _all_dNdX = np.zeros((mesh.ne, _gps_per_elem, 8, 3))
    _all_detJ = np.zeros((mesh.ne, _gps_per_elem))
    _X_ref_local = np.array([list(mesh.vertices[i].point) for i in range(mesh.nv)])

    for e in range(mesh.ne):
        verts = _elem_verts[e]
        X_el = _X_ref_local[verts]   # (8, 3) physical coords of element vertices
        for ig in range(_gps_per_elem):
            xi, eta, zeta = _gp_ref[ig]
            dNdxi = _hex8_dNdxi(xi, eta, zeta)   # (8, 3)
            Jac = dNdxi.T @ X_el                  # (3, 3) = dX/dξ
            detJ = np.linalg.det(Jac)
            invJT = np.linalg.inv(Jac).T
            _all_dNdX[e, ig] = dNdxi @ invJT      # (8, 3) = dN/dX
            _all_detJ[e, ig] = detJ

    # Element CSR position map: built lazily after first AssembleLinearization
    _elem_csr_pos_plastic = None

    def _build_elem_csr_pos_plastic():
        """Build element-level CSR position map for consistent tangent assembly.

        Maps (element, local_dof_i, local_dof_j) → CSR value index.
        Local DOFs are in block order: [v0..v7, v0+nv..v7+nv, v0+2nv..v7+2nv].
        """
        global _elem_csr_pos_plastic
        mat = a_form.mat
        _, cols_fv, firsti_fv = mat.CSR()
        cols = np.array(cols_fv, copy=False)
        firsti = np.array(firsti_fv, copy=False)
        ne = mesh.ne
        _elem_csr_pos_plastic = np.full((ne, 24, 24), -1, dtype=np.int64)
        for e in range(ne):
            verts = _elem_verts[e]
            # 24 global DOFs in block order: [x-comp(8), y-comp(8), z-comp(8)]
            global_dofs = np.concatenate([verts, verts + nv, verts + 2*nv])
            for ii in range(24):
                row = int(global_dofs[ii])
                row_start = int(firsti[row])
                row_end = int(firsti[row + 1])
                row_cols = cols[row_start:row_end]
                for jj in range(24):
                    col = int(global_dofs[jj])
                    pos = np.searchsorted(row_cols, col)
                    if pos < len(row_cols) and row_cols[pos] == col:
                        _elem_csr_pos_plastic[e, ii, jj] = row_start + pos

    # Cache for F_flat during Newton iteration (set by return mapping, used by tangent)
    _F_flat_cache = None

else:
    J_sym  = Det(F_sym)
    I1_sym = Trace(F_sym.trans * F_sym)
    psi_sym = c10 * (I1_sym - 3 - 2*log(J_sym)) + d1 * log(J_sym)**2

    a_form = BilinearForm(fes)
    a_form += Variation(psi_sym.Compile(realcompile=realcompile, wait=True) * dx)

# --- Stress CFs for VTK output (built only if needed by vtk_fields) ------
# Only compile what's needed — matrix-valued stress tensors are expensive.
F_gfu   = Id(3) + Grad(gfu)

# For plasticity, compute elastic deformation gradient Fe for stress evaluation
if plastic:
    Fe_gfu  = F_gfu * Inv(gf_Fp)
    J_gfu   = Det(Fe_gfu)
    B_e     = Fe_gfu * Fe_gfu.trans   # elastic left Cauchy-Green
else:
    J_gfu   = Det(F_gfu)
    B_e     = F_gfu * F_gfu.trans     # B = F·Fᵀ (same as Be when Fp=I)
F_inv_T = Inv(F_gfu).trans

stress_1piola = stress_cauchy = stress_mandel = None
vm_cauchy = vm_mandel = None

if plot > 0:
    # Cauchy: σ = (1/J) [2c10(B_e - I) + 2d1·ln(J_e)·I]
    stress_cauchy = ((1/J_gfu) * (2*c10*(B_e - Id(3))
                     + 2*d1*log(J_gfu)*Id(3))).Compile()
    s_dev_cauchy = stress_cauchy - (1.0/3.0) * Trace(stress_cauchy) * Id(3)
    vm_cauchy = sqrt(1.5 * InnerProduct(s_dev_cauchy, s_dev_cauchy)).Compile()

    if vtk_fields == "full":
        # 1st Piola-Kirchhoff: P = Pe · Fp^{-T}  (= Pe when Fp=I)
        # Mandel: M = Fe^T · Pe  (symmetric for isotropic W)
        if plastic:
            Fe_inv_T = Inv(Fe_gfu).trans
            Pe_gfu = 2*c10*(Fe_gfu - Fe_inv_T) + 2*d1*log(J_gfu)*Fe_inv_T
            stress_1piola = (Pe_gfu * Inv(gf_Fp).trans).Compile()
            stress_mandel = (Fe_gfu.trans * Pe_gfu).Compile()
        else:
            stress_1piola = (2*c10*(F_gfu - F_inv_T)
                             + 2*d1*log(J_gfu)*F_inv_T).Compile()
            stress_mandel = (F_gfu.trans * stress_1piola).Compile()
        s_dev_mandel = stress_mandel - (1.0/3.0) * Trace(stress_mandel) * Id(3)
        vm_mandel = sqrt(1.5 * InnerProduct(s_dev_mandel, s_dev_mandel)).Compile()

# ══════════════════════════════════════════════════════════════════════════════
# 3.  POTATO + GREGORY PATCHES
# ══════════════════════════════════════════════════════════════════════════════
[ptt] = pickle.load(open("Dat/PotatoAssembly.dat", "rb"))
if hasattr(ptt, 'hexas') and not hasattr(ptt, 'elements'):
    ptt.elements = ptt.hexas
ptt.isRigid = True
n_ptt_nodes = len(ptt.X)
ndofs_ptt   = 3 * n_ptt_nodes

# Set up DoFs array expected by GregoryPatches internals (node→[dofx,dofy,dofz])
ptt.DoFs = np.array([[3*i, 3*i+1, 3*i+2] for i in range(n_ptt_nodes)])
ptt.surf.ComputeGrgPatches(np.zeros(ndofs_ptt), range(len(ptt.surf.nodes)))
patches = ptt.surf.patches

# Precompute bounding-sphere data
xm_matrix   = np.array([p.BS.x for p in patches], dtype=np.float64)     # (npatch, 3)
ctrlpts_all = np.vstack([np.array(p.flatCtrlPts()) for p in patches])   # (npatch*20, 3)
radii       = np.array([p.BS.r for p in patches], dtype=np.float64)
eps         = patches[0].eps
npatches    = len(patches)

# --- NN contact initialization ------------------------------------------------
hybrid_cda = None       # Phase 1: multitask NN broad phase + C++ refinement
neural_pull_cda = None  # Phase 2: pure NN SDF (g, ∇g, ∇²g directly)

# Training data was generated with potato shifted to origin (x -= 6).
# The NN expects coordinates in that shifted frame.
_ptt_center = np.array(ptt.X).mean(axis=0)
_nn_coord_offset = -_ptt_center  # [-6, 0, 0] approximately
# Bounding-sphere pre-filter: nodes beyond this distance can't be in contact.
# Training data covers gap in [-0.5, 1.5], so max distance from center is R + gn_max.
_ptt_bounding_r = np.linalg.norm(np.array(ptt.X) - _ptt_center, axis=1).max()
_nn_cutoff_r = _ptt_bounding_r + 1.5  # conservative: beyond training data range

if nn_contact:
    try:
        if nn_contact_mode == "neural_pull":
            from nn_contact.evaluation.integration import NeuralPullCDA
            neural_pull_cda = NeuralPullCDA.from_checkpoint(device=nn_contact_device)
            neural_pull_cda.coord_offset = _nn_coord_offset
            print(f"NN contact enabled: mode=neural_pull, device={nn_contact_device}, "
                  f"char_length={neural_pull_cda.char_length}")
        elif nn_contact_mode == "multitask":
            from nn_contact.evaluation.integration import HybridCDA
            hybrid_cda = HybridCDA.from_variant(
                nn_contact_model, device=nn_contact_device,
            )
            hybrid_cda.topk = nn_contact_topk
            hybrid_cda.coord_offset = _nn_coord_offset
            print(f"NN contact enabled: mode=multitask, model={nn_contact_model}, "
                  f"device={nn_contact_device}, topk={nn_contact_topk}")
        else:
            raise ValueError(f"Unknown nn_contact_mode: {nn_contact_mode}")
    except Exception as e:
        print(f"WARNING: NN contact disabled — {e}")
        hybrid_cda = None
        neural_pull_cda = None

# Dense surface sampling for KD-tree candidate selection
# Skip when NN is active (only needed for classical path / multitask fallback)
surf_kdtree = None
surf_pts = None
surf_pids = None

def _ensure_kdtree():
    """Lazy KD-tree construction — only built on first use."""
    global surf_kdtree, surf_pts, surf_pids
    if surf_kdtree is not None:
        return
    sample_t = np.linspace(0, 1, 50)
    pts, pids = [], []
    for pid, patch in enumerate(patches):
        for u_s in sample_t:
            for v_s in sample_t:
                pts.append(patch.Grg0(np.array([u_s, v_s], dtype=np.float64)))
                pids.append(pid)
    surf_pts  = np.asarray(pts,  dtype=np.float64)
    surf_pids = np.asarray(pids, dtype=np.int32)
    surf_kdtree = cKDTree(surf_pts)
    print(f"  KD-tree built: {len(surf_pts)} surface samples")

_nn_active = (hybrid_cda is not None) or (neural_pull_cda is not None)
if not _nn_active:
    _ensure_kdtree()
    print(f"Potato: {npatches} patches, {len(surf_pts)} surface samples for KD-tree")
else:
    print(f"Potato: {npatches} patches (KD-tree deferred, NN active)")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  IDENTIFY SLAVE (BOTTOM) AND DIRICHLET (TOP) VERTICES
# ══════════════════════════════════════════════════════════════════════════════
X_ref = np.array([list(mesh.vertices[i].point) for i in range(nv)])

# Use named boundary identifiers: ('back','left','front','right','bottom','top')
bnd_names = mesh.GetBoundaries()

def boundary_vertices(bnd_name):
    """Collect unique vertex indices on a named boundary."""
    verts = set()
    for el in mesh.Elements(BND):
        if bnd_names[el.index] == bnd_name:
            for v in el.vertices:
                verts.add(v.nr)
    return np.array(sorted(verts), dtype=np.int32)

slave_verts = boundary_vertices("bottom")
top_verts   = boundary_vertices("top")

print(f"Block mesh: {nv} vertices, {len(slave_verts)} slave (bottom), "
      f"{len(top_verts)} Dirichlet (top)")

# Free DOFs (those not on the Dirichlet boundary "top")
freedofs_ba = fes.FreeDofs()
free_dofs   = np.array([i for i in range(ndof) if freedofs_ba[i]], dtype=np.int32)

# DOF indices for top boundary (all 3 components, for prescribed displacement)
top_dofs_x = np.array([v for v in top_verts], dtype=np.int32)           # x-comp
top_dofs_y = np.array([v + nv for v in top_verts], dtype=np.int32)      # y-comp
top_dofs_z = np.array([v + 2*nv for v in top_verts], dtype=np.int32)    # z-comp

# ══════════════════════════════════════════════════════════════════════════════
# 5.  CONTACT PROJECTION
# ══════════════════════════════════════════════════════════════════════════════
# TR projection parameters (following ContactPotato_Ex1.py)
TR_INIT = 0.1
TR_MIN  = 1e-15
TR_MAX  = 1.0
BASE_NCAND = 12
MIN_NCAND  = 5
MAX_NCAND  = 96
RADIUS_FACTOR = 1.5
K_SURF = 15


# --- Contact cache for warm-starting within a load step -------------------
# Between consecutive solver evaluations the slave positions barely move,
# so we can reuse the previous projection result (patch id + parametric coords)
# and only recompute gap/normal cheaply via Grg + D3Grg (~0.006 ms/node)
# instead of a full TR search (~5 ms/node).

class ContactCache:
    """Caches TR projection results; recomputes gap from cached patch/param."""

    def __init__(self):
        self.prev_pos    = None      # (n_slave, 3)
        self.patch_ids   = None      # (n_slave,) int
        self.params      = None      # (n_slave, 2) parametric coords
        self.tol_reuse   = contact_tol_reuse
        self.xc_surf     = None      # (n_slave, 3) surface points for linesearch
        self.last_normals = None     # (n_slave, 3) normals for linesearch

    def reset(self):
        """Call at the start of each load step."""
        self.prev_pos = None

    def _neural_pull_evaluate(self, slave_pos, compute_hessian):
        """Pure NN evaluation: g, ∇g, ∇²g directly from Neural-Pull SDF.

        No C++ calls, no patch/param caching — the NN is the sole source.
        Returns (gn, normals, active, dndxs_all).
        """
        n_slave = slave_pos.shape[0]
        if perf: _t0 = perf_counter()

        # Bounding-sphere pre-filter: skip nodes far from the potato
        dist_to_center = np.linalg.norm(slave_pos - _ptt_center, axis=1)
        near_mask = dist_to_center < _nn_cutoff_r

        gn = np.full(n_slave, np.inf)
        normals = np.zeros((n_slave, 3))
        xc_surf = np.zeros((n_slave, 3))
        dndxs_all = np.zeros((n_slave, 3, 3)) if compute_hessian else None

        idx_near = np.where(near_mask)[0]
        if idx_near.size > 0:
            out = neural_pull_cda.evaluate(
                slave_pos[idx_near], compute_hessian=compute_hessian)
            gn[idx_near] = out["gn"]
            normals[idx_near] = out["normals"]
            xc_surf[idx_near] = out["xc_surf"]
            if compute_hessian and "dndxs" in out:
                dndxs_all[idx_near] = out["dndxs"]

        if perf: perf.record("contact_neural_pull", perf_counter() - _t0)

        # Store for linesearch
        self.xc_surf = xc_surf
        self.last_normals = normals
        self.patch_ids = np.zeros(n_slave, dtype=np.int32)
        self.prev_pos = slave_pos.copy()

        active = gn < 0
        return gn, normals, active, dndxs_all

    def _nn_project(self, pos_full, compute_hessian):
        """NN broad phase → warm-started Newton refinement.

        Two-phase approach for NN-active nodes:
        1. Fast path: Newton refinement from NN's (patch, xi) prediction
           using batch_refine_from_init — converges in 1-3 iterations.
        2. Slow path: full TR projection for the ~1% that don't converge.

        Three categories:
        - NN-active (confident + near surface): Newton warm-start → exact gap
        - Far-field (confident + far): skip, gn=+inf (no contact)
        - Low-confidence (rare): classical TR fallback with KD-tree

        Returns: (pids, params, gn, normals, xc_surf, dndxs).
        """
        n = pos_full.shape[0]

        # Bounding-sphere pre-filter: only query NN for nodes near the potato
        dist_to_center = np.linalg.norm(pos_full - _ptt_center, axis=1)
        near_mask = dist_to_center < _nn_cutoff_r

        # Allocate output arrays (far nodes keep gn=+inf → no contact)
        pids   = np.full(n, -1, dtype=np.int32)
        params = np.zeros((n, 2), dtype=np.float64)
        gn_out = np.full(n, np.inf)
        nor_out = np.zeros((n, 3))
        xc_out  = np.zeros((n, 3))
        dndxs_out = np.zeros((n, 3, 3)) if compute_hessian else None

        idx_near = np.where(near_mask)[0]
        if idx_near.size == 0:
            return pids, params, gn_out, nor_out, xc_out, dndxs_out

        nn_out = hybrid_cda.predict(pos_full[idx_near])

        # Map NN indices back to full arrays
        idx_nn_local = np.where(nn_out["active_mask"])[0]
        idx_fb_local = np.where(nn_out["needs_fallback"])[0]
        idx_nn = idx_near[idx_nn_local]
        idx_fb = idx_near[idx_fb_local]

        # ── NN path: Newton refinement from NN initial guess (fast path) ──
        if idx_nn.size > 0:
            nn_pids = nn_out["patch_ids"][idx_nn_local, 0]   # top-1 patch
            nn_xi   = nn_out["xi_init"][idx_nn_local, 0]     # top-1 xi (N, 2)
            pos_nn  = pos_full[idx_nn]

            # Fast path: Newton refinement from NN's (patch, xi) guess
            conv_mask, ref_pids, ref_t1, ref_t2, ref_gn, ref_nor, ref_xc = \
                gb.batch_refine_from_init(
                    ctrlpts_all,
                    nn_pids.astype(np.int32),
                    nn_xi.astype(np.float64),
                    pos_nn, radii, eps)

            conv = np.asarray(conv_mask, dtype=bool)
            idx_conv = idx_nn[conv]
            if idx_conv.size > 0:
                pids[idx_conv]      = np.asarray(ref_pids)[conv]
                params[idx_conv, 0] = np.asarray(ref_t1)[conv]
                params[idx_conv, 1] = np.asarray(ref_t2)[conv]
                gn_out[idx_conv]    = np.asarray(ref_gn)[conv]
                nor_out[idx_conv]   = np.asarray(ref_nor)[conv]
                xc_out[idx_conv]    = np.asarray(ref_xc)[conv]
                if compute_hessian:
                    params_c = np.column_stack([
                        np.asarray(ref_t1)[conv],
                        np.asarray(ref_t2)[conv]])
                    _, _, _, dndxs_c = gb.batch_evaluate_contact(
                        ctrlpts_all, np.asarray(ref_pids)[conv].astype(np.int32),
                        params_c, pos_nn[conv], eps, True)
                    if dndxs_c.size > 0:
                        dndxs_out[idx_conv] = dndxs_c

            # Slow path: full TR for non-converged nodes (~1% expected)
            idx_nonconv = idx_nn[~conv]
            if idx_nonconv.size > 0:
                n_nc = idx_nonconv.size
                print(f"  [NN contact] Newton non-conv: {n_nc}/{idx_nn.size} "
                      f"({n_nc/idx_nn.size*100:.1f}%) → full TR", flush=True)
                # Use NN top-K as candidate seeds for full TR
                nn_pids_k = nn_out["patch_ids"][idx_nn_local[~conv]]  # (n_nc, K)
                pos_nc = pos_full[idx_nonconv]
                fb_pids, fb_t1, fb_t2, fb_gn, fb_nor, fb_xc = \
                    gb.find_signed_distance_multi_points(
                        ctrlpts_all, pos_nc, xm_matrix,
                        nn_pids_k.astype(np.int32),
                        radii, eps, TR_INIT, TR_MIN, TR_MAX,
                        BASE_NCAND, MIN_NCAND, MAX_NCAND, RADIUS_FACTOR,
                    )
                fb_pids = np.asarray(fb_pids, dtype=np.int32)
                valid_fb = fb_pids >= 0
                idx_vfb = idx_nonconv[valid_fb]
                if idx_vfb.size > 0:
                    pids[idx_vfb]      = fb_pids[valid_fb]
                    params[idx_vfb, 0] = np.asarray(fb_t1)[valid_fb]
                    params[idx_vfb, 1] = np.asarray(fb_t2)[valid_fb]
                    gn_out[idx_vfb]    = np.asarray(fb_gn)[valid_fb]
                    nor_out[idx_vfb]   = np.asarray(fb_nor)[valid_fb]
                    xc_out[idx_vfb]    = np.asarray(fb_xc)[valid_fb]
                    if compute_hessian:
                        params_fb = np.column_stack([
                            np.asarray(fb_t1)[valid_fb],
                            np.asarray(fb_t2)[valid_fb]])
                        _, _, _, dndxs_fb = gb.batch_evaluate_contact(
                            ctrlpts_all, fb_pids[valid_fb], params_fb,
                            pos_nc[valid_fb], eps, True)
                        if dndxs_fb.size > 0:
                            dndxs_out[idx_vfb] = dndxs_fb

        # ── Fallback path: classical TR for low-confidence nodes ──
        if idx_fb.size > 0:
            print(f"  [NN contact] low-conf fallback to TR for {idx_fb.size}/{n} nodes "
                  f"({idx_fb.size/n*100:.1f}%)", flush=True)
            _ensure_kdtree()
            pos_fb = pos_full[idx_fb]
            pids_fb, t1_fb, t2_fb, gn_fb, nor_fb, xc_fb = project_points_tr_multi_batch(
                pos_fb, xm_matrix, ctrlpts_all, radii, eps,
                TR_INIT, TR_MIN, TR_MAX,
                surf_kdtree, surf_pids, BASE_NCAND, MIN_NCAND,
                MAX_NCAND, RADIUS_FACTOR, K_SURF,
            )
            pids[idx_fb]      = pids_fb.astype(np.int32)
            params[idx_fb, 0] = t1_fb
            params[idx_fb, 1] = t2_fb
            gn_out[idx_fb]    = gn_fb
            nor_out[idx_fb]   = nor_fb
            xc_out[idx_fb]    = xc_fb

            if compute_hessian:
                params_fb = np.column_stack([t1_fb, t2_fb])
                _, _, _, dndxs_fb = gb.batch_evaluate_contact(
                    ctrlpts_all, pids_fb.astype(np.int32), params_fb,
                    pos_fb, eps, True)
                if dndxs_fb.size > 0:
                    dndxs_out[idx_fb] = dndxs_fb

        return pids, params, gn_out, nor_out, xc_out, dndxs_out

    def evaluate(self, slave_pos, compute_hessian=False):
        """Return (gn, normals, active, dndxs_all) using cache when possible.

        Parameters
        ----------
        slave_pos : (n_slave, 3) array
            Current slave node positions
        compute_hessian : bool
            If True, compute dn/dx_s for contact Hessian

        Returns
        -------
        gn : (n_slave,) array
            Gap values (negative = penetration)
        normals : (n_slave, 3) array
            Unit normals at projection points
        active : (n_slave,) bool array
            Mask of penetrating nodes
        dndxs_all : (n_slave, 3, 3) array or None
            dn/dx_s matrices (only if compute_hessian=True)
        """
        # ── Neural-Pull: pure NN path, no C++ at all ──
        if neural_pull_cda is not None:
            return self._neural_pull_evaluate(slave_pos, compute_hessian)

        n_slave = slave_pos.shape[0]
        gn      = np.full(n_slave, np.inf)
        normals = np.zeros((n_slave, 3))
        xc_surf = np.zeros((n_slave, 3))
        dndxs_all = np.zeros((n_slave, 3, 3)) if compute_hessian else None

        # Decide per-node: reuse cache or full re-project
        need_full = np.ones(n_slave, dtype=bool)
        if self.prev_pos is not None:
            disp = np.linalg.norm(slave_pos - self.prev_pos, axis=1)
            can_reuse = disp < self.tol_reuse

            # Batch C++ evaluation for cached nodes (single pybind11 dispatch)
            idx_reuse = np.where(can_reuse)[0]
            if idx_reuse.size > 0:
                if perf: _t0 = perf_counter()
                g_r, n_r, xc_r, dndxs_r = gb.batch_evaluate_contact(
                    ctrlpts_all, self.patch_ids[idx_reuse].astype(np.int32),
                    self.params[idx_reuse], slave_pos[idx_reuse],
                    eps, compute_hessian)
                gn[idx_reuse] = g_r
                normals[idx_reuse] = n_r
                xc_surf[idx_reuse] = xc_r
                if compute_hessian and dndxs_r.size > 0:
                    dndxs_all[idx_reuse] = dndxs_r
                need_full[idx_reuse] = False
                if perf: perf.record("contact_cached", perf_counter() - _t0)

        # Full projection for nodes that need it
        idx_full = np.where(need_full)[0]
        if idx_full.size > 0:
            if perf: _t0 = perf_counter()
            if self.patch_ids is None:
                self.patch_ids = np.full(n_slave, -1, dtype=np.int32)
                self.params    = np.zeros((n_slave, 2), dtype=np.float64)

            pos_full = slave_pos[idx_full].astype(np.float64)

            if hybrid_cda is not None:
                # ── NN broad phase + C++ refinement (Phase 1c) ──
                pids_b, params_b, gn_b, nor_b, xc_b, dndxs_b = self._nn_project(
                    pos_full, compute_hessian)
                t1_b, t2_b = params_b[:, 0], params_b[:, 1]
                if perf: perf.record("contact_nn_project", perf_counter() - _t0)
            else:
                # ── Classical: KD-tree broad phase + TR projection ──
                pids_b, t1_b, t2_b, gn_b, nor_b, xc_b = project_points_tr_multi_batch(
                    pos_full, xm_matrix, ctrlpts_all, radii, eps,
                    TR_INIT, TR_MIN, TR_MAX,
                    surf_kdtree, surf_pids, BASE_NCAND, MIN_NCAND,
                    MAX_NCAND, RADIUS_FACTOR, K_SURF,
                )
                # Batch C++ Hessian computation from known (patch, params)
                dndxs_b = None
                if compute_hessian:
                    params_full = np.column_stack([t1_b, t2_b])
                    _, _, _, dndxs_b = gb.batch_evaluate_contact(
                        ctrlpts_all, pids_b.astype(np.int32), params_full,
                        pos_full, eps, True)
                if perf: perf.record("contact_full_tr", perf_counter() - _t0)

            # Scatter batch results back (vectorized where possible)
            gn[idx_full] = gn_b
            normals[idx_full] = nor_b
            xc_surf[idx_full] = xc_b
            self.patch_ids[idx_full] = pids_b.astype(np.int32)
            valid = pids_b >= 0
            if np.any(valid):
                idx_valid = idx_full[valid]
                self.params[idx_valid, 0] = t1_b[valid]
                self.params[idx_valid, 1] = t2_b[valid]
            if compute_hessian and dndxs_b is not None and dndxs_b.size > 0:
                dndxs_all[idx_full] = dndxs_b

        self.prev_pos = slave_pos.copy()
        self.xc_surf = xc_surf
        self.last_normals = normals
        active = gn < 0
        return gn, normals, active, dndxs_all

contact_cache = ContactCache()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  OBJECTIVE + CONTACT FORCES  (shared across solvers)
# ══════════════════════════════════════════════════════════════════════════════
res_vec = gfu.vec.CreateVector()       # scratch vector for Apply


def compute_contact_forces(gn, normals, active):
    """Assemble nodal contact penalty forces into a full DOF vector.

    f_con[v]    = kn * gn * n_x   for each active (penetrating) slave node
    f_con[v+nv] = kn * gn * n_y
    f_con[v+2*nv]= kn * gn * n_z

    Parameters
    ----------
    gn : (n_slave,) array — gap values (negative = penetration)
    normals : (n_slave, 3) array — unit normals at projection points
    active : (n_slave,) bool array — mask of penetrating nodes

    Returns
    -------
    f_con : (ndof,) array — contact force vector in full DOF space
    """
    f_con = np.zeros(ndof)
    if np.any(active):
        act = np.where(active)[0]
        verts_act = slave_verts[act]
        kgn = kn * gn[act]
        f_con[verts_act]          = kgn * normals[act, 0]
        f_con[verts_act + nv]     = kgn * normals[act, 1]
        f_con[verts_act + 2*nv]   = kgn * normals[act, 2]
    return f_con


def objective(x_free):
    """Return (total_energy, gradient_on_free_dofs) for scipy.minimize."""
    vec = gfu.vec.FV().NumPy()
    vec[free_dofs] = x_free

    # --- Plasticity: run return mapping at current u to update gf_Fp -------
    # Without this, Energy() and Apply() use stale Fp from previous step.
    # The envelope theorem guarantees that the gradient of W(Fe(u; Fp*(u)))
    # w.r.t. u equals dW/du|_{Fp*}, so Apply() gives the correct gradient
    # after updating Fp via return mapping.
    if plastic:
        global _Fp_temp, _delta_epcum, _F_flat_cache
        F_flat = _read_F_at_ips()
        _F_flat_cache = F_flat
        _Fp_temp, _delta_epcum, rm_ok = return_mapping(
            F_flat, _Fp_conv, _epcum_conv, c10, d1, My0, H_hard, m_hard)
        if not rm_ok:
            E_penalty = 1e4 * (1.0 + np.dot(x_free, x_free))
            g_penalty = 2e4 * x_free
            return E_penalty, g_penalty
        _write_Fp_to_gf(_Fp_temp)

    # --- Material energy + forces (NGSolve) --------------------------------
    E_mat = a_form.Energy(gfu.vec)

    # Guard against invalid states (element inversion J<0 → NaN energy).
    # Return a large but scaled energy so L-BFGS-B's linesearch backs off
    # without corrupting its internal Hessian approximation.  The gradient
    # is set to a large positive multiple of x_free (elastic restoring force
    # toward reference configuration) so the search direction stays valid.
    if not np.isfinite(E_mat):
        E_penalty = 1e4 * (1.0 + np.dot(x_free, x_free))
        g_penalty = 2e4 * x_free
        return E_penalty, g_penalty

    a_form.Apply(gfu.vec, res_vec)
    f_mat = res_vec.FV().NumPy().copy()        # residual = dE/du

    if not np.isfinite(f_mat[free_dofs]).all():
        E_penalty = 1e4 * (1.0 + np.dot(x_free, x_free))
        g_penalty = 2e4 * x_free
        return E_penalty, g_penalty

    # --- Current slave-node positions --------------------------------------
    slave_pos = np.column_stack([
        X_ref[slave_verts, 0] + vec[slave_verts],
        X_ref[slave_verts, 1] + vec[slave_verts + nv],
        X_ref[slave_verts, 2] + vec[slave_verts + 2*nv],
    ])

    # --- Contact penalty (cached TR projection) ----------------------------
    gn, normals, active, _ = contact_cache.evaluate(slave_pos)
    E_con = 0.5 * kn * np.sum(gn[active]**2)
    f_con = compute_contact_forces(gn, normals, active)

    f_total = f_mat + f_con
    return E_mat + E_con, f_total[free_dofs].copy()


# ══════════════════════════════════════════════════════════════════════════════
# 6a. HESSIAN-VECTOR PRODUCT FOR NEWTON-CG
# ══════════════════════════════════════════════════════════════════════════════
# Scratch vectors for Hessian-vector product
_hess_tmp = gfu.vec.CreateVector()
_hess_out = gfu.vec.CreateVector()

# Cache for AssembleLinearization and contact state in hessp.
# Scipy Newton-CG calls hessp(x_free, p) many times at the SAME x_free with
# different p (CG inner iterations). Reassembling the tangent and recomputing
# contact state each time is wasteful. We cache both and only recompute when
# x_free actually changes.
_hessp_cache = {
    "x_free": None,           # last linearization point
    "contact_data": None,     # (gn, normals, active_idx, dndxs_all)
}

# Precompute mapping from slave vertex DOFs to free_dofs index (avoids
# searchsorted in inner loop). -1 means the DOF is not free (Dirichlet).
_slave_dof_to_free = {}
for sv in slave_verts:
    for dof in [int(sv), int(sv) + nv, int(sv) + 2*nv]:
        idx = np.searchsorted(free_dofs, dof)
        if idx < len(free_dofs) and free_dofs[idx] == dof:
            _slave_dof_to_free[dof] = int(idx)


def hessp(x_free, p):
    """Hessian-vector product: H @ p for scipy Newton-CG.

    Computes (K_mat + K_con) @ p where:
    - K_mat: material tangent from NGSolve (linearization of hyperelastic energy)
    - K_con: contact Hessian = kn * (n ⊗ n + g * dn/dx_s) for penetrating nodes

    Caches the tangent assembly and contact evaluation across CG inner
    iterations (same x_free, different p).

    Parameters
    ----------
    x_free : ndarray
        Current solution on free DOFs
    p : ndarray
        Direction vector on free DOFs

    Returns
    -------
    Hp : ndarray
        Hessian-vector product on free DOFs
    """
    vec = gfu.vec.FV().NumPy()

    # Check if linearization point changed
    need_reassemble = (
        _hessp_cache["x_free"] is None
        or not np.array_equal(x_free, _hessp_cache["x_free"])
    )

    if need_reassemble:
        vec[free_dofs] = x_free

        # Plasticity: ensure gf_Fp is current (objective() may have been
        # called at this x_free already, but if hessp is called directly
        # we need to update Fp first for correct tangent).
        if plastic:
            global _Fp_temp, _delta_epcum, _F_flat_cache
            F_flat = _read_F_at_ips()
            _F_flat_cache = F_flat
            _Fp_temp, _delta_epcum, rm_ok = return_mapping(
                F_flat, _Fp_conv, _epcum_conv, c10, d1, My0, H_hard, m_hard)
            if not rm_ok:
                # Consistency with objective(): when RM fails, objective returns
                # penalty E = 1e4*(1 + x·x), grad = 2e4*x.  The Hessian of
                # that penalty is 2e4*I, so Hp = 2e4*p.
                return 2e4 * p
            _write_Fp_to_gf(_Fp_temp)

        # Assemble material tangent at new state (uses current gf_Fp)
        a_form.AssembleLinearization(gfu.vec)

        # Plastic consistent tangent correction (yielding GPs only)
        if plastic and consistent_tangent and np.any(_delta_epcum > 0):
            _add_plastic_tangent_correction()

        # Evaluate contact state at new state
        slave_pos = np.column_stack([
            X_ref[slave_verts, 0] + vec[slave_verts],
            X_ref[slave_verts, 1] + vec[slave_verts + nv],
            X_ref[slave_verts, 2] + vec[slave_verts + 2*nv],
        ])
        gn, normals, active, dndxs_all = contact_cache.evaluate(
            slave_pos, compute_hessian=full_hessian
        )
        active_idx = np.where(active)[0] if np.any(active) else np.array([], dtype=int)

        _hessp_cache["x_free"] = x_free.copy()
        _hessp_cache["contact_data"] = (gn, normals, active_idx, dndxs_all)
    else:
        gn, normals, active_idx, dndxs_all = _hessp_cache["contact_data"]

    # --- Material Hessian-vector product ---
    _hess_tmp.FV().NumPy()[:] = 0.0
    _hess_tmp.FV().NumPy()[free_dofs] = p

    _hess_out.data = a_form.mat * _hess_tmp
    Hp = _hess_out.FV().NumPy()[free_dofs].copy()

    # --- Contact Hessian-vector product ---
    if len(active_idx) > 0:
        p_full = np.zeros(ndof)
        p_full[free_dofs] = p
        p_slave = np.column_stack([
            p_full[slave_verts],
            p_full[slave_verts + nv],
            p_full[slave_verts + 2*nv],
        ])

        for i in active_idx:
            g = gn[i]
            if g >= 0:
                continue

            nor = normals[i]
            v = int(slave_verts[i])
            p_v = p_slave[i]

            # Full contact Hessian-vector product: K_con @ p_v
            # Uses absolute eigenvalue filtering for PSD guarantee
            if full_hessian and dndxs_all is not None and np.any(dndxs_all[i]):
                K_con = compute_contact_hessian(g, nor, dndxs_all[i])
                Hp_contact = K_con @ p_v
            else:
                # Simple Hessian: kn * (n ⊗ n) @ p_v = kn * (n · p_v) * n
                Hp_contact = kn * (nor @ p_v) * nor

            # Map back to free DOF indices via precomputed lookup
            for k, dof in enumerate([v, v + nv, v + 2*nv]):
                idx_in_free = _slave_dof_to_free.get(dof, -1)
                if idx_in_free >= 0:
                    Hp[idx_in_free] += Hp_contact[k]

    return Hp


# ══════════════════════════════════════════════════════════════════════════════
# 6b-pre. RETURN MAPPING (J2 plasticity, multiplicative decomposition)
# ══════════════════════════════════════════════════════════════════════════════
if plastic:
    def _write_Fp_to_gf(Fp_flat):
        """Write flattened Fp array (n_ip*9,) into gf_Fp GridFunction.

        DOF layout of MatrixValued(IntegrationRuleSpace): block by component.
        [comp0_ip0..comp0_ipN, comp1_ip0..comp1_ipN, ..., comp8_ip0..comp8_ipN]
        where comp = row*3 + col (row-major).
        """
        vec = gf_Fp.vec.FV().NumPy()
        Fp_all = Fp_flat.reshape(n_ip, 9)   # (n_ip, 9)  row-major per GP
        for comp in range(9):
            vec[comp * n_ip : (comp + 1) * n_ip] = Fp_all[:, comp]

    def _read_F_at_ips():
        """Evaluate F = I + Grad(u) at all integration points.

        Returns flattened array (n_ip*9,) in the same per-GP row-major layout
        used by return_mapping().
        """
        gf_F_ir.Interpolate(Id(3) + Grad(gfu))
        vec = gf_F_ir.vec.FV().NumPy()
        F_all = np.zeros((n_ip, 9))
        for comp in range(9):
            F_all[:, comp] = vec[comp * n_ip : (comp + 1) * n_ip]
        return F_all.ravel()

    def return_mapping(F_flat, Fp_conv_flat, epcum_conv,
                       c10_, d1_, My0_, H_, m_):
        """J2 return mapping with exponential update at all integration points.

        Implements the multiplicative plasticity algorithm from paper.tex:
        - Trial elastic predictor: Fe_trial = F · Fp_old^{-1}
        - Mandel stress: M = 2c10(Ce - I) + 2d1·ln(Je)·I
        - J2 yield: f = σ_vm - σ_y(epcum)
        - Exponential map update: Fp_new = exp(Δλ·N_flow) · Fp_old

        Parameters
        ----------
        F_flat : ndarray (n_ip*9,)
            Total deformation gradients at all IPs
        Fp_conv_flat : ndarray (n_ip*9,)
            Converged Fp from previous load step
        epcum_conv : ndarray (n_ip,)
            Converged cumulative plastic strain
        c10_, d1_ : float
            Neo-Hookean material parameters
        My0_, H_, m_ : float
            Yield stress, hardening modulus, hardening exponent

        Returns
        -------
        Fp_new_flat : ndarray (n_ip*9,)
        delta_epcum : ndarray (n_ip,)
        success : bool
        """
        n = len(epcum_conv)
        F_all  = F_flat.reshape(n, 3, 3)
        Fp_old = Fp_conv_flat.reshape(n, 3, 3)
        Fp_new = Fp_old.copy()
        delta_epcum = np.zeros(n)
        I3 = np.eye(3)
        n_local_fail = 0   # count of GPs where local Newton did not converge

        for ip in range(n):
            F  = F_all[ip]
            Fp = Fp_old[ip]
            epcum = epcum_conv[ip]

            # --- Trial elastic predictor ---
            Fp_inv = np.linalg.inv(Fp)
            Fe = F @ Fp_inv
            Ce = Fe.T @ Fe
            Je = np.linalg.det(Fe)
            if Je <= 1e-15:
                # Element inversion — keep old Fp, let global Newton handle it
                continue

            lnJe = np.log(Je)

            # Mandel stress: M = 2c10*(Ce - I) + 2d1*ln(Je)*I
            M = 2*c10_*(Ce - I3) + 2*d1_*lnJe*I3
            M_dev = M - (1.0/3.0)*np.trace(M)*I3
            norm_Mdev_sq = np.sum(M_dev * M_dev)     # Frobenius norm squared
            sigma_vm = np.sqrt(1.5 * norm_Mdev_sq)

            # Yield stress
            sigma_y = My0_ + (H_ * epcum**m_ if epcum > 1e-30 else 0.0)

            # Yield check
            f_yield = sigma_vm - sigma_y
            if f_yield <= 0.0:
                continue    # elastic — Fp unchanged

            # --- Plastic: local Newton on scalar Δλ ---
            # Flow direction (deviatoric, traceless, associated J2 flow)
            if sigma_vm > 1e-30:
                N_flow = 1.5 * M_dev / sigma_vm
            else:
                continue

            # Eigendecomposition of N_flow (symmetric): compute once
            eigvals_N, V_N = np.linalg.eigh(N_flow)

            dlam = 1e-8     # initial guess
            converged_rm = False
            R_val = f_yield  # initial residual
            Fp_trial = Fp.copy()
            for k_rm in range(50):
                # Exponential map: Fp_trial = expm(dlam * N_flow) @ Fp_old
                exp_diag = np.exp(dlam * eigvals_N)
                exp_dlam_N = (V_N * exp_diag) @ V_N.T
                Fp_trial = exp_dlam_N @ Fp

                # Recompute elastic state (with robust inversion)
                det_Fp_trial = np.linalg.det(Fp_trial)
                if abs(det_Fp_trial) < 1e-15:
                    break
                try:
                    Fe_new = F @ np.linalg.inv(Fp_trial)
                except np.linalg.LinAlgError:
                    break
                Ce_new = Fe_new.T @ Fe_new
                Je_new = np.linalg.det(Fe_new)
                if Je_new <= 1e-15 or not np.isfinite(Je_new):
                    break

                M_new = 2*c10_*(Ce_new - I3) + 2*d1_*np.log(Je_new)*I3
                M_dev_new = M_new - (1.0/3.0)*np.trace(M_new)*I3
                svm_new = np.sqrt(1.5 * np.sum(M_dev_new * M_dev_new))

                ep_total = epcum + dlam
                sy_new = My0_ + (H_ * ep_total**m_ if ep_total > 1e-30 else 0.0)

                R_val = svm_new - sy_new

                if abs(R_val) < 1e-15:
                    converged_rm = True
                    break

                # Finite-difference derivative dR/d(dlam)
                h_fd = max(1e-10, abs(dlam) * 1e-6)
                exp_diag_h = np.exp((dlam + h_fd) * eigvals_N)
                Fp_h = ((V_N * exp_diag_h) @ V_N.T) @ Fp
                det_Fp_h = np.linalg.det(Fp_h)
                if abs(det_Fp_h) < 1e-15:
                    break
                try:
                    Fe_h = F @ np.linalg.inv(Fp_h)
                except np.linalg.LinAlgError:
                    break
                Ce_h = Fe_h.T @ Fe_h
                Je_h = np.linalg.det(Fe_h)
                if Je_h <= 1e-15 or not np.isfinite(Je_h):
                    break
                M_h = 2*c10_*(Ce_h - I3) + 2*d1_*np.log(Je_h)*I3
                M_dev_h = M_h - (1.0/3.0)*np.trace(M_h)*I3
                svm_h = np.sqrt(1.5 * np.sum(M_dev_h * M_dev_h))
                ep_h = epcum + dlam + h_fd
                sy_h = My0_ + (H_ * ep_h**m_ if ep_h > 1e-30 else 0.0)
                R_h = svm_h - sy_h

                dR = (R_h - R_val) / h_fd
                if abs(dR) < 1e-30:
                    break

                dlam -= R_val / dR
                dlam = max(dlam, 0.0)

            if not converged_rm and abs(R_val) > 1e-6:
                n_local_fail += 1

            Fp_new[ip]     = Fp_trial
            delta_epcum[ip] = max(dlam, 0.0)

        # Check for NaN/Inf in output
        if not np.isfinite(Fp_new).all():
            return Fp_conv_flat.copy(), np.zeros(n), False

        if n_local_fail > 0:
            print(f"    return_mapping: {n_local_fail}/{n} GPs failed local Newton")
            return Fp_conv_flat.copy(), np.zeros(n), False

        return Fp_new.ravel(), delta_epcum, True


    # ── Consistent tangent correction (FD-based, per yielding GP) ────────
    _I3 = np.eye(3)

    def _rm_single_gp(F, Fp_old, epcum, c10_, d1_, My0_, H_, m_):
        """Single-GP return mapping. Returns (Fp_new, dlam).

        Lightweight version of return_mapping() for FD tangent perturbations.
        Same algorithm: trial predictor, J2 yield check, exponential map update.
        """
        I3 = _I3
        Fp_inv = np.linalg.inv(Fp_old)
        Fe = F @ Fp_inv
        Ce = Fe.T @ Fe
        Je = np.linalg.det(Fe)
        if Je <= 1e-15:
            return Fp_old.copy(), 0.0

        M = 2*c10_*(Ce - I3) + 2*d1_*np.log(Je)*I3
        M_dev = M - np.trace(M)/3.0 * I3
        svm = np.sqrt(1.5 * np.sum(M_dev * M_dev))
        sy = My0_ + (H_ * epcum**m_ if epcum > 1e-30 else 0.0)
        if svm <= sy:
            return Fp_old.copy(), 0.0

        N_flow = 1.5 * M_dev / svm
        eigvals_N, V_N = np.linalg.eigh(N_flow)

        dlam = 1e-8
        Fp_trial = Fp_old.copy()
        for _ in range(50):
            exp_diag = np.exp(dlam * eigvals_N)
            Fp_trial = (V_N * exp_diag) @ V_N.T @ Fp_old
            det_Fp = np.linalg.det(Fp_trial)
            if abs(det_Fp) < 1e-15:
                break
            Fe_n = F @ np.linalg.inv(Fp_trial)
            Ce_n = Fe_n.T @ Fe_n
            Je_n = np.linalg.det(Fe_n)
            if Je_n <= 1e-15 or not np.isfinite(Je_n):
                break
            M_n = 2*c10_*(Ce_n - I3) + 2*d1_*np.log(Je_n)*I3
            M_dev_n = M_n - np.trace(M_n)/3.0 * I3
            svm_n = np.sqrt(1.5 * np.sum(M_dev_n * M_dev_n))
            ep_t = epcum + dlam
            sy_n = My0_ + (H_ * ep_t**m_ if ep_t > 1e-30 else 0.0)
            R = svm_n - sy_n
            if abs(R) < 1e-15:
                break
            h_fd = max(1e-10, abs(dlam) * 1e-6)
            exp_h = np.exp((dlam + h_fd) * eigvals_N)
            Fp_h = (V_N * exp_h) @ V_N.T @ Fp_old
            if abs(np.linalg.det(Fp_h)) < 1e-15:
                break
            Fe_h = F @ np.linalg.inv(Fp_h)
            Je_h = np.linalg.det(Fe_h)
            if Je_h <= 1e-15 or not np.isfinite(Je_h):
                break
            Ce_h = Fe_h.T @ Fe_h
            M_h = 2*c10_*(Ce_h - I3) + 2*d1_*np.log(Je_h)*I3
            M_dev_h = M_h - np.trace(M_h)/3.0 * I3
            svm_h = np.sqrt(1.5 * np.sum(M_dev_h * M_dev_h))
            ep_h = epcum + dlam + h_fd
            R_h = svm_h - (My0_ + (H_ * ep_h**m_ if ep_h > 1e-30 else 0.0))
            dR = (R_h - R) / h_fd
            if abs(dR) < 1e-30:
                break
            dlam -= R / dR
            dlam = max(dlam, 0.0)
        return Fp_trial, max(dlam, 0.0)

    def _compute_P_at_gp(F, Fp_new, c10_, d1_):
        """First Piola-Kirchhoff stress P = Pe · Fp^{-T} at one GP."""
        Fp_inv = np.linalg.inv(Fp_new)
        Fe = F @ Fp_inv
        Je = np.linalg.det(Fe)
        if Je <= 1e-15:
            return None
        invFeT = np.linalg.inv(Fe).T
        lnJe = np.log(Je)
        Pe = 2*c10_*(Fe - invFeT) + 2*d1_*lnJe*invFeT
        return Pe @ Fp_inv.T

    def _add_plastic_tangent_correction():
        """Add consistent tangent correction for all yielding GPs.

        For each yielding GP, computes the correction ΔCxx = Cxx_full - Cxx_elastic
        via finite differences, where:
        - Cxx_full[i,A,j,B]:    (P(F+h*e_jB, Fp_new(F+h*e_jB)) - P0) / h
        - Cxx_elastic[i,A,j,B]: (P(F+h*e_jB, Fp_new_frozen)    - P0) / h

        Both are FD-based to avoid any analytical/symbolic mismatch with NGSolve's
        AssembleLinearization. The elastic parts cancel exactly in the subtraction:
            dCxx[:,:,j,B] = (P_full_pert - P_elastic_pert) / h

        Cost: 9 return-mapping + 18 stress evaluations per yielding GP.
        """
        global _elem_csr_pos_plastic

        if _elem_csr_pos_plastic is None:
            _build_elem_csr_pos_plastic()

        vals_np = np.array(a_form.mat.CSR()[0], copy=False)

        F_all = _F_flat_cache.reshape(n_ip, 3, 3)
        Fp_old_all = _Fp_conv.reshape(n_ip, 3, 3)
        Fp_new_all = _Fp_temp.reshape(n_ip, 3, 3)

        for ip in range(n_ip):
            dlam = _delta_epcum[ip]
            if dlam <= 0:
                continue

            e  = ip // _gps_per_elem
            ig = ip %  _gps_per_elem

            F      = F_all[ip]
            Fp_old = Fp_old_all[ip]
            Fp_new = Fp_new_all[ip]
            epcum  = _epcum_conv[ip]

            h = max(1e-7, np.linalg.norm(F) * 1e-7)
            dCxx = np.zeros((3, 3, 3, 3))
            skip = False

            for j in range(3):
                for B in range(3):
                    F_pert = F.copy()
                    F_pert[j, B] += h

                    # Full tangent: run return mapping at perturbed F
                    Fp_pert, _ = _rm_single_gp(
                        F_pert, Fp_old, epcum, c10, d1, My0, H_hard, m_hard)
                    P_full = _compute_P_at_gp(F_pert, Fp_pert, c10, d1)
                    if P_full is None:
                        skip = True; break

                    # Elastic tangent: Fp frozen at current Fp_new
                    P_elastic = _compute_P_at_gp(F_pert, Fp_new, c10, d1)
                    if P_elastic is None:
                        skip = True; break

                    # Correction: the (P0 - P0) base terms cancel exactly
                    dCxx[:, :, j, B] = (P_full - P_elastic) / h
                if skip:
                    break
            if skip:
                continue

            # Assembly: ΔK[(I,a),(J,b)] = w*detJ * Σ_{k,l} dNdX[I,k] * dCxx[a,k,b,l] * dNdX[J,l]
            dNdX = _all_dNdX[e, ig]
            w_detJ = _gp_weight * _all_detJ[e, ig]

            temp = np.einsum('Ik,akbl->Iabl', dNdX, dCxx)
            dK_4d = np.einsum('Iabl,Jl->IaJb', temp, dNdX)

            # Block DOF order: dK_4d[I,a,J,b] → dK[a*8+I, b*8+J]
            dK = w_detJ * dK_4d.transpose(1, 0, 3, 2).reshape(24, 24)

            pos_map = _elem_csr_pos_plastic[e]
            mask = pos_map >= 0
            vals_np[pos_map[mask]] += dK[mask]


# ══════════════════════════════════════════════════════════════════════════════
# 6b. NEWTON SOLVER WITH DYNAMIC CONTACT
# ══════════════════════════════════════════════════════════════════════════════

def compute_slave_pos():
    """Get current slave node positions from gfu."""
    vec = gfu.vec.FV().NumPy()
    return np.column_stack([
        X_ref[slave_verts, 0] + vec[slave_verts],
        X_ref[slave_verts, 1] + vec[slave_verts + nv],
        X_ref[slave_verts, 2] + vec[slave_verts + 2*nv],
    ])


def compute_contact_hessian(g, nor, dndxs=None):
    """Compute the PSD contact Hessian for a penetrating node.

    The contact energy is E_con = (1/2) * kn * g_n^2, where g_n = (x_s - x_c) · n.
    The full Hessian is:
        K_con = kn * (n ⊗ n + g_n * dn/dx_s)

    When full_hessian is enabled and the curvature term g*dn/dx_s makes K_con
    indefinite, we project to PSD via absolute eigenvalue filtering:
        K_con = Q |Λ| Qᵀ
    This preserves the magnitude of all curvature information while guaranteeing
    positive semi-definiteness (Chen et al. 2024, "Stabler Neo-Hookean Simulation:
    Absolute Eigenvalue Filtering for Projected Newton", SIGGRAPH 2024).

    Parameters
    ----------
    g : float
        Current gap value (negative for penetration)
    nor : ndarray (3,)
        Unit normal at projection point
    dndxs : ndarray (3, 3) or None
        Pre-computed dn/dx_s matrix from projection. If None, only n⊗n is used.

    Returns
    -------
    K_con : ndarray (3, 3)
        PSD contact Hessian contribution for this node
    """
    K_base = np.outer(nor, nor)

    if full_hessian and dndxs is not None and np.any(dndxs):
        K_raw = kn * (K_base + g * dndxs)

        # Absolute eigenvalue filtering: K_psd = Q |Λ| Qᵀ
        eigvals, Q = np.linalg.eigh(K_raw)
        eigvals = np.abs(eigvals)
        K_con = (Q * eigvals) @ Q.T
    else:
        K_con = kn * K_base

    return K_con


_w_vec = gfu.vec.CreateVector()   # scratch for Newton direction
_uh_vec = gfu.vec.CreateVector()  # scratch for linesearch trial
_cached_inv = None                # Direct solver inverse (symbolic reuse via .Update() when supported)
# Solvers supporting .Update() (numeric refactorization only, reuses symbolic):
#   umfpack, sparsecholesky
# Solvers NOT supporting .Update() (must recreate each time):
#   pardiso, mumps
_inv_supports_update = linear_solver in ("umfpack", "sparsecholesky")

# CSR position map for contact Hessian assembly (lazy-init on first use).
# Maps each slave vertex's 3×3 DOF block to positions in the sparse matrix
# CSR value array, enabling direct numpy += instead of 9 mat[i,j] accesses.
_csr_pos_map = None     # ndarray (n_slave, 3, 3) of CSR value indices


def _build_csr_pos_map():
    """Build CSR position map for slave vertex DOF blocks.

    Called once (lazy) after the first AssembleLinearization.
    Maps (slave_idx, a, b) → position in CSR values array where
    mat[slave_dofs[a], slave_dofs[b]] is stored.
    """
    global _csr_pos_map
    mat = a_form.mat
    _, cols_fv, firsti_fv = mat.CSR()
    cols = np.array(cols_fv, copy=False)
    firsti = np.array(firsti_fv, copy=False)
    n_slave = len(slave_verts)
    _csr_pos_map = np.full((n_slave, 3, 3), -1, dtype=np.int64)
    for si in range(n_slave):
        v = int(slave_verts[si])
        dofs = [v, v + nv, v + 2*nv]
        for a in range(3):
            row = dofs[a]
            row_start = int(firsti[row])
            row_end = int(firsti[row + 1])
            row_cols = cols[row_start:row_end]
            for b in range(3):
                col = dofs[b]
                pos = np.searchsorted(row_cols, col)
                if pos < len(row_cols) and row_cols[pos] == col:
                    _csr_pos_map[si, a, b] = row_start + pos

# Precomputed surface data for vectorized linesearch energy evaluation.
# Populated once per Newton iteration in newton_solve(); used by _linesearch_energy().
# When contact_tol_reuse=inf, projections are locked and surface points/
# normals don't change between linesearch evaluations — only slave_pos changes.
_ls_xc    = np.zeros((0, 3))     # surface projection points
_ls_nor   = np.zeros((0, 3))     # unit normals at projection points
_ls_valid = np.zeros(0, dtype=bool)  # mask: True if projection exists


def _precompute_ls_data():
    """Precompute surface points and normals for vectorized linesearch energy.

    Called once per Newton iteration after contact_cache.evaluate().
    Reuses xc_surf and normals already computed by evaluate() — no extra C++ call.
    """
    global _ls_xc, _ls_nor, _ls_valid
    n_slave = len(slave_verts)
    if contact_cache.patch_ids is None or contact_cache.xc_surf is None:
        _ls_xc    = np.zeros((n_slave, 3))
        _ls_nor   = np.zeros((n_slave, 3))
        _ls_valid = np.zeros(n_slave, dtype=bool)
        return
    _ls_xc  = contact_cache.xc_surf
    _ls_nor = contact_cache.last_normals
    _ls_valid = contact_cache.patch_ids >= 0


def _linesearch_energy(u_vec):
    """Total energy (material + contact) at u_vec using precomputed projections.

    Uses _ls_xc, _ls_nor, _ls_valid (precomputed once per Newton iteration)
    to evaluate contact energy via vectorized numpy ops.
    Returns np.inf for invalid states (NaN, element inversion).
    """
    if perf: _t0 = perf_counter()
    E_mat = a_form.Energy(u_vec)
    if perf: perf.record("ls_energy_mat", perf_counter() - _t0)
    if not np.isfinite(E_mat):
        return np.inf

    vec_np = u_vec.FV().NumPy()
    if not np.isfinite(vec_np).all():
        return np.inf

    E_con = 0.0
    if np.any(_ls_valid):
        if perf: _t0 = perf_counter()
        slave_pos = np.column_stack([
            X_ref[slave_verts, 0] + vec_np[slave_verts],
            X_ref[slave_verts, 1] + vec_np[slave_verts + nv],
            X_ref[slave_verts, 2] + vec_np[slave_verts + 2*nv],
        ])
        # Vectorized gap computation: g_n = (x_s - x_c) · n
        gaps = np.einsum('ij,ij->i', slave_pos - _ls_xc, _ls_nor)
        pen_mask = _ls_valid & (gaps < 0)
        if np.any(pen_mask):
            E_con = 0.5 * kn * np.sum(gaps[pen_mask]**2)
        if perf: perf.record("ls_energy_contact", perf_counter() - _t0)
    return E_mat + E_con


def newton_solve():
    """Newton solver with dynamic contact and Armijo linesearch.

    Unlike the previous active-set Newton, this solver does NOT use an
    outer active-set loop.  Instead it evaluates contact at every Newton
    iteration through the cache (projections locked for consistency,
    same strategy as the Newton-CG / L-BFGS-B scipy paths).

    The active set evolves naturally: nodes enter/exit contact as their
    gap changes sign during Newton iterations.  This avoids the active-
    set oscillation that plagued the old approach.

    Returns
    -------
    n_iter : int
        Number of Newton iterations
    converged : bool
        True if residual met tolerance or stagnation after significant reduction
    gn, normals, active : arrays
        Final contact state (from last iteration)
    """
    # Neural-Pull bypasses C++ entirely → irreducible gap error ~0.01 →
    # residual floor ~kn * 0.01.  Multitask uses C++ TR refinement → exact gaps.
    if neural_pull_cda is not None:
        newton_gtol = max(gtol, 1e-3)
    else:
        newton_gtol = gtol
    _u_backup = gfu.vec.FV().NumPy().copy()

    rnorm_prev = np.inf
    rnorm_initial = np.inf
    stag_count = 0
    converged = False
    gn_out, normals_out, active_out = None, None, None

    for nit in range(max_iter):
        # 0. Return mapping (plasticity only): recompute Fp at all Gauss points
        #    from the committed Fp_conv, given the current displacement estimate.
        #    This matches the paper: "every time u changes, the plastic variables
        #    are recomputed exactly" (monolithic displacement-driven approach).
        #    AssembleLinearization gives the elastic tangent (frozen Fp); the
        #    plastic correction is added afterwards by _add_plastic_tangent_correction.
        if plastic:
            if perf: _t0 = perf_counter()
            global _Fp_temp, _delta_epcum, _F_flat_cache
            F_flat = _read_F_at_ips()
            _F_flat_cache = F_flat  # cache for tangent correction
            _Fp_temp, _delta_epcum, rm_ok = return_mapping(
                F_flat, _Fp_conv, _epcum_conv, c10, d1, My0, H_hard, m_hard)
            if not rm_ok:
                print(f"    Newton {nit}: return mapping FAILED — aborting")
                gfu.vec.FV().NumPy()[:] = _u_backup
                break
            _write_Fp_to_gf(_Fp_temp)
            if perf: perf.record("return_mapping", perf_counter() - _t0)

        # 1. Material residual (uses Fe = F·Fp^{-1} when plastic, via gf_Fp)
        if perf: _t0 = perf_counter()
        a_form.Apply(gfu.vec, res_vec)
        if perf: perf.record("apply", perf_counter() - _t0)
        r_np = res_vec.FV().NumPy()

        # 2. Contact evaluation (cached projections, first call does full TR)
        slave_pos = compute_slave_pos()
        if not np.isfinite(slave_pos).all():
            gfu.vec.FV().NumPy()[:] = _u_backup
            break

        if perf: _t0 = perf_counter()
        gn_out, normals_out, active_out, dndxs = contact_cache.evaluate(
            slave_pos, compute_hessian=full_hessian
        )
        if perf: perf.record("contact_eval", perf_counter() - _t0)
        active_idx = np.where(active_out)[0]

        # 2b. Precompute surface data for vectorized linesearch energy
        _precompute_ls_data()

        # 3. Add contact forces (vectorized) + collect Hessian data
        if perf: _t0 = perf_counter()
        pen_data = []
        if len(active_idx) > 0:
            verts_act = slave_verts[active_idx]
            kgn = kn * gn_out[active_idx]
            r_np[verts_act]          += kgn * normals_out[active_idx, 0]
            r_np[verts_act + nv]     += kgn * normals_out[active_idx, 1]
            r_np[verts_act + 2*nv]   += kgn * normals_out[active_idx, 2]
            for i in active_idx:
                K_con = compute_contact_hessian(
                    gn_out[i], normals_out[i],
                    dndxs[i] if dndxs is not None else None
                )
                pen_data.append((i, gn_out[i], normals_out[i], K_con))
        if perf: perf.record("contact_force_asm", perf_counter() - _t0)

        # 4. Convergence check
        rnorm = np.linalg.norm(r_np[free_dofs])
        if nit == 0:
            rnorm_initial = rnorm
        if rnorm < newton_gtol:
            converged = True
            break

        # 5. Stagnation detection
        #    Require at least 10 iterations AND significant residual reduction
        #    before declaring stagnation.  In contact problems, Newton can have
        #    slow initial convergence as the active set settles — premature
        #    stagnation exit wastes the entire load step.
        #
        #    NOTE (plasticity): stagnation sets converged=True, which commits
        #    _Fp_conv/_epcum_conv even though rnorm may be above gtol.  This
        #    is acceptable because: (1) Fp is consistent with u (RM ran at
        #    this displacement), (2) 100× reduction ensures we are in the
        #    basin of attraction, (3) the active-set oscillation that causes
        #    stagnation is a contact phenomenon unrelated to plastic state,
        #    (4) the cutback mechanism catches any downstream issues.  The
        #    printed |r| makes non-strict convergence visible to the user.
        if nit >= 10 and rnorm < 1e-2 * rnorm_initial:
            if rnorm > 0.95 * rnorm_prev:
                stag_count += 1
                if stag_count >= 5:
                    converged = True  # stagnated after significant reduction
                    break
            else:
                stag_count = 0
        rnorm_prev = rnorm

        # 6. Material tangent
        if perf: _t0 = perf_counter()
        a_form.AssembleLinearization(gfu.vec)
        if perf: perf.record("assemble_lin", perf_counter() - _t0)

        # 6b. Plastic consistent tangent correction (yielding GPs only)
        #     Adds the rank-1 correction ΔK[ia,jb] = B[ia]·D[jb] per GP
        #     that accounts for dFp/dF.  B is analytical (exact), D is FD-based
        #     (exact: captures full dN/dF chain via return-mapping perturbation).
        if plastic and consistent_tangent and np.any(_delta_epcum > 0):
            if perf: _t0 = perf_counter()
            _add_plastic_tangent_correction()
            if perf: perf.record("plastic_tangent", perf_counter() - _t0)

        # 7. Add contact Hessian: K_con = kn * (n⊗n + g·dn/dx_s)
        #    Uses CSR direct write: position map built once, then numpy +=
        if perf: _t0 = perf_counter()
        if _csr_pos_map is None:
            _build_csr_pos_map()
        mat = a_form.mat
        vals_np = np.array(mat.CSR()[0], copy=False)
        for idx, g, nor, K_con in pen_data:
            positions = _csr_pos_map[idx]  # (3, 3) array of CSR positions
            for a in range(3):
                for b in range(3):
                    pos = positions[a, b]
                    if pos >= 0:
                        vals_np[pos] += K_con[a, b]
        if perf: perf.record("contact_hess_asm", perf_counter() - _t0)

        # 8. Solve K·Δu = -r
        if perf: _t0 = perf_counter()
        global _cached_inv
        solve_ok = False
        try:
            if _inv_supports_update and _cached_inv is not None:
                _cached_inv.Update()
            else:
                _cached_inv = mat.Inverse(fes.FreeDofs(), inverse=linear_solver)
            _w_vec.data = _cached_inv * res_vec
            if np.isfinite(_w_vec.FV().NumPy()).all():
                solve_ok = True
            else:
                _cached_inv = None  # invalidate on NaN
        except Exception:
            _cached_inv = None  # invalidate on failure
        if perf: perf.record("linear_solve", perf_counter() - _t0)

        if not solve_ok:
            r_max = np.max(np.abs(r_np[free_dofs]))
            if r_max > 1e-30:
                scale = 0.1 * h_contact / r_max
                w_np = _w_vec.FV().NumPy()
                w_np[:] = 0
                w_np[free_dofs] = scale * r_np[free_dofs]
            else:
                break

        # 9. Armijo linesearch
        #    NOTE: _linesearch_energy uses frozen gf_Fp (from the RM at the
        #    start of this iteration).  The energy evaluated at trial points
        #    u_k - τ·w is therefore W(u_trial; Fp*(u_k)), not the fully
        #    plastic-consistent W(u_trial; Fp*(u_trial)).  This is standard
        #    practice (Simo & Hughes 1998, de Souza Neto 2008): the envelope
        #    theorem guarantees that the slope φ'(0) = -r·w is exact at τ=0,
        #    so the Armijo condition remains valid.  Running RM at each trial
        #    τ would cost up to 30× RM per iteration for marginal benefit.
        if perf: _t0 = perf_counter()
        w_free = _w_vec.FV().NumPy()[free_dofs]
        slope = -np.dot(r_np[free_dofs], w_free)  # φ'(0) = -∇E·w

        if slope >= 0:
            # Not a descent direction — use gradient step
            r_max = np.max(np.abs(r_np[free_dofs]))
            if r_max > 1e-30:
                scale = 0.1 * h_contact / r_max
                w_np = _w_vec.FV().NumPy()
                w_np[:] = 0
                w_np[free_dofs] = scale * r_np[free_dofs]
                w_free = w_np[free_dofs]
                slope = -np.dot(r_np[free_dofs], w_free)
            else:
                if perf: perf.record("linesearch", perf_counter() - _t0)
                break

        energy_old = _linesearch_energy(gfu.vec)
        tau = 1.0
        accepted = False
        c1 = 1e-4
        for _ in range(30):
            _uh_vec.data = gfu.vec - tau * _w_vec
            energy_new = _linesearch_energy(_uh_vec)
            if np.isfinite(energy_new) and energy_new <= energy_old + c1 * tau * slope:
                accepted = True
                break
            tau *= 0.5
        if perf: perf.record("linesearch", perf_counter() - _t0)

        if accepted:
            gfu.vec.data = _uh_vec
        else:
            # Linesearch failed — take a small gradient step to avoid
            # stagnation from repeating the same solution.
            r_max = np.max(np.abs(r_np[free_dofs]))
            if r_max > 1e-30:
                scale = 0.01 * h_contact / r_max
                uh_np = _uh_vec.FV().NumPy()
                uh_np[:] = gfu.vec.FV().NumPy()
                uh_np[free_dofs] -= scale * r_np[free_dofs]
                gfu.vec.data = _uh_vec

    if not np.isfinite(gfu.vec.FV().NumPy()).all():
        gfu.vec.FV().NumPy()[:] = _u_backup
        converged = False

    return nit + 1, converged, gn_out, normals_out, active_out


# ══════════════════════════════════════════════════════════════════════════════
# 7.  GRIDFUNCTION FIELDS FOR VTK OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
# Scalar spaces for contact observables
fes_scalar = H1(mesh, order=1)
gf_penetration     = GridFunction(fes_scalar)
gf_contact_active  = GridFunction(fes_scalar)
gf_contact_rx      = GridFunction(fes)              # vector field (reactions)


def update_contact_fields(gn, normals, active, f_con):
    """Push contact data into GridFunctions for VTK export."""
    pen_np = gf_penetration.vec.FV().NumPy()
    pen_np[:] = 0.0
    act_np = gf_contact_active.vec.FV().NumPy()
    act_np[:] = 0.0

    if np.any(active):
        act = np.where(active)[0]
        pen_np[slave_verts[act]] = gn[act]
        act_np[slave_verts[act]] = 1.0

    gf_contact_rx.vec.FV().NumPy()[:] = f_con


# Build VTK field list based on vtk_fields setting
_vtk_coefs = [gfu, vm_cauchy, gf_contact_active, gf_penetration, gf_contact_rx]
_vtk_names = ["displacement", "vm_cauchy", "contact_active", "penetration", "contact_reaction"]

if vtk_fields in ("standard", "full") and stress_cauchy is not None:
    _vtk_coefs.insert(1, stress_cauchy)
    _vtk_names.insert(1, "stress_cauchy")

if vtk_fields == "full":
    for cf, nm in [(stress_1piola, "stress_1piola"),
                   (stress_mandel, "stress_mandel"),
                   (vm_mandel, "vm_mandel")]:
        if cf is not None:
            _vtk_coefs.append(cf)
            _vtk_names.append(nm)

# Plastic strain field for VTK export — mass-matrix L2 projection from IRS to H1
# (IRS GridFunctions are only defined at Gauss points, not at VTK nodes).
# Pattern from NGSolve i-tutorial unit-6.3-plasticity/plasticity.ipynb:
#   M_draw * u_draw = integral(gf_irs * v_draw * irs_dx)
# IMPORTANT: use L2 (not H1) for the target space.  H1 enforces inter-element
# continuity, which causes Gibbs oscillations (negative values) when projecting
# a sharp-edged field like epcum.  L2 gives element-local projection.
if plastic:
    _fes_epcum_draw = L2(mesh, order=0)
    _pd_ep, _qd_ep = _fes_epcum_draw.TnT()
    _M_epcum = BilinearForm(_pd_ep * _qd_ep * irs_dx, symmetric=True).Assemble()
    _M_epcum_inv = _M_epcum.mat.Inverse()
    gf_epcum_irs = GridFunction(fes_ir)       # source (GP values)
    gf_epcum_vtk = GridFunction(_fes_epcum_draw)  # target (element values)
    _vtk_coefs.append(gf_epcum_vtk)
    _vtk_names.append("epcum")

vtk = VTKOutput(
    mesh,
    coefs=_vtk_coefs,
    names=_vtk_names,
    filename=os.path.join(out_dir, "contact_potato"),
    subdivision=0, order=1,
) if plot > 0 else None


# ══════════════════════════════════════════════════════════════════════════════
# 8.  TIME-STEPPING LOOP
# ══════════════════════════════════════════════════════════════════════════════
dt_base = 1.0 / nsteps
disp_per_unit = np.array([12.0, 0.0, 0.0])   # total prescribed displacement

print(f"\n{'='*60}")
print(f"  ContactPotato NGSolve — mesh {n}x{n}x{n}")
print(f"  E={E_val}, nu={nu_val}, kn={kn:.4f}")
if solver == "newton":
    print(f"  {nsteps} steps, solver=newton, max_iter={max_iter}, gtol={gtol:.0e}, linear={linear_solver}")
elif solver in ("newton-cg", "trust-constr"):
    hess_type = "full (with curvature)" if full_hessian else "simple (n⊗n)"
    print(f"  {nsteps} steps, solver={solver}, max_iter={max_iter}, gtol={gtol:.0e}, hess={hess_type}")
else:
    print(f"  {nsteps} steps, solver={solver}, max_iter={max_iter}, gtol={gtol:.0e}")
perf_opts = []
if taskmanager: perf_opts.append("TaskManager")
if realcompile: perf_opts.append("realcompile")
if profile:     perf_opts.append("profile")
if plastic:
    ct_label = "FD full" if consistent_tangent else "off"
    print(f"  Plasticity: J2 von Mises, My0={My0:.4f}, H={H_hard:.4f}, m={m_hard:.1f}, tangent={ct_label}")
print(f"  Cutback: max_level={max_cutback} (up to {2**max_cutback}x refinement)")
print(f"  VTK: every {plot} steps ({vtk_fields}), "
      f"perf: {', '.join(perf_opts) if perf_opts else 'none'}")
print(f"{'='*60}\n")

t_wall_start = perf_counter()

# --- Helper: compute contact forces and update fields ---
def finalize_contact_state():
    """Evaluate contact state and update output fields."""
    slave_pos = compute_slave_pos()
    gn, normals, active, _ = contact_cache.evaluate(slave_pos)
    f_con = compute_contact_forces(gn, normals, active)
    update_contact_fields(gn, normals, active, f_con)
    n_active = int(np.sum(active))
    max_pen  = -np.min(gn[active]) if n_active > 0 else 0.0
    return gn, normals, active, n_active, max_pen

# Adaptive stepping state
load = 0.0              # accumulated load fraction [0, 1]
dt = dt_base            # current sub-step size
bisect_level = 0        # current bisection depth
prev_base_step = 0      # last completed base step (for VTK/compare triggering)
total_substeps = 0      # total sub-step count (for summary)
total_cutbacks = 0      # total bisection count (for summary)
total_forced   = 0      # steps force-accepted at max cutback

# Enable multi-threaded NGSolve operations (assembly, integration, solve).
# TaskManager activates parallel element-loop assembly, integration, and
# matrix-vector products inside Apply/AssembleLinearization/Inverse.
# See: NGSolve howto_parallel.rst, py_tutorials/navierstokes.py
# NOTE: For small problems (<1000 elements), threading overhead may dominate.
#       Benefit grows with mesh size (10k+ elements).
with TaskManager() if taskmanager else nullcontext():
    while load < 1.0 - 1e-12:
        dt = min(dt, 1.0 - load)       # don't overshoot
        total_substeps += 1
        t_step = perf_counter()

        # Reset contact cache at each sub-step (force full re-projection once)
        contact_cache.reset()
        _hessp_cache["x_free"] = None   # invalidate hessp tangent/contact cache

        # --- Save displacement state before Dirichlet increment -----------
        vec = gfu.vec.FV().NumPy()
        u_step_backup = vec.copy()

        # --- Incremental Dirichlet on "top" boundary ----------------------
        disp_inc = disp_per_unit * dt
        vec[top_dofs_x] += disp_inc[0]
        vec[top_dofs_y] += disp_inc[1]
        vec[top_dofs_z] += disp_inc[2]

        # ══════════════════════════════════════════════════════════════════
        # SOLVER DISPATCH
        # ══════════════════════════════════════════════════════════════════

        if solver == "newton":
            n_iters, step_converged, _, _, _ = newton_solve()
        else:
            # ── Scipy minimize solvers ──
            SCIPY_SOLVERS = {
                "newton-cg":    ("Newton-CG",    True,  {"xtol": gtol}),
                "trust-constr": ("trust-constr", True,  {"gtol": gtol, "xtol": 1e-30}),
                "lbfgsb":       ("L-BFGS-B",     False, {"gtol": gtol, "maxls": 4000, "ftol": 0}),
            }
            method_name, uses_hessp, opts_override = SCIPY_SOLVERS[solver]
            options = {"maxiter": max_iter, **opts_override}

            x0 = vec[free_dofs].copy()
            if uses_hessp:
                result = minimize(objective, x0, method=method_name, jac=True,
                                  hessp=hessp, options=options)
            else:
                result = minimize(objective, x0, method=method_name, jac=True,
                                  options=options)
            vec[free_dofs] = result.x
            n_iters = result.nit
            step_converged = result.success

        # ══════════════════════════════════════════════════════════════════
        # CUTBACK ON NON-CONVERGENCE
        # ══════════════════════════════════════════════════════════════════

        if not step_converged and bisect_level < max_cutback:
            vec[:] = u_step_backup
            dt /= 2
            bisect_level += 1
            total_cutbacks += 1
            dt_step = perf_counter() - t_step
            print(f"  >> Non-converged ({n_iters} iters), "
                  f"cutback {bisect_level}/{max_cutback} "
                  f"(dt = dt_base/{2**bisect_level})  t={dt_step:.1f}s")
            continue

        # ══════════════════════════════════════════════════════════════════
        # STEP ACCEPTED (converged OR max cutback reached)
        # ══════════════════════════════════════════════════════════════════

        load += dt
        forced_accept = not step_converged
        if forced_accept:
            total_forced += 1
            print(f"  >> Max cutback {max_cutback} reached — "
                  f"accepting step (load={load*100:.1f}%)")

        # ── Plastic history commit (unified for Newton and scipy) ────────
        if plastic:
            if solver == "newton" and step_converged:
                # Newton converged: _Fp_temp is consistent with gfu (RM ran
                # at the start of the converging iteration, before the
                # convergence check).  Direct commit is safe.
                if np.isfinite(_Fp_temp).all():
                    _Fp_conv[:] = _Fp_temp
                    _epcum_conv += _delta_epcum
            else:
                # Scipy (converged or forced), or Newton forced-accept:
                # Run final RM at current displacement to ensure Fp-
                # consistency (after Newton's last linesearch update,
                # _Fp_temp lags behind gfu.vec by one iteration).
                F_flat = _read_F_at_ips()
                _F_flat_cache = F_flat
                _Fp_temp, _delta_epcum, rm_ok = return_mapping(
                    F_flat, _Fp_conv, _epcum_conv, c10, d1, My0, H_hard, m_hard)
                if rm_ok and np.isfinite(_Fp_temp).all():
                    _write_Fp_to_gf(_Fp_temp)
                    _Fp_conv[:] = _Fp_temp
                    _epcum_conv += _delta_epcum

        # ── Finalize contact state ───────────────────────────────────────
        gn, normals, active, n_active, max_pen = finalize_contact_state()

        # ── Print step info ──────────────────────────────────────────────
        cutback_tag = f"  *cutback" if bisect_level > 0 else ""
        curr_base_step = int(load / dt_base + 1e-9)  # floor with eps (avoids banker's rounding)

        if solver == "newton":
            a_form.Apply(gfu.vec, res_vec)
            f_con = compute_contact_forces(gn, normals, active)
            res_vec.FV().NumPy()[:] += f_con
            rnorm = np.linalg.norm(res_vec.FV().NumPy()[free_dofs])

            ep_info = ""
            if plastic:
                max_epcum = np.max(_epcum_conv)
                n_plastic = np.sum(_epcum_conv > 1e-12)
                ep_info = f"  ep_max={max_epcum:.2e}  n_plast={n_plastic}"

            dt_step = perf_counter() - t_step
            print(f"Step {curr_base_step:3d}/{nsteps}  newton: nit={n_iters:3d}  "
                  f"|r|={rnorm:.2e}  active={n_active:3d}  maxpen={max_pen:.2e}"
                  f"{ep_info}  t={dt_step:.1f}s{cutback_tag}")
        else:
            dt_step = perf_counter() - t_step
            if hasattr(result, "optimality") and result.optimality is not None:
                grad_norm = result.optimality
            elif result.jac is not None and np.size(result.jac) > 0:
                grad_norm = np.linalg.norm(result.jac)
            else:
                grad_norm = float("nan")

            ep_info = ""
            if plastic:
                max_epcum = np.max(_epcum_conv)
                n_plastic = np.sum(_epcum_conv > 1e-12)
                ep_info = f"  ep_max={max_epcum:.2e}  n_plast={n_plastic}"

            print(f"Step {curr_base_step:3d}/{nsteps}  {solver}: nit={result.nit:2d}  "
                  f"nfev={result.nfev:3d}  |grad|={grad_norm:.2e}  "
                  f"active={n_active:3d}  maxpen={max_pen:.2e}"
                  f"{ep_info}  t={dt_step:.1f}s{cutback_tag}")
            if not result.success:
                print(f"  >> {result.message}")

        # ── Step size recovery + VTK/compare at base-step boundaries ─────
        if curr_base_step > prev_base_step:
            # Crossed a base-step boundary → try to recover step size
            if bisect_level > 0:
                dt = min(dt * 2, dt_base)
                bisect_level = max(bisect_level - 1, 0)

            # VTK snapshot at base-step boundaries
            if plot > 0 and curr_base_step % plot == 0:
                if perf: _t0 = perf_counter()
                if plastic:
                    gf_epcum_irs.vec.FV().NumPy()[:] = _epcum_conv
                    _f_ep = LinearForm(gf_epcum_irs * _qd_ep * irs_dx)
                    _f_ep.Assemble()
                    gf_epcum_vtk.vec.data = _M_epcum_inv * _f_ep.vec
                vtk.Do(time=curr_base_step * dt_base)
                if perf: perf.record("vtk_output", perf_counter() - _t0)

            # Comparison output at base-step boundaries
            if compare:
                np.save(os.path.join(out_dir, f"u_step{curr_base_step:04d}.npy"),
                        gfu.vec.FV().NumPy().copy())
                a_form.Apply(gfu.vec, res_vec)
                f_top = res_vec.FV().NumPy()
                rx = np.sum(f_top[top_dofs_x])
                ry = np.sum(f_top[top_dofs_y])
                rz = np.sum(f_top[top_dofs_z])
                with open(os.path.join(out_dir, "reactions.csv"), "a") as fout:
                    if curr_base_step == 1:
                        fout.write("step,time,rx,ry,rz\n")
                    fout.write(f"{curr_base_step},{curr_base_step*dt_base:.6f},"
                               f"{rx:.10e},{ry:.10e},{rz:.10e}\n")

            prev_base_step = curr_base_step

        # --- Profiling step summary ----------------------------------------
        if perf:
            perf.record("step_total", perf_counter() - t_step)
            print(perf.step_summary(total_substeps))
            perf.reset_step()

t_total = perf_counter() - t_wall_start
cutback_info = f", {total_cutbacks} cutbacks" if total_cutbacks > 0 else ""
forced_info = f", {total_forced} forced" if total_forced > 0 else ""
print(f"\nTotal wall time: {t_total:.1f} s "
      f"({total_substeps} sub-steps{cutback_info}{forced_info})")
print(f"Output directory: {out_dir}")
if perf:
    print(perf.final_summary())

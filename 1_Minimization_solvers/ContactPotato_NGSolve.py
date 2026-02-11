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
  evaluated dynamically at each iteration with locked projections for
  energy consistency.
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
n           = 10                # mesh density (n x n x n hex elements)

# Material (compressible neo-Hookean)
E_val       = 0.05          # Young's modulus
nu_val      = 0.3           # Poisson ratio

# Contact
kn_factor   = 20.0          # kn = kn_factor * E / h  (Wriggers 2006)

# Solver: "newton", "newton-cg", "trust-constr", "lbfgsb"
solver      = "newton"
nsteps      = 100           # number of load steps
max_iter    = 200            # max iterations per step
gtol        = 1e-8          # gradient/residual tolerance

# Hessian
full_hessian = True         # include curvature term dn/dx_s in contact Hessian

# Output
compare     = False         # write comparison outputs (u arrays, reactions CSV)
plot        = 1            # VTK export every N steps (0 = off)
vtk_fields  = "minimal"    # "minimal" | "standard" | "full" (see Section 7)

# Performance
taskmanager = "auto"       # True/False/"auto" — auto enables for n >= 30 (27k+ elements)
realcompile = "auto"       # True/False/"auto" — auto enables for nsteps >= 20
profile     = False         # built-in per-operation timing (prints breakdown per step)
# ──────────────────────────────────────────────────────────────────────────────

# ── Profiling instrumentation ────────────────────────────────────────────────
class PerfCounters:
    """Lightweight per-operation timing accumulator."""
    def __init__(self):
        self.data = {}
        self.step_data = {}

    def reset_step(self):
        self.step_data = {}

    def record(self, name, duration):
        self.data.setdefault(name, []).append(duration)
        self.step_data.setdefault(name, []).append(duration)

    def step_summary(self, step):
        if not self.step_data:
            return ""
        lines = [f"  [PROFILE] Step {step} breakdown:"]
        for name, times in sorted(self.step_data.items(), key=lambda x: -sum(x[1])):
            total = sum(times)
            count = len(times)
            lines.append(f"    {name:30s}  {total:8.3f}s  ({count:4d} calls, "
                         f"avg {total/count*1000:7.2f}ms)")
        lines.append(f"    {'STEP TOTAL':30s}  {sum(sum(v) for v in self.step_data.values()):8.3f}s")
        return "\n".join(lines)

    def final_summary(self):
        if not self.data:
            return ""
        total_all = sum(sum(v) for v in self.data.values())
        lines = ["\n" + "=" * 70, "  PROFILING SUMMARY (all steps)", "=" * 70]
        for name, times in sorted(self.data.items(), key=lambda x: -sum(x[1])):
            total = sum(times)
            count = len(times)
            pct = 100.0 * total / total_all if total_all > 0 else 0
            lines.append(f"  {name:30s}  {total:8.2f}s  ({pct:5.1f}%)  "
                         f"[{count} calls, avg {total/count*1000:.2f}ms]")
        lines.append(f"  {'TOTAL':30s}  {total_all:8.2f}s")
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
J_sym  = Det(F_sym)
I1_sym = Trace(F_sym.trans * F_sym)
psi_sym = c10 * (I1_sym - 3 - 2*log(J_sym)) + d1 * log(J_sym)**2

a_form = BilinearForm(fes)
a_form += Variation(psi_sym.Compile(realcompile=realcompile, wait=True) * dx)

# --- Stress CFs for VTK output (built only if needed by vtk_fields) ------
# Only compile what's needed — matrix-valued stress tensors are expensive.
F_gfu   = Id(3) + Grad(gfu)
J_gfu   = Det(F_gfu)
F_inv_T = Inv(F_gfu).trans

stress_1piola = stress_cauchy = stress_mandel = None
vm_cauchy = vm_mandel = None

if plot > 0:
    # Cauchy + vm_cauchy: needed by "minimal", "standard", "full"
    stress_cauchy = ((1/J_gfu) * (2*c10*(F_gfu * F_gfu.trans - Id(3))
                     + 2*d1*log(J_gfu)*Id(3))).Compile()
    s_dev_cauchy = stress_cauchy - (1.0/3.0) * Trace(stress_cauchy) * Id(3)
    vm_cauchy = sqrt(1.5 * InnerProduct(s_dev_cauchy, s_dev_cauchy)).Compile()

    if vtk_fields == "full":
        # 1st Piola-Kirchhoff, Mandel, vm_mandel
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

# Dense surface sampling for KD-tree candidate selection
sample_t  = np.linspace(0, 1, 50)
surf_pts, surf_pids = [], []
for pid, patch in enumerate(patches):
    for u_s in sample_t:
        for v_s in sample_t:
            surf_pts.append(patch.Grg0(np.array([u_s, v_s], dtype=np.float64)))
            surf_pids.append(pid)
surf_pts  = np.asarray(surf_pts,  dtype=np.float64)
surf_pids = np.asarray(surf_pids, dtype=np.int32)
surf_kdtree = cKDTree(surf_pts)

print(f"Potato: {npatches} patches, {len(surf_pts)} surface samples for KD-tree")

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


def _evaluate_projection(patch, t, xs, compute_hessian=False):
    """Evaluate gap, normal, and optionally dndxs at a projection point.

    This is the core computation shared by:
    - _project_single (after TR finds the best patch)
    - ContactCache.evaluate (when reusing cached patch/params)
    - Newton inner loop (recomputing during iterations)

    Parameters
    ----------
    patch : GregoryPatch
        The patch containing the projection point
    t : ndarray (2,)
        Parametric coordinates on the patch
    xs : ndarray (3,)
        Slave node position
    compute_hessian : bool
        If True, compute dn/dx_s for contact Hessian

    Returns
    -------
    gn : float
        Gap value (negative = penetration)
    nor : ndarray (3,)
        Unit normal at projection point
    dndxs : ndarray (3, 3) or None
        dn/dx_s matrix (only if compute_hessian=True and successful)
    """
    if compute_hessian:
        # Get surface point and derivatives for Hessian
        xc, dxcdt, d2xcd2t = patch.Grg(t, deriv=2)
        n_raw = patch.D3Grg(t)
        nor = n_raw / np.linalg.norm(n_raw)

        # Compute dn/dx_s via implicit function theorem
        # Projection minimizes |xs - xc(t)|^2, so at minimum:
        #   f(t) = -2*(xs - xc)·(dxc/dt) = 0
        # Implicit diff: dt/dxs = -(df/dt)^{-1} @ (df/dxs)
        delta = xs - xc
        dfdt = -2 * np.tensordot(delta, d2xcd2t, axes=1) + 2 * (dxcdt.T @ dxcdt)
        dfdxs = -2 * dxcdt.T
        try:
            dtdxs = np.linalg.solve(-dfdt, dfdxs)  # (2, 3)
            dndt = patch.dndt(t)  # (3, 2)
            dndxs = dndt @ dtdxs  # (3, 3)
        except np.linalg.LinAlgError:
            dndxs = None  # Singular — skip curvature term
    else:
        xc = patch.Grg(t, deriv=0)
        n_raw = patch.D3Grg(t)
        nor = n_raw / np.linalg.norm(n_raw)
        dndxs = None

    gn = (xs - xc) @ nor
    return gn, nor, dndxs


def _project_single(xsi, distances, sorted_idx, kd_pids, compute_hessian=False):
    """TR projection for a single slave node.

    Parameters
    ----------
    xsi : ndarray (3,)
        Slave node position
    distances : ndarray (npatches,)
        Distance from slave to each patch bounding sphere center
    sorted_idx : ndarray
        Indices that sort patches by distance
    kd_pids : ndarray
        Candidate patch IDs from KD-tree
    compute_hessian : bool
        If True, compute dn/dx_s for contact Hessian

    Returns
    -------
    gn : float
        Gap value (negative = penetration)
    nor : ndarray (3,)
        Unit normal at projection point
    patch_id : int
        Patch ID (-1 if projection failed)
    t : ndarray (2,) or None
        Parametric coordinates
    dndxs : ndarray (3,3) or None
        dn/dx_s matrix (only if compute_hessian=True and projection succeeded)
    """
    radius_factor = RADIUS_FACTOR
    for attempt in range(2):
        base_idx = min(BASE_NCAND - 1, npatches - 1)
        radius   = distances[sorted_idx[base_idx]] * radius_factor
        cands_bs = np.nonzero(distances <= radius)[0]
        if cands_bs.size < MIN_NCAND:
            cands_bs = sorted_idx[:min(MIN_NCAND, npatches)]

        merged = np.unique(np.concatenate(
            [cands_bs.astype(np.int32), kd_pids.astype(np.int32)]
        ))
        if merged.size > MAX_NCAND:
            merged = merged[np.argsort(distances[merged])[:MAX_NCAND]]

        best_patch, t1, t2, _ = gb.find_projection_tr_multi(
            ctrlpts_all, xsi, merged.astype(np.int32), radii, eps,
            TR_INIT, TR_MIN, TR_MAX
        )
        if int(best_patch) >= 0:
            pid = int(best_patch)
            t   = np.array([t1, t2], dtype=np.float64)
            gn, nor, dndxs = _evaluate_projection(
                patches[pid], t, xsi, compute_hessian=compute_hessian
            )
            return gn, nor, pid, t, dndxs

        radius_factor *= 2.0
    return np.inf, np.zeros(3), -1, None, None


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
        self.tol_reuse   = 0.05      # reuse if ||Δx|| < tol per node

    def reset(self):
        """Call at the start of each load step."""
        self.prev_pos = None

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
        n_slave = slave_pos.shape[0]
        gn      = np.full(n_slave, np.inf)
        normals = np.zeros((n_slave, 3))
        dndxs_all = np.zeros((n_slave, 3, 3)) if compute_hessian else None

        # Decide per-node: reuse cache or full re-project
        need_full = np.ones(n_slave, dtype=bool)
        if self.prev_pos is not None:
            disp = np.linalg.norm(slave_pos - self.prev_pos, axis=1)
            can_reuse = disp < self.tol_reuse

            if perf: _t0 = perf_counter()
            for i in np.where(can_reuse)[0]:
                pid = self.patch_ids[i]
                if pid < 0:
                    continue
                t = self.params[i]
                gn[i], normals[i], dndxs = _evaluate_projection(
                    patches[pid], t, slave_pos[i], compute_hessian=compute_hessian
                )
                if compute_hessian and dndxs is not None:
                    dndxs_all[i] = dndxs
                need_full[i] = False
            if perf: perf.record("contact_cached", perf_counter() - _t0)

        # Full TR projection for nodes that need it (batch C++ with OpenMP)
        idx_full = np.where(need_full)[0]
        if idx_full.size > 0:
            if perf: _t0 = perf_counter()
            if self.patch_ids is None:
                self.patch_ids = np.full(n_slave, -1, dtype=np.int32)
                self.params    = np.zeros((n_slave, 2), dtype=np.float64)

            pos_full = slave_pos[idx_full].astype(np.float64)
            pids_b, t1_b, t2_b, gn_b, nor_b, _ = project_points_tr_multi_batch(
                pos_full, xm_matrix, ctrlpts_all, radii, eps,
                TR_INIT, TR_MIN, TR_MAX,
                surf_kdtree, surf_pids, BASE_NCAND, MIN_NCAND,
                MAX_NCAND, RADIUS_FACTOR, K_SURF,
            )

            # Scatter batch results back
            for j, i in enumerate(idx_full):
                pid = int(pids_b[j])
                gn[i] = gn_b[j]
                normals[i] = nor_b[j]
                self.patch_ids[i] = pid
                if pid >= 0:
                    self.params[i] = [t1_b[j], t2_b[j]]

            # Compute Hessian data from known (patch, params) if needed
            if compute_hessian:
                for j, i in enumerate(idx_full):
                    pid = int(pids_b[j])
                    if pid < 0:
                        continue
                    t = np.array([t1_b[j], t2_b[j]], dtype=np.float64)
                    _, _, dndxs_i = _evaluate_projection(
                        patches[pid], t, slave_pos[i], compute_hessian=True
                    )
                    if dndxs_i is not None:
                        dndxs_all[i] = dndxs_i
            if perf: perf.record("contact_full_tr", perf_counter() - _t0)

        self.prev_pos = slave_pos.copy()
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

        # Assemble material tangent at new state
        a_form.AssembleLinearization(gfu.vec)

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
_cached_inv = None                # UMFPACK inverse (symbolic reuse via .Update())

# Precomputed surface data for vectorized linesearch energy evaluation.
# Populated once per Newton iteration in newton_solve(); used by _linesearch_energy().
# Since projections are locked during Newton (tol_reuse=inf), the surface points
# and normals don't change between linesearch evaluations — only slave_pos changes.
_ls_xc    = np.zeros((0, 3))     # surface projection points
_ls_nor   = np.zeros((0, 3))     # unit normals at projection points
_ls_valid = np.zeros(0, dtype=bool)  # mask: True if projection exists


def _precompute_ls_data():
    """Precompute surface points and normals for vectorized linesearch energy.

    Called once per Newton iteration after contact_cache.evaluate().
    Since projections are locked (tol_reuse=inf), the surface geometry at
    each cached (patch, params) doesn't change during linesearch — only the
    slave positions change.  Precomputing xc and nor here allows the
    linesearch energy to use fully vectorized numpy instead of a per-node
    Python loop with C++ calls.
    """
    global _ls_xc, _ls_nor, _ls_valid
    n_slave = len(slave_verts)
    _ls_xc    = np.zeros((n_slave, 3))
    _ls_nor   = np.zeros((n_slave, 3))
    _ls_valid = np.zeros(n_slave, dtype=bool)
    if contact_cache.patch_ids is None:
        return
    for i in range(n_slave):
        pid = int(contact_cache.patch_ids[i])
        if pid < 0:
            continue
        t = contact_cache.params[i]
        _ls_xc[i] = patches[pid].Grg(t, deriv=0)
        _ls_nor[i] = patches[pid].D3Grg(t)
        _ls_valid[i] = True


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
    gn, normals, active : arrays
        Final contact state (from last iteration)
    """
    newton_gtol = max(gtol, 1e-10)
    _u_backup = gfu.vec.FV().NumPy().copy()

    # Lock projections during Newton (same as scipy paths).
    # Prevents energy discontinuities from projection switching.
    saved_tol = contact_cache.tol_reuse
    contact_cache.tol_reuse = np.inf

    rnorm_prev = np.inf
    stag_count = 0
    gn_out, normals_out, active_out = None, None, None

    for nit in range(max_iter):
        # 1. Material residual
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
        if rnorm < newton_gtol:
            break

        # 5. Stagnation detection (5% decrease over 5 iterations)
        if rnorm > 0.95 * rnorm_prev:
            stag_count += 1
            if stag_count >= 5:
                break
        else:
            stag_count = 0
        rnorm_prev = rnorm

        # 6. Material tangent
        if perf: _t0 = perf_counter()
        a_form.AssembleLinearization(gfu.vec)
        if perf: perf.record("assemble_lin", perf_counter() - _t0)

        # 7. Add contact Hessian: K_con = kn * (n⊗n + g·dn/dx_s)
        if perf: _t0 = perf_counter()
        mat = a_form.mat
        for idx, g, nor, K_con in pen_data:
            v = int(slave_verts[idx])
            dofs = [v, v + nv, v + 2*nv]
            for a in range(3):
                for b in range(3):
                    mat[dofs[a], dofs[b]] = mat[dofs[a], dofs[b]] + K_con[a, b]
        if perf: perf.record("contact_hess_asm", perf_counter() - _t0)

        # 8. Solve K·Δu = -r (reuse symbolic factorization via .Update())
        if perf: _t0 = perf_counter()
        global _cached_inv
        solve_ok = False
        try:
            if _cached_inv is None:
                _cached_inv = mat.Inverse(fes.FreeDofs(), inverse="umfpack")
            else:
                _cached_inv.Update()
            _w_vec.data = _cached_inv * res_vec
            if np.isfinite(_w_vec.FV().NumPy()).all():
                solve_ok = True
            else:
                _cached_inv = None  # invalidate on NaN
        except Exception:
            _cached_inv = None  # invalidate on failure
        if perf: perf.record("umfpack_solve", perf_counter() - _t0)

        if not solve_ok:
            r_max = np.max(np.abs(r_np[free_dofs]))
            if r_max > 1e-30:
                scale = 0.1 * h_contact / r_max
                _w_vec.FV().NumPy()[:] = scale * r_np
            else:
                break

        # 9. Armijo linesearch
        if perf: _t0 = perf_counter()
        w_free = _w_vec.FV().NumPy()[free_dofs]
        slope = -np.dot(r_np[free_dofs], w_free)  # φ'(0) = -∇E·w

        if slope >= 0:
            # Not a descent direction — use gradient step
            r_max = np.max(np.abs(r_np[free_dofs]))
            if r_max > 1e-30:
                scale = 0.1 * h_contact / r_max
                _w_vec.FV().NumPy()[:] = scale * r_np
                w_free = _w_vec.FV().NumPy()[free_dofs]
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

    # Restore cache tolerance
    contact_cache.tol_reuse = saved_tol

    if not np.isfinite(gfu.vec.FV().NumPy()).all():
        gfu.vec.FV().NumPy()[:] = _u_backup

    return nit + 1, gn_out, normals_out, active_out


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
dt = 1.0 / nsteps
disp_increment = np.array([12.0, 0.0, 0.0]) * dt   # prescribed sliding

print(f"\n{'='*60}")
print(f"  ContactPotato NGSolve — mesh {n}x{n}x{n}")
print(f"  E={E_val}, nu={nu_val}, kn={kn:.4f}")
if solver == "newton":
    print(f"  {nsteps} steps, solver=newton, max_iter={max_iter}, gtol={gtol:.0e}")
elif solver in ("newton-cg", "trust-constr"):
    hess_type = "full (with curvature)" if full_hessian else "simple (n⊗n)"
    print(f"  {nsteps} steps, solver={solver}, max_iter={max_iter}, gtol={gtol:.0e}, hess={hess_type}")
else:
    print(f"  {nsteps} steps, solver={solver}, max_iter={max_iter}, gtol={gtol:.0e}")
perf_opts = []
if taskmanager: perf_opts.append("TaskManager")
if realcompile: perf_opts.append("realcompile")
if profile:     perf_opts.append("profile")
print(f"  VTK: every {plot} steps ({vtk_fields}), "
      f"perf: {', '.join(perf_opts) if perf_opts else 'none'}")
print(f"{'='*60}\n")

t_wall_start = perf_counter()

# Enable multi-threaded NGSolve operations (assembly, integration, solve).
# TaskManager activates parallel element-loop assembly, integration, and
# matrix-vector products inside Apply/AssembleLinearization/Inverse.
# See: NGSolve howto_parallel.rst, py_tutorials/navierstokes.py
# NOTE: For small problems (<1000 elements), threading overhead may dominate.
#       Benefit grows with mesh size (10k+ elements).
with TaskManager() if taskmanager else nullcontext():
    for step in range(1, nsteps + 1):
        t_step = perf_counter()

        # Reset contact cache at each load step (force full re-projection once)
        contact_cache.reset()

        # --- Incremental Dirichlet on "top" boundary ----------------------
        vec = gfu.vec.FV().NumPy()
        vec[top_dofs_x] += disp_increment[0]
        vec[top_dofs_y] += disp_increment[1]
        vec[top_dofs_z] += disp_increment[2]

        # ══════════════════════════════════════════════════════════════════
        # SOLVER DISPATCH
        # ══════════════════════════════════════════════════════════════════

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

        if solver == "newton":
            # ── Newton solver (dynamic contact, no active-set loop) ──
            n_newton, _, _, _ = newton_solve()

            # Finalize contact state at converged solution
            gn, normals, active, n_active, max_pen = finalize_contact_state()

            # Compute final residual norm for reporting
            a_form.Apply(gfu.vec, res_vec)
            f_con = compute_contact_forces(gn, normals, active)
            res_vec.FV().NumPy()[:] += f_con
            rnorm = np.linalg.norm(res_vec.FV().NumPy()[free_dofs])

            dt_step  = perf_counter() - t_step
            print(f"Step {step:3d}/{nsteps}  newton: nit={n_newton:3d}  "
                  f"|r|={rnorm:.2e}  active={n_active:3d}  maxpen={max_pen:.2e}  "
                  f"t={dt_step:.1f}s")

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
            # Lock contact projections during optimization: always reuse the
            # projection points found on the first objective() call after
            # reset().  Prevents energy discontinuities from cache-switching
            # mid-linesearch (full re-projection can find a different patch,
            # violating Wolfe conditions).
            saved_tol_reuse = contact_cache.tol_reuse
            contact_cache.tol_reuse = np.inf
            if uses_hessp:
                result = minimize(objective, x0, method=method_name, jac=True,
                                  hessp=hessp, options=options)
            else:
                result = minimize(objective, x0, method=method_name, jac=True,
                                  options=options)
            contact_cache.tol_reuse = saved_tol_reuse
            vec[free_dofs] = result.x

            # Finalize contact state and get metrics
            _, _, _, n_active, max_pen = finalize_contact_state()
            dt_step   = perf_counter() - t_step
            # trust-constr: result.jac is constraint Jacobians (empty for
            # unconstrained); use result.optimality (||∇f||_∞) instead.
            # Other methods: result.jac is the objective gradient.
            if hasattr(result, "optimality") and result.optimality is not None:
                grad_norm = result.optimality
            elif result.jac is not None and np.size(result.jac) > 0:
                grad_norm = np.linalg.norm(result.jac)
            else:
                grad_norm = float("nan")

            # Print result
            print(f"Step {step:3d}/{nsteps}  {solver}: nit={result.nit:2d}  "
                  f"nfev={result.nfev:3d}  |grad|={grad_norm:.2e}  "
                  f"active={n_active:3d}  maxpen={max_pen:.2e}  t={dt_step:.1f}s")
            if not result.success:
                print(f"  >> {result.message}")

        # --- VTK snapshot --------------------------------------------------
        if plot > 0 and step % plot == 0:
            if perf: _t0 = perf_counter()
            vtk.Do(time=step * dt)
            if perf: perf.record("vtk_output", perf_counter() - _t0)

        # --- Comparison output ---------------------------------------------
        if compare:
            np.save(os.path.join(out_dir, f"u_step{step:04d}.npy"),
                    gfu.vec.FV().NumPy().copy())
            # Reaction forces on top boundary (material internal forces only,
            # contact forces are not included since top boundary is not in contact)
            a_form.Apply(gfu.vec, res_vec)
            f_top = res_vec.FV().NumPy()
            rx = np.sum(f_top[top_dofs_x])
            ry = np.sum(f_top[top_dofs_y])
            rz = np.sum(f_top[top_dofs_z])
            with open(os.path.join(out_dir, "reactions.csv"), "a") as fout:
                if step == 1:
                    fout.write("step,time,rx,ry,rz\n")
                fout.write(f"{step},{step*dt:.6f},{rx:.10e},{ry:.10e},{rz:.10e}\n")

        # --- Profiling step summary ----------------------------------------
        if perf:
            perf.record("step_total", perf_counter() - t_step)
            print(perf.step_summary(step))
            perf.reset_step()

t_total = perf_counter() - t_wall_start
print(f"\nTotal wall time: {t_total:.1f} s")
print(f"Output directory: {out_dir}")
if perf:
    print(perf.final_summary())

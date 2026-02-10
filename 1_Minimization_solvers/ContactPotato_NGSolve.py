"""
ContactPotato_NGSolve.py — NGSolve hyperelastic contact simulation
==================================================================
Single self-contained script simulating a hyperelastic block sliding over a
rigid "potato" body.  ALL FE computations use the NGSolve API (mesh, FE space,
hyperelastic energy via Variation, Dirichlet BCs).  Contact detection reuses
the existing Gregory-patch / trust-region projection backend.

Solvers
-------
- **newton**: Full Newton with active-set iteration for contact. Uses the
  material tangent from NGSolve plus contact Hessian (kn * n ⊗ n). Optionally
  includes the curvature term (dn/dx_s) via --full_hessian flag (experimental,
  can cause matrix singularity). Typically converges in 10-25 iterations per
  load step during contact.
- **newton-cg**: Newton-CG via scipy.optimize.minimize with Hessian-vector
  product. More robust than direct Newton for ill-conditioned problems.
  Supports --full_hessian for curvature term without matrix singularity risk.
- **lbfgsb**: L-BFGS-B quasi-Newton via scipy.optimize.minimize. Uses cached
  TR projections for efficiency (~500 iterations per step).

Usage
-----
    python ContactPotato_NGSolve.py --mesh 10 --solver newton
    python ContactPotato_NGSolve.py --mesh 10 --solver lbfgsb --max_iter 500
    python ContactPotato_NGSolve.py --mesh 5  --solver newton --compare
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1.  IMPORTS & CLI
# ══════════════════════════════════════════════════════════════════════════════
import argparse, os, sys, pickle
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

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="NGSolve ContactPotato simulation")
parser.add_argument("--mesh",       type=int,   default=20,      help="mesh density n (nxnxn hex)")
parser.add_argument("--solver",     type=str,   default="newton",
                    choices=["newton", "newton-cg", "trust-constr", "lbfgsb"],
                    help="solver: newton, newton-cg, trust-constr, or lbfgsb")
parser.add_argument("--E",          type=float, default=0.05,    help="Young's modulus")
parser.add_argument("--nu",         type=float, default=0.3,     help="Poisson ratio")
parser.add_argument("--kn_factor",  type=float, default=20.0,    help="kn = kn_factor * E / h (mesh-dependent, Wriggers 2006)")
parser.add_argument("--nsteps",     type=int,   default=100,     help="number of load steps")
parser.add_argument("--max_iter",   type=int,   default=50,      help="max iterations per step (Newton inner or L-BFGS-B)")
parser.add_argument("--max_as",     type=int,   default=5,       help="max active-set iterations (Newton only)")
parser.add_argument("--gtol",       type=float, default=1e-8,    help="gradient/residual tolerance for convergence")
parser.add_argument("--full_hessian", action="store_true",       help="use full contact Hessian with curvature term (experimental)")
parser.add_argument("--compare",    action="store_true",         help="write comparison outputs")
parser.add_argument("--plot",       type=int,   default=1,       help="VTK export every N steps (0=off)")
parser.add_argument("--taskmanager", action="store_true",        help="enable NGSolve TaskManager (parallel assembly, beneficial for large meshes)")
parser.add_argument("--realcompile", action="store_true",        help="use realcompile=True for C++ JIT of energy form (startup cost, faster iterations)")
args = parser.parse_args()

n       = args.mesh
E_val   = args.E
nu_val  = args.nu
h_contact = 4.0 / n              # element edge length at contact surface ([-2,2]^3 block)
kn      = args.kn_factor * E_val / h_contact   # Wriggers (2006): kn ~ alpha * E / h
nsteps  = args.nsteps
max_iter = args.max_iter

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
a_form += Variation(psi_sym.Compile(realcompile=args.realcompile, wait=True) * dx)

# Energy CF defined on gfu (for VTK stress output)
F_gfu  = Id(3) + Grad(gfu)
J_gfu  = Det(F_gfu)
I1_gfu = Trace(F_gfu.trans * F_gfu)

# --- Stress CFs for VTK output -------------------------------------------
# 1st Piola-Kirchhoff stress: P = dW/dF = 2*c10*(F - F^{-T}) + 2*d1*ln(J)*F^{-T}
# Cauchy stress: sigma = (1/J) * P * F^T = (1/J)[2*c10*(F*F^T - I) + 2*d1*ln(J)*I]
# Mandel stress: M = F^T * P (work conjugate to velocity gradient in material frame)
F_inv_T = Inv(F_gfu).trans
stress_1piola = (2*c10*(F_gfu - F_inv_T) + 2*d1*log(J_gfu)*F_inv_T).Compile()
stress_cauchy = ((1/J_gfu) * (2*c10*(F_gfu * F_gfu.trans - Id(3)) + 2*d1*log(J_gfu)*Id(3))).Compile()
stress_mandel = (F_gfu.trans * stress_1piola).Compile()

# Von Mises from Cauchy stress
s_dev_cauchy = stress_cauchy - (1.0/3.0) * Trace(stress_cauchy) * Id(3)
vm_cauchy = sqrt(1.5 * InnerProduct(s_dev_cauchy, s_dev_cauchy)).Compile()

# Von Mises from Mandel stress
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

# DOF indices for top boundary (x-component only, for prescribed sliding)
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
BASE_NCAND = 12 #24
MIN_NCAND  = 5 #10
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
# Between consecutive L-BFGS-B evaluations the slave positions barely move,
# so we can reuse the previous projection result (patch id + parametric coords)
# and only recompute gap/normal cheaply via Grg + D3Grg (0.006 ms/node)
# instead of a full TR search (5 ms/node).

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

        # Full TR projection for nodes that need it
        idx_full = np.where(need_full)[0]
        if idx_full.size > 0:
            pos_full = slave_pos[idx_full]
            dist_mat = np.linalg.norm(
                xm_matrix[:, None, :] - pos_full[None, :, :], axis=2
            )
            for j, i in enumerate(idx_full):
                xsi        = slave_pos[i].astype(np.float64)
                distances  = dist_mat[:, j]
                sorted_idx = np.argsort(distances)
                _, idx_kd  = surf_kdtree.query(xsi, k=min(K_SURF, len(surf_pts)))
                kd_pids    = np.unique(surf_pids[np.atleast_1d(idx_kd)])
                gn[i], normals[i], pid, t, dndxs = _project_single(
                    xsi, distances, sorted_idx, kd_pids, compute_hessian=compute_hessian
                )
                if self.patch_ids is None:
                    self.patch_ids = np.full(n_slave, -1, dtype=np.int32)
                    self.params    = np.zeros((n_slave, 2), dtype=np.float64)
                self.patch_ids[i] = pid
                if t is not None:
                    self.params[i] = t
                if compute_hessian and dndxs is not None:
                    dndxs_all[i] = dndxs

        self.prev_pos = slave_pos.copy()
        active = gn < 0
        return gn, normals, active, dndxs_all

contact_cache = ContactCache()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  OBJECTIVE FUNCTION  (energy + gradient for scipy)
# ══════════════════════════════════════════════════════════════════════════════
res_vec = gfu.vec.CreateVector()       # scratch vector for Apply


def compute_contact_forces(gn, normals, active):
    """Assemble nodal contact penalty forces into a full DOF vector.

    f_con[v]    = kn * gn * n_x   for each active (penetrating) slave node
    f_con[v+nv] = kn * gn * n_y
    f_con[v+2nv]= kn * gn * n_z

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
            slave_pos, compute_hessian=args.full_hessian
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
            if args.full_hessian and dndxs_all is not None and np.any(dndxs_all[i]):
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
# 6b. NEWTON SOLVER WITH ACTIVE-SET ITERATION
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

    When --full_hessian is enabled and the curvature term g*dn/dx_s makes K_con
    indefinite, we project to PSD via absolute eigenvalue filtering:
        K_con = Q |Λ| Qᵀ
    This preserves the magnitude of all curvature information while guaranteeing
    positive semi-definiteness (Li et al. 2020 / SIGGRAPH 2024 pattern).

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

    if args.full_hessian and dndxs is not None and np.any(dndxs):
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


def _total_energy_at(u_vec, saved_pids, saved_params, active_idx):
    """Evaluate total energy (material + contact) at a given state.

    Used by the linesearch to decide on step size.
    Returns np.inf for invalid states (NaN, element inversion).
    """
    E_mat = a_form.Energy(u_vec)
    if not np.isfinite(E_mat):
        return np.inf

    if len(active_idx) == 0:
        return E_mat

    vec_np = u_vec.FV().NumPy()
    if not np.isfinite(vec_np).all():
        return np.inf

    slave_pos = np.column_stack([
        X_ref[slave_verts, 0] + vec_np[slave_verts],
        X_ref[slave_verts, 1] + vec_np[slave_verts + nv],
        X_ref[slave_verts, 2] + vec_np[slave_verts + 2*nv],
    ])
    E_con = 0.0
    for j, idx in enumerate(active_idx):
        pid = int(saved_pids[j])
        t = saved_params[j]
        xs = slave_pos[idx]
        g, _, _ = _evaluate_projection(patches[pid], t, xs)
        if g < 0:
            E_con += 0.5 * kn * g**2
    return E_mat + E_con


def newton_active_set_solve():
    """Newton solver with active-set iteration and energy-based linesearch.

    Outer loop: iterate the active set (which nodes are in contact).
    Inner loop: Newton iterations with fixed active set and backtracking
    linesearch following the pattern in NGSolve's NewtonSolver.Solve().

    Includes NaN safeguards for UMFPACK near-singular cases, stagnation
    detection, and state recovery.

    Returns
    -------
    total_newton : int
        Total Newton iterations across all active-set iterations
    n_as : int
        Number of active-set iterations
    gn : ndarray
        Final gap values for all slave nodes
    normals : ndarray
        Final surface normals for all slave nodes
    active : ndarray
        Boolean mask of nodes in contact
    """
    prev_active_idx = None
    total_newton = 0
    gn_out, normals_out, active_out = None, None, None

    # Newton convergence floor: contact projection accuracy limits
    # achievable residual to ~1e-10.  Prevent infinite looping when
    # args.gtol is set very tight (e.g. 1e-18 for scipy solvers).
    newton_gtol = max(args.gtol, 1e-10)

    # Save state before this step for NaN recovery
    _u_backup = gfu.vec.FV().NumPy().copy()

    for as_iter in range(args.max_as):
        # ── Full contact evaluation (expensive TR projection, with Hessian data) ──
        contact_cache.reset()
        slave_pos = compute_slave_pos()

        # Guard: if state is corrupted, restore backup and abort
        if not np.isfinite(slave_pos).all():
            gfu.vec.FV().NumPy()[:] = _u_backup
            break

        gn_out, normals_out, active_out, dndxs_out = contact_cache.evaluate(
            slave_pos, compute_hessian=args.full_hessian
        )
        active_idx = np.where(active_out)[0]

        # Active-set convergence check
        if prev_active_idx is not None and np.array_equal(active_idx, prev_active_idx):
            break
        prev_active_idx = active_idx.copy()

        # Cache projection data for cheap Newton re-evaluation
        n_act = len(active_idx)
        if n_act > 0 and contact_cache.patch_ids is not None:
            saved_pids = contact_cache.patch_ids[active_idx].copy()
            saved_params = contact_cache.params[active_idx].copy()
        else:
            saved_pids = np.array([], dtype=np.int32)
            saved_params = np.zeros((0, 2))

        # ── Newton iterations (fixed active set) ──
        rnorm_prev = np.inf
        stagnation_count = 0

        for nit in range(args.max_iter):
            # 1. Material residual via NGSolve
            a_form.Apply(gfu.vec, res_vec)
            r_np = res_vec.FV().NumPy()

            # 2. Add contact forces and compute Hessian data
            slave_pos_now = compute_slave_pos()
            pen_data = []

            for j in range(n_act):
                pid = int(saved_pids[j])
                t = saved_params[j]
                xs = slave_pos_now[active_idx[j]]

                g, nor, dndxs = _evaluate_projection(
                    patches[pid], t, xs, compute_hessian=args.full_hessian
                )

                if g < 0:
                    K_con = compute_contact_hessian(g, nor, dndxs)
                    pen_data.append((j, g, nor, K_con))

                    v = int(slave_verts[active_idx[j]])
                    r_np[v]          += kn * g * nor[0]
                    r_np[v + nv]     += kn * g * nor[1]
                    r_np[v + 2*nv]   += kn * g * nor[2]

            # 3. Convergence check on free DOFs
            rnorm = np.linalg.norm(r_np[free_dofs])
            total_newton += 1
            if rnorm < newton_gtol:
                break

            # Stagnation detection: if residual doesn't decrease by at
            # least 10% over 3 consecutive iterations, the fixed-projection
            # Newton has exhausted its accuracy — break early.
            if rnorm > 0.9 * rnorm_prev:
                stagnation_count += 1
                if stagnation_count >= 3:
                    break
            else:
                stagnation_count = 0
            rnorm_prev = rnorm

            # 4. Material tangent (linearization at current state)
            a_form.AssembleLinearization(gfu.vec)

            # 5. Add contact Hessian
            mat = a_form.mat
            for j_local, g, nor, K_con in pen_data:
                v = int(slave_verts[active_idx[j_local]])
                dofs = [v, v + nv, v + 2*nv]
                for a in range(3):
                    for b in range(3):
                        mat[dofs[a], dofs[b]] = mat[dofs[a], dofs[b]] + K_con[a, b]

            # 6. Solve K·Δu = -r
            solve_ok = False
            try:
                inv = mat.Inverse(fes.FreeDofs(), inverse="umfpack")
                _w_vec.data = inv * res_vec
                # UMFPACK near-singular: returns NaN without exception
                if np.isfinite(_w_vec.FV().NumPy()).all():
                    solve_ok = True
            except Exception:
                pass

            if not solve_ok:
                # Fallback: scaled gradient descent with physical step size.
                # Step proportional to element size to avoid element inversion.
                r_max = np.max(np.abs(r_np[free_dofs]))
                if r_max > 1e-30:
                    scale = 0.1 * h_contact / r_max
                    _w_vec.FV().NumPy()[:] = scale * r_np
                else:
                    break

            # 7. Energy-based backtracking linesearch with NaN guard
            energy_old = _total_energy_at(
                gfu.vec, saved_pids, saved_params, active_idx
            )
            tau = 1.0
            accepted = False
            ls_tol = max(1e-14 * abs(energy_old), newton_gtol)
            for _ in range(30):  # max 30 halvings (tau down to ~1e-9)
                _uh_vec.data = gfu.vec - tau * _w_vec
                energy_new = _total_energy_at(
                    _uh_vec, saved_pids, saved_params, active_idx
                )
                if np.isfinite(energy_new) and energy_new <= energy_old + ls_tol:
                    accepted = True
                    break
                tau *= 0.5

            if accepted:
                gfu.vec.data = _uh_vec
            # else: keep current state (don't write NaN or bad state)

    # If state got corrupted despite guards, restore backup
    if not np.isfinite(gfu.vec.FV().NumPy()).all():
        gfu.vec.FV().NumPy()[:] = _u_backup

    return total_newton, as_iter + 1, gn_out, normals_out, active_out


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


vtk = VTKOutput(
    mesh,
    coefs=[gfu, stress_1piola, stress_cauchy, stress_mandel,
           vm_cauchy, vm_mandel,
           gf_contact_active, gf_penetration, gf_contact_rx],
    names=["displacement", "stress_1piola", "stress_cauchy", "stress_mandel",
           "vm_cauchy", "vm_mandel",
           "contact_active", "penetration", "contact_reaction"],
    filename=os.path.join(out_dir, "contact_potato"),
    subdivision=0, order=1,
)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  TIME-STEPPING LOOP
# ══════════════════════════════════════════════════════════════════════════════
dt = 1.0 / nsteps
disp_increment = np.array([12.0, 0.0, 0.0]) * dt   # prescribed sliding

print(f"\n{'='*60}")
print(f"  ContactPotato NGSolve — mesh {n}x{n}x{n}")
print(f"  E={E_val}, nu={nu_val}, kn={kn:.4f}")
if args.solver == "newton":
    print(f"  {nsteps} steps, solver=Newton, max_iter={max_iter}, max_as={args.max_as}, gtol={args.gtol:.0e}")
elif args.solver in ("newton-cg", "trust-constr"):
    hess_type = "full (with curvature)" if args.full_hessian else "simple (n⊗n)"
    print(f"  {nsteps} steps, solver={args.solver}, max_iter={max_iter}, gtol={args.gtol:.0e}, hess={hess_type}")
else:
    print(f"  {nsteps} steps, solver={args.solver}, max_iter={max_iter}, gtol={args.gtol:.0e}")
print(f"{'='*60}\n")

t_wall_start = perf_counter()

# Enable multi-threaded NGSolve operations (assembly, integration, solve).
# TaskManager activates parallel element-loop assembly, integration, and
# matrix-vector products inside Apply/AssembleLinearization/Inverse.
# See: NGSolve howto_parallel.rst, py_tutorials/navierstokes.py
# NOTE: For small problems (<1000 elements), threading overhead may dominate.
#       Benefit grows with mesh size (10k+ elements).
with TaskManager() if args.taskmanager else nullcontext():
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

        if args.solver == "newton":
            # ── Newton + active-set solver ──
            n_newton, n_as, gn, normals, active = newton_active_set_solve()

            # Compute contact forces for output
            f_con = compute_contact_forces(gn, normals, active)
            update_contact_fields(gn, normals, active, f_con)

            # Compute final residual norm for reporting
            a_form.Apply(gfu.vec, res_vec)
            res_vec.FV().NumPy()[:]  += f_con
            rnorm = np.linalg.norm(res_vec.FV().NumPy()[free_dofs])

            n_active = int(np.sum(active))
            max_pen  = -np.min(gn[active]) if n_active > 0 else 0.0
            dt_step  = perf_counter() - t_step
            print(f"Step {step:3d}/{nsteps}  Newton: nit={n_newton:2d} as={n_as}  "
                  f"|r|={rnorm:.2e}  active={n_active:3d}  maxpen={max_pen:.2e}  "
                  f"t={dt_step:.1f}s")

        else:
            # ── Scipy minimize solvers ──
            SCIPY_SOLVERS = {
                "newton-cg":    ("Newton-CG",    True,  {"xtol": args.gtol}),
                "trust-constr": ("trust-constr", True,  {"gtol": args.gtol, "xtol": 1e-30}),
                "lbfgsb":       ("L-BFGS-B",     False, {"gtol": args.gtol, "maxls": 40, "ftol": 0}),
            }

            method_name, uses_hessp, opts_override = SCIPY_SOLVERS[args.solver]
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
            print(f"Step {step:3d}/{nsteps}  {args.solver}: nit={result.nit:2d}  "
                  f"nfev={result.nfev:3d}  |grad|={grad_norm:.2e}  "
                  f"active={n_active:3d}  maxpen={max_pen:.2e}  t={dt_step:.1f}s")
            if not result.success:
                print(f"  >> {result.message}")

        # --- VTK snapshot --------------------------------------------------
        if args.plot > 0 and step % args.plot == 0:
            vtk.Do(time=step * dt)

        # --- Comparison output ---------------------------------------------
        if args.compare:
            np.save(os.path.join(out_dir, f"u_step{step:04d}.npy"),
                    gfu.vec.FV().NumPy().copy())
            # Reaction forces on top boundary
            a_form.Apply(gfu.vec, res_vec)
            f_top = res_vec.FV().NumPy()
            rx = np.sum(f_top[top_dofs_x])
            ry = np.sum(f_top[top_dofs_y])
            rz = np.sum(f_top[top_dofs_z])
            with open(os.path.join(out_dir, "reactions.csv"), "a") as fout:
                if step == 1:
                    fout.write("step,time,rx,ry,rz\n")
                fout.write(f"{step},{step*dt:.6f},{rx:.10e},{ry:.10e},{rz:.10e}\n")

t_total = perf_counter() - t_wall_start
print(f"\nTotal wall time: {t_total:.1f} s")
print(f"Output directory: {out_dir}")

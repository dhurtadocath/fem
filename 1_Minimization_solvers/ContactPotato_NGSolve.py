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
parser.add_argument("--mesh",       type=int,   default=10,      help="mesh density n (nxnxn hex)")
parser.add_argument("--solver",     type=str,   default="newton",
                    choices=["newton", "lbfgsb", "newton-cg", "trust-ncg", "trust-krylov",
                             "trust-constr", "bfgs", "cg", "tnc", "slsqp"],
                    help="solver: newton, newton-cg, trust-ncg, trust-krylov, trust-constr, bfgs, cg, tnc, slsqp, or lbfgsb")
parser.add_argument("--E",          type=float, default=0.05,    help="Young's modulus")
parser.add_argument("--nu",         type=float, default=0.3,     help="Poisson ratio")
parser.add_argument("--kn_factor",  type=float, default=20.0,    help="kn = kn_factor * E")
parser.add_argument("--nsteps",     type=int,   default=100,     help="number of load steps")
parser.add_argument("--max_iter",   type=int,   default=50,      help="max iterations per step (Newton inner or L-BFGS-B)")
parser.add_argument("--max_as",     type=int,   default=5,       help="max active-set iterations (Newton only)")
parser.add_argument("--gtol",       type=float, default=1e-8,    help="gradient/residual tolerance for convergence")
parser.add_argument("--full_hessian", action="store_true",       help="use full contact Hessian with curvature term (experimental)")
parser.add_argument("--compare",    action="store_true",         help="write comparison outputs")
parser.add_argument("--plot",       type=int,   default=1,       help="VTK export every N steps (0=off)")
args = parser.parse_args()

n       = args.mesh
E_val   = args.E
nu_val  = args.nu
kn      = args.kn_factor * E_val
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
a_form += Variation(psi_sym.Compile() * dx)

# Energy CF defined on gfu (for VTK stress output)
F_gfu  = Id(3) + Grad(gfu)
J_gfu  = Det(F_gfu)
I1_gfu = Trace(F_gfu.trans * F_gfu)

# --- Stress CFs for VTK output -------------------------------------------
# 1st Piola-Kirchhoff stress: P = dW/dF = 2*c10*(F - F^{-T}) + 2*d1*ln(J)*F^{-T}
# Cauchy stress: sigma = (1/J) * P * F^T = (1/J)[2*c10*(F*F^T - I) + 2*d1*ln(J)*I]
# Mandel stress: M = F^T * P (work conjugate to velocity gradient in material frame)
F_inv_T = Inv(F_gfu).trans
stress_1piola = 2*c10*(F_gfu - F_inv_T) + 2*d1*log(J_gfu)*F_inv_T
stress_cauchy = (1/J_gfu) * (2*c10*(F_gfu * F_gfu.trans - Id(3)) + 2*d1*log(J_gfu)*Id(3))
stress_mandel = F_gfu.trans * stress_1piola

# Von Mises from Cauchy stress
s_dev_cauchy = stress_cauchy - (1.0/3.0) * Trace(stress_cauchy) * Id(3)
vm_cauchy = sqrt(1.5 * InnerProduct(s_dev_cauchy, s_dev_cauchy))

# Von Mises from Mandel stress
s_dev_mandel = stress_mandel - (1.0/3.0) * Trace(stress_mandel) * Id(3)
vm_mandel = sqrt(1.5 * InnerProduct(s_dev_mandel, s_dev_mandel))

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
BASE_NCAND = 24
MIN_NCAND  = 10
MAX_NCAND  = 96
RADIUS_FACTOR = 1.5
K_SURF = 15


def _project_single(xsi, distances, sorted_idx, kd_pids):
    """TR projection for a single slave node. Returns (gn, normal, patch_id, params)."""
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
            t   = np.array([t1, t2], dtype=np.float64)
            xc  = patches[int(best_patch)].Grg(t, deriv=0)
            n_raw = patches[int(best_patch)].D3Grg(t)
            nor = n_raw / np.linalg.norm(n_raw)
            return (xsi - xc) @ nor, nor, int(best_patch), t
        radius_factor *= 2.0
    return np.inf, np.zeros(3), -1, None


def compute_contact(slave_pos):
    """Compute gap and normal for each slave node via TR projection.

    Parameters
    ----------
    slave_pos : (n_slave, 3) array of current slave-node positions

    Returns
    -------
    gn      : (n_slave,) gap values (negative = penetration)
    normals : (n_slave, 3) outward unit normals on master surface
    active  : boolean mask of penetrating nodes
    """
    n_slave = slave_pos.shape[0]
    gn      = np.full(n_slave, np.inf)
    normals = np.zeros((n_slave, 3))

    # Distance to each BS centre
    dist_mat = np.linalg.norm(
        xm_matrix[:, None, :] - slave_pos[None, :, :], axis=2
    )  # (npatches, n_slave)

    for i in range(n_slave):
        xsi        = slave_pos[i].astype(np.float64)
        distances  = dist_mat[:, i]
        sorted_idx = np.argsort(distances)

        # KD-tree candidates
        _, idx_kd  = surf_kdtree.query(xsi, k=min(K_SURF, len(surf_pts)))
        kd_pids    = np.unique(surf_pids[np.atleast_1d(idx_kd)])

        gn[i], normals[i], _, _ = _project_single(xsi, distances, sorted_idx, kd_pids)

    active = gn < 0
    return gn, normals, active


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

    def evaluate(self, slave_pos):
        """Return (gn, normals, active) using cache when possible."""
        n_slave = slave_pos.shape[0]
        gn      = np.full(n_slave, np.inf)
        normals = np.zeros((n_slave, 3))

        # Decide per-node: reuse cache or full re-project
        need_full = np.ones(n_slave, dtype=bool)
        if self.prev_pos is not None:
            disp = np.linalg.norm(slave_pos - self.prev_pos, axis=1)
            can_reuse = disp < self.tol_reuse

            for i in np.where(can_reuse)[0]:
                pid = self.patch_ids[i]
                if pid < 0:
                    continue
                t   = self.params[i]
                xc  = patches[pid].Grg(t, deriv=0)
                n_raw = patches[pid].D3Grg(t)
                nor = n_raw / np.linalg.norm(n_raw)
                gn[i]      = (slave_pos[i] - xc) @ nor
                normals[i] = nor
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
                gn[i], normals[i], pid, t = _project_single(
                    xsi, distances, sorted_idx, kd_pids
                )
                if self.patch_ids is None:
                    self.patch_ids = np.full(n_slave, -1, dtype=np.int32)
                    self.params    = np.zeros((n_slave, 2), dtype=np.float64)
                self.patch_ids[i] = pid
                if t is not None:
                    self.params[i] = t

        self.prev_pos = slave_pos.copy()
        active = gn < 0
        return gn, normals, active

contact_cache = ContactCache()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  OBJECTIVE FUNCTION  (energy + gradient for scipy)
# ══════════════════════════════════════════════════════════════════════════════
res_vec = gfu.vec.CreateVector()       # scratch vector for Apply


def objective(x_free):
    """Return (total_energy, gradient_on_free_dofs) for scipy.minimize."""
    vec = gfu.vec.FV().NumPy()
    vec[free_dofs] = x_free

    # --- Material energy + forces (NGSolve) --------------------------------
    E_mat = a_form.Energy(gfu.vec)
    a_form.Apply(gfu.vec, res_vec)
    f_mat = res_vec.FV().NumPy().copy()        # residual = dE/du

    # --- Current slave-node positions --------------------------------------
    slave_pos = np.column_stack([
        X_ref[slave_verts, 0] + vec[slave_verts],
        X_ref[slave_verts, 1] + vec[slave_verts + nv],
        X_ref[slave_verts, 2] + vec[slave_verts + 2*nv],
    ])

    # --- Contact penalty (cached TR projection) ----------------------------
    gn, normals, active = contact_cache.evaluate(slave_pos)
    E_con = 0.5 * kn * np.sum(gn[active]**2)

    f_con = np.zeros(ndof)
    if np.any(active):
        act = np.where(active)[0]
        verts_act = slave_verts[act]
        kgn = kn * gn[act]                              # (n_active,)
        f_con[verts_act]          = kgn * normals[act, 0]
        f_con[verts_act + nv]     = kgn * normals[act, 1]
        f_con[verts_act + 2*nv]   = kgn * normals[act, 2]

    f_total = f_mat + f_con
    return E_mat + E_con, f_total[free_dofs].copy()


# ══════════════════════════════════════════════════════════════════════════════
# 6a. HESSIAN-VECTOR PRODUCT FOR NEWTON-CG
# ══════════════════════════════════════════════════════════════════════════════
# Scratch vectors for Hessian-vector product
_hess_tmp = gfu.vec.CreateVector()
_hess_out = gfu.vec.CreateVector()


def hessp(x_free, p):
    """Hessian-vector product: H @ p for scipy Newton-CG.

    Computes (K_mat + K_con) @ p where:
    - K_mat: material tangent from NGSolve (linearization of hyperelastic energy)
    - K_con: contact Hessian = kn * (n ⊗ n + g * dn/dx_s) for penetrating nodes

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
    vec[free_dofs] = x_free

    # --- Material Hessian-vector product ---
    # Assemble linearization (tangent stiffness) at current state
    a_form.AssembleLinearization(gfu.vec)

    # Set up direction vector
    _hess_tmp.FV().NumPy()[:] = 0.0
    _hess_tmp.FV().NumPy()[free_dofs] = p

    # Apply material tangent: K_mat @ p
    _hess_out.data = a_form.mat * _hess_tmp
    Hp = _hess_out.FV().NumPy()[free_dofs].copy()

    # --- Contact Hessian-vector product ---
    # Get current slave positions
    slave_pos = np.column_stack([
        X_ref[slave_verts, 0] + vec[slave_verts],
        X_ref[slave_verts, 1] + vec[slave_verts + nv],
        X_ref[slave_verts, 2] + vec[slave_verts + 2*nv],
    ])

    # Get direction components for slave nodes
    p_full = np.zeros(ndof)
    p_full[free_dofs] = p
    p_slave = np.column_stack([
        p_full[slave_verts],
        p_full[slave_verts + nv],
        p_full[slave_verts + 2*nv],
    ])

    # Contact contribution using cached projection
    gn, normals, active = contact_cache.evaluate(slave_pos)

    if np.any(active):
        act_idx = np.where(active)[0]
        for i in act_idx:
            g = gn[i]
            if g >= 0:
                continue  # Not penetrating

            nor = normals[i]
            v = int(slave_verts[i])
            p_v = p_slave[i]  # (3,) direction at this node

            # Simple Hessian: kn * (n ⊗ n) @ p_v = kn * (n · p_v) * n
            Hp_contact = kn * (nor @ p_v) * nor

            # Full Hessian with curvature term (if enabled)
            if args.full_hessian and contact_cache.patch_ids is not None:
                pid = contact_cache.patch_ids[i]
                t = contact_cache.params[i]
                if pid >= 0:
                    # Compute curvature contribution: kn * g * (dn/dx_s) @ p_v
                    try:
                        patch = patches[pid]
                        xc, dxcdt, d2xcd2t = patch.Grg(t, deriv=2)
                        delta = slave_pos[i] - xc
                        dfdt = -2 * np.tensordot(delta, d2xcd2t, axes=1) + 2 * (dxcdt.T @ dxcdt)
                        dfdxs = -2 * dxcdt.T
                        dtdxs = np.linalg.solve(-dfdt, dfdxs)
                        dndt = patch.dndt(t)
                        dndxs = dndt @ dtdxs  # (3, 3)

                        # Curvature contribution: kn * g * dndxs @ p_v
                        curv_contrib = kn * g * (dndxs @ p_v)

                        # Safeguard: limit magnitude
                        curv_norm = np.linalg.norm(curv_contrib)
                        base_norm = np.linalg.norm(Hp_contact)
                        if curv_norm > 0.3 * max(base_norm, 1e-10):
                            curv_contrib *= 0.3 * base_norm / curv_norm

                        Hp_contact += curv_contrib
                    except (np.linalg.LinAlgError, ValueError):
                        pass  # Skip curvature if computation fails

            # Map back to DOF indices
            dofs_v = np.array([v, v + nv, v + 2*nv])
            # Find which DOFs are free
            for k, dof in enumerate(dofs_v):
                idx_in_free = np.searchsorted(free_dofs, dof)
                if idx_in_free < len(free_dofs) and free_dofs[idx_in_free] == dof:
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


def compute_contact_hessian(pid, t, xs, g, nor):
    """Compute the full consistent contact Hessian for a penetrating node.

    The contact energy is E_con = (1/2) * kn * g_n^2, where g_n = (x_s - x_c) · n.
    The full Hessian is:
        K_con = kn * (n ⊗ n + g_n * dn/dx_s)

    where dn/dx_s is computed via the chain rule:
        dn/dx_s = (dn/dt) @ (dt/dx_s)

    and dt/dx_s comes from implicit differentiation of the projection equations.

    Parameters
    ----------
    pid : int
        Patch ID
    t : ndarray (2,)
        Parametric coordinates of projection point
    xs : ndarray (3,)
        Current slave node position
    g : float
        Current gap value (negative for penetration)
    nor : ndarray (3,)
        Unit normal at projection point (from D3Grg, equivalent to D1 × D2)

    Returns
    -------
    K_con : ndarray (3, 3)
        Contact Hessian contribution for this node
    """
    patch = patches[pid]

    # Get surface point, first and second derivatives w.r.t. parametric coords
    # Shapes (from compute_tr_projection_batch.py):
    #   xc: (3,)
    #   dxcdt: (3, 2) - columns are tangent vectors D1, D2
    #   d2xcd2t: (3, 2, 2) - second derivatives
    xc, dxcdt, d2xcd2t = patch.Grg(t, deriv=2)

    # ── Compute dt/dx_s via implicit function theorem ──
    # The projection minimizes |x_s - x_c(t)|^2, so at the minimum:
    #   f(t, x_s) = d/dt |x_s - x_c(t)|^2 = -2 * (x_s - x_c) · dx_c/dt = 0
    #
    # Differentiating: df/dt * dt + df/dx_s * dx_s = 0
    # => dt/dx_s = -(df/dt)^{-1} @ (df/dx_s)

    delta = xs - xc  # (3,)

    # df/dt (2x2 Hessian of distance^2 w.r.t. t)
    # Using tensordot as in compute_tr_projection_batch.py:
    #   dfdt = -2 * tensordot(delta, d2xcd2t, axes=1) + 2 * (dxcdt.T @ dxcdt)
    # tensordot contracts delta (3,) with d2xcd2t (3,2,2) over first axis → (2,2)
    # dxcdt.T @ dxcdt is (2,3) @ (3,2) → (2,2)
    dfdt = -2 * np.tensordot(delta, d2xcd2t, axes=1) + 2 * (dxcdt.T @ dxcdt)

    # df/dx_s (2x3)
    dfdxs = -2 * dxcdt.T  # (2, 3)

    # dt/dx_s (2x3) via implicit function theorem
    try:
        dtdxs = np.linalg.solve(-dfdt, dfdxs)  # (2, 3)
    except np.linalg.LinAlgError:
        # Singular Hessian (degenerate projection) - fall back to n⊗n only
        return kn * np.outer(nor, nor)

    # ── Compute dn/dx_s ──
    dndt = patch.dndt(t)  # (3, 2) - dn/dt
    dndxs = dndt @ dtdxs  # (3, 3) - dn/dx_s

    # ── Full contact Hessian ──
    # K_con = kn * (n ⊗ n + g * dn/dx_s)
    #
    # The curvature term (g * dndxs) can make the Hessian indefinite when
    # |g| is large relative to the surface curvature. This causes Newton
    # to diverge or the matrix to become singular.
    #
    # Safeguard: limit the curvature contribution to avoid indefiniteness.
    # The n⊗n term has spectral norm 1. We limit |g * dndxs| relative to this.
    K_base = np.outer(nor, nor)
    K_curv = g * dndxs

    # The curvature term can make the Hessian indefinite or singular when |g| is
    # large relative to the surface curvature. Only include if --full_hessian is set.
    if args.full_hessian:
        # Safeguard: limit curvature contribution to avoid indefiniteness
        curv_norm = np.linalg.norm(K_curv, ord=2)  # Spectral norm
        max_curv_ratio = 0.3  # Limit curvature contribution to 30% of base term

        if curv_norm > max_curv_ratio:
            K_curv = K_curv * (max_curv_ratio / curv_norm)

        K_con = kn * (K_base + K_curv)
    else:
        # Simple Hessian: just n ⊗ n (more robust)
        K_con = kn * K_base

    return K_con


def newton_active_set_solve():
    """Newton solver with active-set iteration for contact.

    The outer loop iterates the active set (which nodes are in contact).
    The inner loop performs Newton iterations with fixed active set.

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

    for as_iter in range(args.max_as):
        # ── Full contact evaluation (expensive TR projection) ──
        contact_cache.reset()
        slave_pos = compute_slave_pos()
        gn_out, normals_out, active_out = contact_cache.evaluate(slave_pos)
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
        for nit in range(args.max_iter):
            # 1. Material residual via NGSolve
            a_form.Apply(gfu.vec, res_vec)
            r_np = res_vec.FV().NumPy()

            # 2. Add contact forces and compute Hessian data
            slave_pos_now = compute_slave_pos()
            pen_data = []  # [(local_idx, gap, normal, K_con), ...] for penetrating nodes

            for j in range(n_act):
                pid = int(saved_pids[j])
                t = saved_params[j]
                xs = slave_pos_now[active_idx[j]]

                # Get surface point and normal for gap computation
                xc  = patches[pid].Grg(t, deriv=0)
                n_raw = patches[pid].D3Grg(t)
                nor = n_raw / np.linalg.norm(n_raw)
                g = (xs - xc) @ nor

                if g < 0:  # penetrating
                    # Compute full consistent contact Hessian
                    K_con = compute_contact_hessian(pid, t, xs, g, nor)
                    pen_data.append((j, g, nor, K_con))

                    # Add contact force to residual
                    v = int(slave_verts[active_idx[j]])
                    r_np[v]          += kn * g * nor[0]
                    r_np[v + nv]     += kn * g * nor[1]
                    r_np[v + 2*nv]   += kn * g * nor[2]

            # 3. Convergence check on free DOFs
            rnorm = np.linalg.norm(r_np[free_dofs])
            total_newton += 1
            if rnorm < args.gtol:
                break

            # 4. Material tangent (linearization at current state)
            a_form.AssembleLinearization(gfu.vec)

            # 5. Add full contact Hessian: kn * (n ⊗ n + g * dn/dx_s)
            mat = a_form.mat
            for j_local, g, nor, K_con in pen_data:
                v = int(slave_verts[active_idx[j_local]])
                dofs = [v, v + nv, v + 2*nv]
                for a in range(3):
                    for b in range(3):
                        mat[dofs[a], dofs[b]] = mat[dofs[a], dofs[b]] + K_con[a, b]

            # 6. Solve K·Δu = -r and update
            inv = mat.Inverse(fes.FreeDofs(), inverse="umfpack")
            gfu.vec.data -= inv * res_vec

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
elif args.solver in ("newton-cg", "trust-ncg", "trust-krylov", "trust-constr"):
    hess_type = "full (with curvature)" if args.full_hessian else "simple (n⊗n)"
    print(f"  {nsteps} steps, solver={args.solver}, max_iter={max_iter}, gtol={args.gtol:.0e}, hess={hess_type}")
else:
    print(f"  {nsteps} steps, solver={args.solver}, max_iter={max_iter}, gtol={args.gtol:.0e}")
print(f"{'='*60}\n")

t_wall_start = perf_counter()

for step in range(1, nsteps + 1):
    t_step = perf_counter()

    # Reset contact cache at each load step (force full re-projection once)
    contact_cache.reset()

    # --- Incremental Dirichlet on "top" boundary --------------------------
    vec = gfu.vec.FV().NumPy()
    vec[top_dofs_x] += disp_increment[0]
    vec[top_dofs_y] += disp_increment[1]
    vec[top_dofs_z] += disp_increment[2]

    # ══════════════════════════════════════════════════════════════════════
    # SOLVER DISPATCH
    # ══════════════════════════════════════════════════════════════════════

    # --- Helper: compute contact forces and update fields ---
    def finalize_contact_state():
        """Evaluate contact state and update output fields. Returns (gn, normals, active, n_active, max_pen)."""
        slave_pos = compute_slave_pos()
        gn, normals, active = contact_cache.evaluate(slave_pos)
        f_con = np.zeros(ndof)
        if np.any(active):
            act = np.where(active)[0]
            verts_act = slave_verts[act]
            kgn = kn * gn[act]
            f_con[verts_act]          = kgn * normals[act, 0]
            f_con[verts_act + nv]     = kgn * normals[act, 1]
            f_con[verts_act + 2*nv]   = kgn * normals[act, 2]
        update_contact_fields(gn, normals, active, f_con)
        n_active = int(np.sum(active))
        max_pen  = -np.min(gn[active]) if n_active > 0 else 0.0
        return gn, normals, active, n_active, max_pen

    if args.solver == "newton":
        # ── Newton + active-set solver ──
        n_newton, n_as, gn, normals, active = newton_active_set_solve()

        # Compute contact forces for output
        f_con = np.zeros(ndof)
        if np.any(active):
            act = np.where(active)[0]
            verts_act = slave_verts[act]
            kgn = kn * gn[act]
            f_con[verts_act]          = kgn * normals[act, 0]
            f_con[verts_act + nv]     = kgn * normals[act, 1]
            f_con[verts_act + 2*nv]   = kgn * normals[act, 2]
        update_contact_fields(gn, normals, active, f_con)

        # Compute final residual norm for reporting
        a_form.Apply(gfu.vec, res_vec)
        r_np = res_vec.FV().NumPy()
        if np.any(active):
            for i in np.where(active)[0]:
                v, g = int(slave_verts[i]), gn[i]
                if g < 0:
                    nor = normals[i]
                    r_np[v]          += kn * g * nor[0]
                    r_np[v + nv]     += kn * g * nor[1]
                    r_np[v + 2*nv]   += kn * g * nor[2]
        rnorm = np.linalg.norm(r_np[free_dofs])

        n_active = int(np.sum(active))
        max_pen  = -np.min(gn[active]) if n_active > 0 else 0.0
        dt_step  = perf_counter() - t_step
        print(f"Step {step:3d}/{nsteps}  Newton: nit={n_newton:2d} as={n_as}  "
              f"|r|={rnorm:.2e}  active={n_active:3d}  maxpen={max_pen:.2e}  "
              f"t={dt_step:.1f}s")

    else:
        # ── Scipy minimize solvers ──
        # Solver configurations: (method_name, uses_hessp, options_override)
        SCIPY_SOLVERS = {
            "newton-cg":    ("Newton-CG",    True,  {"xtol": args.gtol}),
            "trust-ncg":    ("trust-ncg",    True,  {"gtol": args.gtol}),
            "trust-krylov": ("trust-krylov", True,  {"gtol": args.gtol}),
            "trust-constr": ("trust-constr", True,  {"gtol": args.gtol}),
            "bfgs":         ("BFGS",         False, {"gtol": args.gtol}),
            "cg":           ("CG",           False, {"gtol": args.gtol}),
            "tnc":          ("TNC",          False, {"gtol": args.gtol}),
            "slsqp":        ("SLSQP",        False, {"ftol": args.gtol}),
            "lbfgsb":       ("L-BFGS-B",     False, {"gtol": args.gtol, "maxls": 40, "ftol": 0}),
        }

        method_name, uses_hessp, opts_override = SCIPY_SOLVERS[args.solver]
        options = {"maxiter": max_iter, **opts_override}

        x0 = vec[free_dofs].copy()
        if uses_hessp:
            result = minimize(objective, x0, method=method_name, jac=True,
                              hessp=hessp, options=options)
        else:
            result = minimize(objective, x0, method=method_name, jac=True,
                              options=options)
        vec[free_dofs] = result.x

        # Finalize contact state and get metrics
        _, _, _, n_active, max_pen = finalize_contact_state()
        dt_step   = perf_counter() - t_step
        grad_norm = np.linalg.norm(result.jac) if result.jac is not None else float("nan")

        # Print result
        print(f"Step {step:3d}/{nsteps}  {args.solver}: nit={result.nit:2d}  "
              f"nfev={result.nfev:3d}  |grad|={grad_norm:.2e}  "
              f"active={n_active:3d}  maxpen={max_pen:.2e}  t={dt_step:.1f}s")
        if not result.success:
            print(f"  >> {result.message}")

    # --- VTK snapshot ------------------------------------------------------
    if args.plot > 0 and step % args.plot == 0:
        vtk.Do(time=step * dt)

    # --- Comparison output -------------------------------------------------
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

"""
ContactPotato_NGSolve.py — NGSolve hyperelastic contact simulation
==================================================================
Single self-contained script simulating a hyperelastic block sliding over a
rigid "potato" body.  ALL FE computations use the NGSolve API (mesh, FE space,
hyperelastic energy via Variation, Dirichlet BCs).  Contact detection reuses
the existing Gregory-patch / trust-region projection backend.  Minimization
is driven by scipy.optimize.minimize (L-BFGS-B).

Usage
-----
    python ContactPotato_NGSolve.py --mesh 5  --min_method LBFGSB
    python ContactPotato_NGSolve.py --mesh 10 --min_method LBFGSB --compare
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
parser.add_argument("--min_method", type=str,   default="LBFGSB",help="scipy method: LBFGSB")
parser.add_argument("--E",          type=float, default=0.05,    help="Young's modulus")
parser.add_argument("--nu",         type=float, default=0.3,     help="Poisson ratio")
parser.add_argument("--kn_factor",  type=float, default=20.0,    help="kn = kn_factor * E")
parser.add_argument("--nsteps",     type=int,   default=100,     help="number of load steps")
parser.add_argument("--max_iter",   type=int,   default=500,     help="max minimiser iterations per step")
parser.add_argument("--gtol",       type=float, default=1e-6,    help="gradient tolerance for convergence")
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
# Cauchy stress:  sigma = (1/J) * P * F^T   with P = dW/dF
# For this neo-Hookean:  sigma = (1/J)[2*c10*(F*F^T - I) + 2*d1*ln(J)*I]
sigma_cf = (1/J_gfu) * (2*c10*(F_gfu * F_gfu.trans - Id(3)) + 2*d1*log(J_gfu)*Id(3))
s_dev    = sigma_cf - (1.0/3.0) * Trace(sigma_cf) * Id(3)
vonmises_cf = sqrt(1.5 * InnerProduct(s_dev, s_dev))

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
    """TR projection for a single slave node. Returns (gn, normal)."""
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
    coefs=[gfu, sigma_cf, vonmises_cf,
           gf_contact_active, gf_penetration, gf_contact_rx],
    names=["displacement", "stress", "vonmises",
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
print(f"  {nsteps} steps, max_iter={max_iter}, gtol={args.gtol:.0e}, method={args.min_method}")
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

    # --- Minimise total energy on free DOFs --------------------------------
    x0 = vec[free_dofs].copy()
    result = minimize(
        objective, x0, method='L-BFGS-B', jac=True,
        options={'maxiter': max_iter, 'gtol': args.gtol,
                 'maxls': 40, 'ftol': 0},
    )
    vec[free_dofs] = result.x

    # --- Contact state for output (reuse last cache, cheap) ----------------
    slave_pos = np.column_stack([
        X_ref[slave_verts, 0] + vec[slave_verts],
        X_ref[slave_verts, 1] + vec[slave_verts + nv],
        X_ref[slave_verts, 2] + vec[slave_verts + 2*nv],
    ])
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
    dt_step  = perf_counter() - t_step
    print(f"Step {step:3d}/{nsteps}  success={result.success}  "
          f"nit={result.nit:2d}  nfev={result.nfev:3d}  "
          f"|grad|={np.linalg.norm(result.jac):.2e}  "
          f"active={n_active:3d}  maxpen={max_pen:.2e}  "
          f"t={dt_step:.1f}s")
    if not result.success and result.nit == 0:
        print(f"  >> WARNING: {result.message}")

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

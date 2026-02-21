"""Generate training data for the neural return mapping surrogate (Phase 3).

Runs ContactPotato_NGSolve.py-style elastoplastic simulations and records
all Gauss point return mapping inputs/outputs at every Newton iteration.

Each GP sample records:
    Input:  F(9) + Fp_old(9) + epcum(1) = 19 scalars
    Output: Fp_new(9) + delta_ep(1) = 10 scalars
    Meta:   yielding(1) + sigma_vm(1) + step(1) + nit(1) = 4 scalars

Usage:
    # Single run with default parameters:
    python -m nn_contact.scripts.generate_rm_data --out_dir data/rm_train

    # Sweep multiple configs (for training data diversity):
    python -m nn_contact.scripts.generate_rm_data --out_dir data/rm_train \
        --n 10 --E 0.05 --My0 0.01 --nsteps 100

    # HPC batch: run multiple parameter combos
    python -m nn_contact.scripts.generate_rm_data --out_dir data/rm_train --sweep
"""

from __future__ import annotations

import argparse
import os
import sys
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def run_simulation_and_collect(
    n: int = 10,
    E_val: float = 0.05,
    nu_val: float = 0.3,
    My0: float = 0.01,
    H_hard: float = 0.05,
    m_hard: float = 1.0,
    nsteps: int = 100,
    max_iter: int = 200,
    gtol: float = 1e-12,
    max_cutback: int = 5,
    linear_solver: str = "pardiso",
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Run a plastic contact simulation and collect return mapping data.

    Returns dict with arrays:
        inputs:    (N_total, 19)  — [F(9), Fp_old(9), epcum(1)]
        Fp_new:    (N_total, 9)   — updated Fp
        delta_ep:  (N_total,)     — plastic strain increment
        yielding:  (N_total,)     — 1 if GP yielded, 0 if elastic
        sigma_vm:  (N_total,)     — von Mises stress at trial state
        step:      (N_total,)     — load step index
        nit:       (N_total,)     — Newton iteration index within step
    """
    from ngsolve import (
        Id, Grad, Det, Trace, Inv, log, dx,
        VectorH1, GridFunction, BilinearForm, Variation,
        H1, L2, VTKOutput,
    )
    from ngsolve.meshes import MakeStructured3DMesh
    from ngsolve.comp import IntegrationRuleSpace, MatrixValued

    from PyClasses import gregory_patch_backend as gb
    from PyClasses._contact_tr_multi_helpers import project_points_tr_multi_batch

    # ── Mesh ──
    mesh = MakeStructured3DMesh(
        hexes=True, nx=n, ny=n, nz=n,
        mapping=lambda x, y, z: (-2 + 4*x, -2 + 4*y, -2 + 4*z + 3.5)
    )
    fes = VectorH1(mesh, order=1, dirichlet="top")
    gfu = GridFunction(fes)
    nv = mesh.nv
    ndof = fes.ndof

    # ── Material ──
    c10 = E_val / (4 * (1 + nu_val))
    d1 = E_val * nu_val / (2 * (1 + nu_val) * (1 - 2 * nu_val))

    u_trial, v_test = fes.TnT()
    F_sym = Id(3) + Grad(u_trial)

    # ── Plasticity IRS ──
    fes_ir = IntegrationRuleSpace(mesh, order=1)
    irs_dx = dx(intrules=fes_ir.GetIntegrationRules())
    n_ip = fes_ir.ndof

    fes_Fp = MatrixValued(fes_ir, dim=3)
    gf_Fp = GridFunction(fes_Fp)
    gf_Fp.Interpolate(Id(3))

    fes_F_ir = MatrixValued(fes_ir, dim=3)
    gf_F_ir = GridFunction(fes_F_ir)

    _Fp_conv = np.tile(np.eye(3).ravel(), n_ip)
    _epcum_conv = np.zeros(n_ip)
    _Fp_temp = _Fp_conv.copy()
    _delta_epcum = np.zeros(n_ip)

    Fe_sym = F_sym * Inv(gf_Fp)
    J_sym = Det(Fe_sym)
    I1_sym = Trace(Fe_sym.trans * Fe_sym)
    psi_sym = c10 * (I1_sym - 3 - 2*log(J_sym)) + d1 * log(J_sym)**2

    a_form = BilinearForm(fes)
    a_form += Variation(psi_sym * irs_dx)

    if verbose:
        print(f"  Mesh: {n}x{n}x{n}, IPS: {n_ip}, ndof: {ndof}")

    # ── Helper functions ──
    def _write_Fp_to_gf(Fp_flat):
        vec = gf_Fp.vec.FV().NumPy()
        Fp_all = Fp_flat.reshape(n_ip, 9)
        for comp in range(9):
            vec[comp * n_ip : (comp + 1) * n_ip] = Fp_all[:, comp]

    def _read_F_at_ips():
        gf_F_ir.Interpolate(Id(3) + Grad(gfu))
        vec = gf_F_ir.vec.FV().NumPy()
        F_all = np.zeros((n_ip, 9))
        for comp in range(9):
            F_all[:, comp] = vec[comp * n_ip : (comp + 1) * n_ip]
        return F_all.ravel()

    # ── Return mapping with data recording ──
    I3 = np.eye(3)

    def return_mapping_and_record(F_flat, Fp_conv_flat, epcum_conv,
                                   step_idx, nit_idx):
        """Return mapping + record all GP data for training."""
        n_gp = len(epcum_conv)
        F_all = F_flat.reshape(n_gp, 3, 3)
        Fp_old = Fp_conv_flat.reshape(n_gp, 3, 3)
        Fp_new = Fp_old.copy()
        delta_epcum = np.zeros(n_gp)
        yielding_flags = np.zeros(n_gp, dtype=np.float32)
        sigma_vm_arr = np.zeros(n_gp, dtype=np.float32)
        n_local_fail = 0

        for ip in range(n_gp):
            F = F_all[ip]
            Fp = Fp_old[ip]
            epcum = epcum_conv[ip]

            # Trial elastic predictor
            try:
                Fp_inv = np.linalg.inv(Fp)
            except np.linalg.LinAlgError:
                continue
            Fe = F @ Fp_inv
            Ce = Fe.T @ Fe
            Je = np.linalg.det(Fe)
            if Je <= 1e-15:
                continue

            M = 2*c10*(Ce - I3) + 2*d1*np.log(Je)*I3
            M_dev = M - (1.0/3.0)*np.trace(M)*I3
            norm_Mdev_sq = np.sum(M_dev * M_dev)
            sigma_vm = np.sqrt(1.5 * norm_Mdev_sq)
            sigma_vm_arr[ip] = sigma_vm

            sigma_y = My0 + (H_hard * epcum**m_hard if epcum > 1e-30 else 0.0)
            f_yield = sigma_vm - sigma_y

            if f_yield <= 0.0:
                continue  # elastic

            yielding_flags[ip] = 1.0

            if sigma_vm <= 1e-30:
                continue

            N_flow = 1.5 * M_dev / sigma_vm
            eigvals_N, V_N = np.linalg.eigh(N_flow)

            dlam = 1e-8
            Fp_trial = Fp.copy()
            for k_rm in range(50):
                exp_diag = np.exp(dlam * eigvals_N)
                Fp_trial = (V_N * exp_diag) @ V_N.T @ Fp

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

                M_new = 2*c10*(Ce_new - I3) + 2*d1*np.log(Je_new)*I3
                M_dev_new = M_new - (1.0/3.0)*np.trace(M_new)*I3
                svm_new = np.sqrt(1.5 * np.sum(M_dev_new * M_dev_new))

                ep_total = epcum + dlam
                sy_new = My0 + (H_hard * ep_total**m_hard if ep_total > 1e-30 else 0.0)
                R_val = svm_new - sy_new

                if abs(R_val) < 1e-15:
                    break

                # FD derivative
                h_fd = max(1e-10, abs(dlam) * 1e-6)
                exp_diag_h = np.exp((dlam + h_fd) * eigvals_N)
                Fp_h = ((V_N * exp_diag_h) @ V_N.T) @ Fp
                if abs(np.linalg.det(Fp_h)) < 1e-15:
                    break
                try:
                    Fe_h = F @ np.linalg.inv(Fp_h)
                except np.linalg.LinAlgError:
                    break
                Ce_h = Fe_h.T @ Fe_h
                Je_h = np.linalg.det(Fe_h)
                if Je_h <= 1e-15 or not np.isfinite(Je_h):
                    break
                M_h = 2*c10*(Ce_h - I3) + 2*d1*np.log(Je_h)*I3
                M_dev_h = M_h - (1.0/3.0)*np.trace(M_h)*I3
                svm_h = np.sqrt(1.5 * np.sum(M_dev_h * M_dev_h))
                ep_h = epcum + dlam + h_fd
                sy_h = My0 + (H_hard * ep_h**m_hard if ep_h > 1e-30 else 0.0)
                R_h = svm_h - sy_h

                dR = (R_h - R_val) / h_fd
                if abs(dR) < 1e-30:
                    break

                dlam -= R_val / dR
                dlam = max(dlam, 0.0)

            if abs(R_val) > 1e-6:
                n_local_fail += 1

            Fp_new[ip] = Fp_trial
            delta_epcum[ip] = max(dlam, 0.0)

        if not np.isfinite(Fp_new).all():
            return Fp_conv_flat.copy(), np.zeros(n_gp), False, None

        if n_local_fail > 0:
            return Fp_conv_flat.copy(), np.zeros(n_gp), False, None

        # Build training data record
        inputs = np.column_stack([
            F_flat.reshape(n_gp, 9),        # F (9)
            Fp_conv_flat.reshape(n_gp, 9),   # Fp_old (9)
            epcum_conv.reshape(n_gp, 1),     # epcum (1)
        ])  # (n_gp, 19)

        record = {
            "inputs": inputs.astype(np.float32),
            "Fp_new": Fp_new.reshape(n_gp, 9).astype(np.float32),
            "delta_ep": delta_epcum.astype(np.float32),
            "yielding": yielding_flags,
            "sigma_vm": sigma_vm_arr,
            "step": np.full(n_gp, step_idx, dtype=np.int32),
            "nit": np.full(n_gp, nit_idx, dtype=np.int32),
        }

        return Fp_new.ravel(), delta_epcum, True, record

    # ── Contact setup ──
    ptt_file = os.path.join(project_root, '1_Minimization_solvers', 'Dat', 'PotatoAssembly.dat')
    with open(ptt_file, 'rb') as f:
        [ptt] = pickle.load(f)
    if hasattr(ptt, 'hexas') and not hasattr(ptt, 'elements'):
        ptt.elements = ptt.hexas
    ptt.isRigid = True
    n_nodes_ptt = len(ptt.X)
    ndofs_ptt = 3 * n_nodes_ptt
    ptt.DoFs = np.array([[3*i, 3*i+1, 3*i+2] for i in range(n_nodes_ptt)])
    ptt.surf.ComputeGrgPatches(np.zeros(ndofs_ptt), range(len(ptt.surf.nodes)))

    patches = ptt.surf.patches
    n_patches = len(patches)
    ctrlpts_all = np.vstack([np.array(p.flatCtrlPts()) for p in patches])
    radii = np.array([p.BS.r for p in patches], dtype=np.float64)
    eps = patches[0].eps

    # Surface sampling for KD-tree
    from scipy.spatial import cKDTree
    N_SURF = 50
    s1d = np.linspace(0, 1, N_SURF)
    surf_pts = []
    surf_pids = []
    for pid, p in enumerate(patches):
        for u_s in s1d:
            for v_s in s1d:
                pt = p.Grg0(np.array([u_s, v_s], dtype=np.float64))
                surf_pts.append(pt)
                surf_pids.append(pid)
    surf_pts = np.array(surf_pts, dtype=np.float64)
    surf_pids = np.array(surf_pids, dtype=np.int32)
    surf_kdtree = cKDTree(surf_pts)

    # TR parameters
    xm_matrix = np.array([p.BS.x for p in patches], dtype=np.float64)
    TR_INIT, TR_MIN, TR_MAX = 0.3, 1e-6, 2.0
    BASE_NCAND, MIN_NCAND, MAX_NCAND = 5, 3, 12
    RADIUS_FACTOR, K_SURF = 1.5, 10

    # ── Slave node identification ──
    X_ref = np.array([list(mesh.vertices[i].point) for i in range(nv)])
    bottom_z_threshold = X_ref[:, 2].min() + 0.01
    slave_verts = np.where(X_ref[:, 2] < bottom_z_threshold)[0]
    n_slave = len(slave_verts)
    if verbose:
        print(f"  Slave nodes: {n_slave}")

    h_contact = 4.0 / n
    kn = 20.0 * E_val / h_contact

    # ── DOF setup ──
    from ngsolve import BitArray
    free_bits = fes.FreeDofs()
    free_dofs = np.array([i for i in range(ndof) if free_bits[i]])

    top_z = X_ref[:, 2].max()
    top_verts = np.where(np.abs(X_ref[:, 2] - top_z) < 1e-8)[0]
    top_dofs_x = top_verts
    top_dofs_y = top_verts + nv
    top_dofs_z = top_verts + 2 * nv

    # ── Residual vector ──
    res_vec = gfu.vec.CreateVector()
    _w_vec = gfu.vec.CreateVector()
    _uh_vec = gfu.vec.CreateVector()

    def compute_slave_pos():
        vec = gfu.vec.FV().NumPy()
        return np.column_stack([
            X_ref[slave_verts, 0] + vec[slave_verts],
            X_ref[slave_verts, 1] + vec[slave_verts + nv],
            X_ref[slave_verts, 2] + vec[slave_verts + 2*nv],
        ])

    def project_slaves(slave_pos):
        pids, t1, t2, gn, nor, xc = project_points_tr_multi_batch(
            slave_pos, xm_matrix, ctrlpts_all, radii, eps,
            TR_INIT, TR_MIN, TR_MAX,
            surf_kdtree, surf_pids, BASE_NCAND, MIN_NCAND,
            MAX_NCAND, RADIUS_FACTOR, K_SURF,
        )
        return pids, t1, t2, gn, nor, xc

    def compute_contact_hessian_simple(g, nor):
        return kn * np.outer(nor, nor)

    # ── Linesearch energy ──
    def _linesearch_energy(u_vec):
        E_mat = a_form.Energy(u_vec)
        slave_pos = np.column_stack([
            X_ref[slave_verts, 0] + u_vec.FV().NumPy()[slave_verts],
            X_ref[slave_verts, 1] + u_vec.FV().NumPy()[slave_verts + nv],
            X_ref[slave_verts, 2] + u_vec.FV().NumPy()[slave_verts + 2*nv],
        ])
        _, _, _, gn_ls, nor_ls, _ = project_slaves(slave_pos)
        pids_valid = np.array(gn_ls) < 0
        E_con = 0.0
        if np.any(pids_valid):
            E_con = 0.5 * kn * np.sum(np.array(gn_ls)[pids_valid]**2)
        return E_mat + E_con

    # ── Data collection ──
    all_records = []

    # ── Time-stepping loop ──
    dt_base = 1.0 / nsteps
    disp_per_unit = np.array([12.0, 0.0, 0.0])

    if verbose:
        print(f"\n  Running: n={n}, E={E_val}, My0={My0}, "
              f"nsteps={nsteps}, kn={kn:.4f}")

    t_wall = perf_counter()
    load = 0.0
    dt = dt_base
    bisect_level = 0
    total_gp_samples = 0

    while load < 1.0 - 1e-12:
        dt = min(dt, 1.0 - load)
        step_idx = int(load / dt_base + 1e-9)

        # Save backup
        vec = gfu.vec.FV().NumPy()
        u_backup = vec.copy()

        # Apply Dirichlet increment
        disp_inc = disp_per_unit * dt
        vec[top_dofs_x] += disp_inc[0]
        vec[top_dofs_y] += disp_inc[1]
        vec[top_dofs_z] += disp_inc[2]

        # Newton solve
        _u_newton_backup = gfu.vec.FV().NumPy().copy()
        converged = False
        rnorm_prev = np.inf
        rnorm_initial = np.inf
        stag_count = 0

        for nit in range(max_iter):
            # 0. Return mapping with data recording
            F_flat = _read_F_at_ips()
            _Fp_temp, _delta_epcum, rm_ok, record = \
                return_mapping_and_record(
                    F_flat, _Fp_conv, _epcum_conv,
                    step_idx, nit)

            if not rm_ok:
                gfu.vec.FV().NumPy()[:] = _u_newton_backup
                break

            # Save training data
            if record is not None:
                all_records.append(record)
                total_gp_samples += record["inputs"].shape[0]

            _write_Fp_to_gf(_Fp_temp)

            # 1. Residual
            a_form.Apply(gfu.vec, res_vec)
            r_np = res_vec.FV().NumPy()

            # 2. Contact
            slave_pos = compute_slave_pos()
            if not np.isfinite(slave_pos).all():
                gfu.vec.FV().NumPy()[:] = _u_newton_backup
                break

            pids, t1, t2, gn, normals, xc = project_slaves(slave_pos)
            pids = np.asarray(pids, dtype=np.int32)
            gn = np.asarray(gn)
            normals = np.asarray(normals)
            active = (pids >= 0) & (gn < 0)
            active_idx = np.where(active)[0]

            # Add contact forces
            pen_data = []
            if len(active_idx) > 0:
                for i in active_idx:
                    g = gn[i]
                    nor = normals[i]
                    kgn = kn * g
                    v = slave_verts[i]
                    r_np[v] += kgn * nor[0]
                    r_np[v + nv] += kgn * nor[1]
                    r_np[v + 2*nv] += kgn * nor[2]
                    K_con = compute_contact_hessian_simple(g, nor)
                    pen_data.append((i, g, nor, K_con))

            # 3. Convergence
            rnorm = np.linalg.norm(r_np[free_dofs])
            if nit == 0:
                rnorm_initial = rnorm
            if rnorm < gtol:
                converged = True
                break

            # Stagnation
            if nit >= 10 and rnorm < 1e-2 * rnorm_initial:
                if rnorm > 0.95 * rnorm_prev:
                    stag_count += 1
                    if stag_count >= 5:
                        converged = True
                        break
                else:
                    stag_count = 0
            rnorm_prev = rnorm

            # 4. Tangent assembly
            a_form.AssembleLinearization(gfu.vec)

            # Add contact Hessian
            mat = a_form.mat
            for idx, g, nor, K_con in pen_data:
                v = int(slave_verts[idx])
                dofs = [v, v + nv, v + 2*nv]
                for a in range(3):
                    for b in range(3):
                        mat[dofs[a], dofs[b]] += K_con[a, b]

            # 5. Solve
            try:
                inv_mat = mat.Inverse(fes.FreeDofs(), inverse=linear_solver)
                _w_vec.data = inv_mat * res_vec
                if not np.isfinite(_w_vec.FV().NumPy()).all():
                    break
            except Exception:
                break

            # 6. Linesearch
            w_free = _w_vec.FV().NumPy()[free_dofs]
            slope = -np.dot(r_np[free_dofs], w_free)

            if slope >= 0:
                r_max = np.max(np.abs(r_np[free_dofs]))
                if r_max > 1e-30:
                    scale = 0.1 * h_contact / r_max
                    w_np = _w_vec.FV().NumPy()
                    w_np[:] = 0
                    w_np[free_dofs] = scale * r_np[free_dofs]
                    w_free = w_np[free_dofs]
                    slope = -np.dot(r_np[free_dofs], w_free)
                else:
                    break

            energy_old = _linesearch_energy(gfu.vec)
            tau = 1.0
            accepted = False
            for _ in range(30):
                _uh_vec.data = gfu.vec - tau * _w_vec
                energy_new = _linesearch_energy(_uh_vec)
                if np.isfinite(energy_new) and energy_new <= energy_old + 1e-4 * tau * slope:
                    accepted = True
                    break
                tau *= 0.5

            if accepted:
                gfu.vec.data = _uh_vec
            else:
                r_max = np.max(np.abs(r_np[free_dofs]))
                if r_max > 1e-30:
                    scale = 0.01 * h_contact / r_max
                    uh_np = _uh_vec.FV().NumPy()
                    uh_np[:] = gfu.vec.FV().NumPy()
                    uh_np[free_dofs] -= scale * r_np[free_dofs]
                    gfu.vec.data = _uh_vec

        # Cutback or accept
        if not converged and bisect_level < max_cutback:
            vec[:] = u_backup
            dt /= 2
            bisect_level += 1
            if verbose:
                print(f"  Step {step_idx}: cutback {bisect_level}")
            continue

        # Accept step
        load += dt

        # Commit plastic history
        if converged and np.isfinite(_Fp_temp).all():
            _Fp_conv[:] = _Fp_temp
            _epcum_conv += _delta_epcum

        # Reset cutback on success
        if converged and bisect_level > 0:
            dt = dt_base
            bisect_level = 0

        curr_step = int(load / dt_base + 1e-9)
        n_active = int(np.sum(active)) if 'active' in dir() else 0
        n_yield = int(np.sum(_delta_epcum > 0))
        ep_max = np.max(_epcum_conv)

        if verbose:
            print(f"  Step {curr_step:3d}/{nsteps}  nit={nit+1:3d}  "
                  f"|r|={rnorm:.2e}  active={n_active:3d}  "
                  f"yielding={n_yield}  ep_max={ep_max:.2e}  "
                  f"samples={total_gp_samples:,}")

    wall_time = perf_counter() - t_wall
    if verbose:
        print(f"\n  Done in {wall_time:.1f}s, collected {total_gp_samples:,} GP samples")

    # Concatenate all records
    if not all_records:
        raise RuntimeError("No data collected — simulation may have diverged")

    result = {
        key: np.concatenate([r[key] for r in all_records], axis=0)
        for key in all_records[0].keys()
    }

    return result


def save_data(data: dict[str, np.ndarray], out_dir: str, tag: str = ""):
    """Save collected data as .npy files."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    suffix = f"_{tag}" if tag else ""

    np.save(out_path / f"rm_inputs{suffix}.npy", data["inputs"])
    np.save(out_path / f"rm_Fp_new{suffix}.npy", data["Fp_new"])
    np.save(out_path / f"rm_dep{suffix}.npy", data["delta_ep"])
    np.save(out_path / f"rm_yielding{suffix}.npy", data["yielding"])
    np.save(out_path / f"rm_sigma_vm{suffix}.npy", data["sigma_vm"])
    np.save(out_path / f"rm_step{suffix}.npy", data["step"])
    np.save(out_path / f"rm_nit{suffix}.npy", data["nit"])

    n_total = data["inputs"].shape[0]
    n_yield = int(np.sum(data["yielding"] > 0.5))
    print(f"  Saved {n_total:,} samples ({n_yield:,} yielding, "
          f"{n_total - n_yield:,} elastic) to {out_path}/")


def merge_data(out_dir: str):
    """Merge all per-run .npy files into unified training arrays."""
    out_path = Path(out_dir)

    # Find all tagged files
    input_files = sorted(out_path.glob("rm_inputs_*.npy"))
    if not input_files:
        print("No per-run files found to merge")
        return

    tags = [f.stem.replace("rm_inputs_", "") for f in input_files]
    print(f"Merging {len(tags)} runs: {tags}")

    for key in ["inputs", "Fp_new", "dep", "yielding", "sigma_vm", "step", "nit"]:
        arrays = []
        for tag in tags:
            fpath = out_path / f"rm_{key}_{tag}.npy"
            if fpath.exists():
                arrays.append(np.load(fpath))
        if arrays:
            merged = np.concatenate(arrays, axis=0)
            np.save(out_path / f"rm_{key}.npy", merged)
            print(f"  rm_{key}.npy: {merged.shape}")

    # Print summary
    inputs = np.load(out_path / "rm_inputs.npy")
    yielding = np.load(out_path / "rm_yielding.npy")
    n_total = inputs.shape[0]
    n_yield = int(np.sum(yielding > 0.5))
    print(f"\nTotal: {n_total:,} samples ({n_yield:,} yielding = "
          f"{100*n_yield/n_total:.1f}%)")


# ── Sweep configurations ──
SWEEP_CONFIGS = [
    # (tag, n, E, My0, H, m, nsteps)
    ("n5_E005_My001",   5,  0.05, 0.01, 0.05, 1.0, 100),
    ("n10_E005_My001", 10,  0.05, 0.01, 0.05, 1.0, 100),
    ("n10_E005_My001_s50", 10, 0.05, 0.01, 0.05, 1.0, 50),
    ("n10_E003_My001", 10,  0.03, 0.01, 0.05, 1.0, 100),
    ("n10_E007_My001", 10,  0.07, 0.01, 0.05, 1.0, 100),
    ("n10_E005_My005", 10,  0.05, 0.005, 0.05, 1.0, 100),
    ("n10_E005_My002", 10,  0.05, 0.02, 0.05, 1.0, 100),
    ("n15_E005_My001", 15,  0.05, 0.01, 0.05, 1.0, 100),
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate return mapping training data")
    parser.add_argument("--out_dir", default="data/rm_train",
                        help="Output directory")
    parser.add_argument("--n", type=int, default=10, help="Mesh density")
    parser.add_argument("--E", type=float, default=0.05, help="Young's modulus")
    parser.add_argument("--My0", type=float, default=0.01, help="Yield stress")
    parser.add_argument("--H", type=float, default=0.05, help="Hardening modulus")
    parser.add_argument("--m", type=float, default=1.0, help="Hardening exponent")
    parser.add_argument("--nsteps", type=int, default=100, help="Load steps")
    parser.add_argument("--linear_solver", default="pardiso",
                        help="Direct solver: umfpack, pardiso, mumps")
    parser.add_argument("--sweep", action="store_true",
                        help="Run all sweep configs")
    parser.add_argument("--sweep_index", type=int, default=-1,
                        help="Run specific sweep config (for SLURM array)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge existing per-run files")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_data(args.out_dir)
        return

    # SLURM array job support
    if args.sweep_index < 0 and "SLURM_ARRAY_TASK_ID" in os.environ:
        args.sweep_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if args.sweep or args.sweep_index >= 0:
        if args.sweep_index >= 0:
            configs = [SWEEP_CONFIGS[args.sweep_index]]
        else:
            configs = SWEEP_CONFIGS

        for tag, n, E, My0, H, m, nsteps in configs:
            print(f"\n{'='*60}")
            print(f"  Config: {tag}")
            print(f"{'='*60}")
            data = run_simulation_and_collect(
                n=n, E_val=E, My0=My0, H_hard=H, m_hard=m,
                nsteps=nsteps, linear_solver=args.linear_solver,
                verbose=not args.quiet,
            )
            save_data(data, args.out_dir, tag=tag)

        if args.sweep:
            merge_data(args.out_dir)
    else:
        tag = f"n{args.n}_E{args.E:.0e}_My{args.My0:.0e}"
        data = run_simulation_and_collect(
            n=args.n, E_val=args.E, My0=args.My0, H_hard=args.H,
            m_hard=args.m, nsteps=args.nsteps,
            linear_solver=args.linear_solver,
            verbose=not args.quiet,
        )
        save_data(data, args.out_dir, tag=tag)


if __name__ == "__main__":
    main()

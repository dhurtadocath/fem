"""Evaluate Neural-Pull and Multitask checkpoints against C++ ground truth.

Tests all variants from nn_contact/checkpoints/external/ by comparing
NN predictions to C++ TR projection on a standardized set of test points.

Usage:
    python -m nn_contact.scripts.evaluate_checkpoints
    python -m nn_contact.scripts.evaluate_checkpoints --mode neural_pull
    python -m nn_contact.scripts.evaluate_checkpoints --mode multitask
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def setup_potato():
    """Load potato geometry and return patches + projection infrastructure."""
    from PyClasses import gregory_patch_backend as gb
    from PyClasses._contact_tr_multi_helpers import project_points_tr_multi_batch
    from scipy.spatial import cKDTree

    ptt_file = project_root / "1_Minimization_solvers" / "Dat" / "PotatoAssembly.dat"
    with open(ptt_file, "rb") as f:
        [ptt] = pickle.load(f)
    if hasattr(ptt, "hexas") and not hasattr(ptt, "elements"):
        ptt.elements = ptt.hexas
    ptt.isRigid = True
    n_nodes = len(ptt.X)
    ndofs = 3 * n_nodes
    ptt.DoFs = np.array([[3 * i, 3 * i + 1, 3 * i + 2] for i in range(n_nodes)])
    ptt.surf.ComputeGrgPatches(np.zeros(ndofs), range(len(ptt.surf.nodes)))

    patches = ptt.surf.patches
    ctrlpts_all = np.vstack([np.array(p.flatCtrlPts()) for p in patches])
    radii = np.array([p.BS.r for p in patches], dtype=np.float64)
    eps = patches[0].eps
    xm_matrix = np.array([p.BS.x for p in patches], dtype=np.float64)

    # KD-tree
    s1d = np.linspace(0, 1, 50)
    surf_pts, surf_pids = [], []
    for pid, p in enumerate(patches):
        for u in s1d:
            for v in s1d:
                surf_pts.append(p.Grg0(np.array([u, v], dtype=np.float64)))
                surf_pids.append(pid)
    surf_pts = np.array(surf_pts, dtype=np.float64)
    surf_pids = np.array(surf_pids, dtype=np.int32)
    kdtree = cKDTree(surf_pts)

    ptt_center = np.array(ptt.X).mean(axis=0)

    return {
        "ptt": ptt,
        "patches": patches,
        "ctrlpts_all": ctrlpts_all,
        "radii": radii,
        "eps": eps,
        "xm_matrix": xm_matrix,
        "kdtree": kdtree,
        "surf_pids": surf_pids,
        "ptt_center": ptt_center,
        "project_fn": project_points_tr_multi_batch,
    }


def generate_test_points(infra, n_points=2000, seed=42):
    """Generate test points near the potato surface (matching training distribution).

    Strategy: sample surface points on patches, then perturb along the normal
    with gap values in [-0.5, 1.5] (same range as training data).
    """
    rng = np.random.default_rng(seed)
    patches = infra["patches"]
    n_patches = len(patches)

    pts = []
    for _ in range(n_points):
        # Random patch, random parametric coords
        pid = rng.integers(0, n_patches)
        t = rng.uniform(0.05, 0.95, 2)  # avoid patch edges
        t_arr = np.array(t, dtype=np.float64)

        # Surface point and normal
        surf_pt = patches[pid].Grg0(t_arr)
        # Normal via cross product of tangent vectors
        dt = 1e-5
        t1p = t_arr.copy(); t1p[0] += dt
        t2p = t_arr.copy(); t2p[1] += dt
        tau1 = (patches[pid].Grg0(t1p) - surf_pt) / dt
        tau2 = (patches[pid].Grg0(t2p) - surf_pt) / dt
        normal = np.cross(tau1, tau2)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm

        # Perturb along normal: gap in [-0.5, 1.5] (training range)
        gap = rng.uniform(-0.5, 1.5)
        pt = surf_pt + gap * normal
        pts.append(pt)

    return np.array(pts[:n_points], dtype=np.float64)


def project_cpp(pts, infra):
    """Ground truth: C++ TR projection."""
    pids, t1, t2, gn, normals, xc_surf = infra["project_fn"](
        pts, infra["xm_matrix"], infra["ctrlpts_all"], infra["radii"],
        infra["eps"], 0.3, 1e-6, 2.0,
        infra["kdtree"], infra["surf_pids"], 5, 3, 12, 1.5, 10,
    )
    return {
        "patch_ids": np.asarray(pids, dtype=np.int32),
        "gn": np.asarray(gn, dtype=np.float64),
        "normals": np.asarray(normals, dtype=np.float64),
        "xc_surf": np.asarray(xc_surf, dtype=np.float64),
    }


def evaluate_neural_pull(ckpt_dir, pts, gt, ptt_center, device="cuda"):
    """Evaluate a Neural-Pull checkpoint."""
    from nn_contact.evaluation.integration import NeuralPullCDA

    t0 = perf_counter()
    cda = NeuralPullCDA.from_checkpoint(ckpt_dir, device=device)
    cda.coord_offset = -ptt_center
    load_time = perf_counter() - t0

    t0 = perf_counter()
    result = cda.evaluate(pts, compute_hessian=True)
    eval_time = perf_counter() - t0

    gn_nn = result["gn"]
    normals_nn = result["normals"]

    valid = gt["patch_ids"] >= 0
    gn_gt = gt["gn"][valid]
    gn_pred = gn_nn[valid]
    nor_gt = gt["normals"][valid]
    nor_pred = normals_nn[valid]

    # Gap metrics
    gn_err = gn_pred - gn_gt
    gn_rmse = np.sqrt(np.mean(gn_err**2))
    gn_mae = np.mean(np.abs(gn_err))
    gn_max = np.max(np.abs(gn_err))

    # Normal metrics
    cos_sim = np.sum(nor_gt * nor_pred, axis=1)
    cos_sim = np.clip(cos_sim, -1, 1)
    angle_deg = np.degrees(np.arccos(np.abs(cos_sim)))
    angle_mean = np.mean(angle_deg)
    angle_max = np.max(angle_deg)

    # Contact detection: count points with gn < 0
    contact_gt = gn_gt < 0
    contact_nn = gn_pred < 0
    n_contact_gt = np.sum(contact_gt)

    if n_contact_gt > 0:
        # Among GT contacts, how many does NN also detect?
        detect_rate = np.sum(contact_gt & contact_nn) / n_contact_gt
        # False positives
        n_false_pos = np.sum(~contact_gt & contact_nn)
    else:
        detect_rate = 1.0
        n_false_pos = np.sum(contact_nn)

    # Hessian metrics (if available)
    has_hessian = "dndxs" in result and result["dndxs"] is not None
    hess_norm = None
    if has_hessian:
        dndxs = result["dndxs"]
        hess_norm = np.mean(np.linalg.norm(dndxs[valid].reshape(-1, 9), axis=1))

    return {
        "gn_rmse": gn_rmse,
        "gn_mae": gn_mae,
        "gn_max": gn_max,
        "angle_mean": angle_mean,
        "angle_max": angle_max,
        "detect_rate": detect_rate,
        "n_false_pos": n_false_pos,
        "n_contact_gt": n_contact_gt,
        "has_hessian": has_hessian,
        "hess_norm": hess_norm,
        "n_valid": int(np.sum(valid)),
        "load_time": load_time,
        "eval_time": eval_time,
    }


def _infer_arch_flags(name: str) -> tuple[bool, bool]:
    """Infer task_attention and patch_conditioned from sweep config name."""
    # These sweep configs used task_attention
    attn_names = {"task_attn", "focal_ls_attn", "full_combo"}
    # These used patch_conditioned
    pcond_names = {"patch_cond", "focal_ls_pcond"}

    task_attention = any(n in name for n in attn_names)
    patch_conditioned = any(n in name for n in pcond_names)
    return task_attention, patch_conditioned


def evaluate_multitask(ckpt_dir, pts, gt, ptt_center, device="cuda",
                       ref_normalizer=None):
    """Evaluate a Multitask checkpoint."""
    import torch
    from nn_contact.models.multitask import MultiTaskContactNet
    from nn_contact.config import MultiTaskConfig
    from nn_contact.data.normalization import CoordinateNormalizer
    from nn_contact.evaluation.integration import HybridCDA

    model_path = Path(ckpt_dir) / "best_model.pt"
    config_path = Path(ckpt_dir) / "config.pt"

    # Load normalizer from config.pt or use reference
    normalizer = CoordinateNormalizer()
    if config_path.exists():
        config_data = torch.load(config_path, map_location="cpu", weights_only=False)
        if "normalizer" in config_data:
            normalizer = CoordinateNormalizer.from_state_dict(config_data["normalizer"])
    elif ref_normalizer is not None:
        normalizer = ref_normalizer

    # Load checkpoint — extract config from model checkpoint itself
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", MultiTaskConfig())

    # For old checkpoints, infer architecture flags from directory name
    dir_name = Path(ckpt_dir).name
    if not getattr(cfg, "task_attention", False) and not getattr(cfg, "patch_conditioned", False):
        task_attn, patch_cond = _infer_arch_flags(dir_name)
        cfg.task_attention = task_attn
        cfg.patch_conditioned = patch_cond

    t0 = perf_counter()
    model = MultiTaskContactNet.from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    cda = HybridCDA.__new__(HybridCDA)
    cda.device = device
    cda.normalizer = normalizer
    cda.gn_threshold = 0.5
    cda.confidence_threshold = 0.5
    cda.topk = 3
    cda.coord_offset = -ptt_center
    cda.model = model
    load_time = perf_counter() - t0

    t0 = perf_counter()
    result = cda.predict(pts)
    eval_time = perf_counter() - t0

    valid = gt["patch_ids"] >= 0
    active = result["active_mask"]
    both = valid & active  # points where both NN and GT give a result

    # Patch accuracy among NN-active points
    nn_top1 = result["patch_ids"][:, 0]
    nn_topk = result["patch_ids"]  # (N, K)
    gt_pids = gt["patch_ids"]

    if np.sum(both) > 0:
        top1_match = nn_top1[both] == gt_pids[both]
        top1_acc = np.mean(top1_match)
        topk_match = np.any(nn_topk[both] == gt_pids[both, None], axis=1)
        topk_acc = np.mean(topk_match)
    else:
        top1_acc = 0.0
        topk_acc = 0.0

    # Coverage: fraction of valid GT points that NN marks active
    coverage = np.sum(both) / max(np.sum(valid), 1)

    # Gap estimation (among active points)
    gn_approx = result["gn_approx"]
    if np.sum(both) > 0:
        gn_err = gn_approx[both] - gt["gn"][both]
        gn_rmse = np.sqrt(np.mean(gn_err**2))
    else:
        gn_rmse = float("nan")

    return {
        "top1_acc": top1_acc,
        "topk_acc": topk_acc,
        "coverage": coverage,
        "gn_rmse": gn_rmse,
        "n_valid": int(np.sum(valid)),
        "n_active": int(np.sum(active)),
        "n_fallback": int(np.sum(result["needs_fallback"])),
        "load_time": load_time,
        "eval_time": eval_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NN checkpoints")
    parser.add_argument("--mode", choices=["all", "neural_pull", "multitask"],
                        default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_points", type=int, default=2000)
    parser.add_argument("--ckpt_base", default=str(
        project_root / "nn_contact" / "checkpoints" / "external"))
    args = parser.parse_args()

    print("Setting up geometry...")
    infra = setup_potato()

    print(f"Generating {args.n_points} near-surface test points...")
    pts = generate_test_points(infra, args.n_points)

    print("Computing C++ ground truth...")
    t0 = perf_counter()
    gt = project_cpp(pts, infra)
    cpp_time = perf_counter() - t0
    n_valid = np.sum(gt["patch_ids"] >= 0)
    n_contact = np.sum(gt["gn"][gt["patch_ids"] >= 0] < 0)
    print(f"  C++ TR: {cpp_time:.3f}s, {n_valid}/{args.n_points} valid, "
          f"{n_contact} in contact")

    base = Path(args.ckpt_base)

    # ── Neural-Pull evaluation ──
    if args.mode in ("all", "neural_pull"):
        print(f"\n{'='*70}")
        print("  NEURAL-PULL EVALUATION")
        print(f"{'='*70}")

        np_dirs = sorted([d for d in base.iterdir()
                          if d.is_dir() and d.name.startswith("neural_pull")])

        if not np_dirs:
            print("  No neural_pull checkpoints found")
        else:
            print(f"\n{'Name':<22s} {'gn_RMSE':>9s} {'gn_MAE':>9s} "
                  f"{'gn_MAX':>9s} {'angle':>7s} {'a_max':>7s} "
                  f"{'det%':>6s} {'FP':>4s} {'hess':>6s} {'time':>7s}")
            print("-" * 95)

            for d in np_dirs:
                if not (d / "best_model.pt").exists():
                    continue
                try:
                    r = evaluate_neural_pull(d, pts, gt, infra["ptt_center"],
                                             device=args.device)
                    hess_str = f"{r['hess_norm']:.4f}" if r["hess_norm"] else "N/A"
                    det_str = f"{r['detect_rate']*100:.1f}" if r["n_contact_gt"] > 0 else "N/A"
                    print(f"{d.name:<22s} {r['gn_rmse']:9.5f} {r['gn_mae']:9.5f} "
                          f"{r['gn_max']:9.5f} {r['angle_mean']:7.3f} "
                          f"{r['angle_max']:7.2f} {det_str:>6s} "
                          f"{r['n_false_pos']:4d} {hess_str:>6s} "
                          f"{r['eval_time']:7.3f}s")
                except Exception as e:
                    print(f"{d.name:<22s}  ERROR: {e}")

            print(f"\n  C++ reference time: {cpp_time:.3f}s for {args.n_points} points")

    # ── Multitask evaluation ──
    if args.mode in ("all", "multitask"):
        print(f"\n{'='*70}")
        print("  MULTITASK EVALUATION")
        print(f"{'='*70}")

        # First: versioned multitask (v1, v2, v3)
        mt_versioned = sorted([d for d in base.iterdir()
                                if d.is_dir() and d.name.startswith("multitask_v")])
        # Then: sweep configs
        mt_sweep = sorted([d for d in base.iterdir()
                           if d.is_dir() and d.name.startswith("mt_sweep_")])

        all_mt = mt_versioned + mt_sweep

        if not all_mt:
            print("  No multitask checkpoints found")
        else:
            # Load reference normalizer from a versioned checkpoint with config.pt
            import torch as _torch
            from nn_contact.data.normalization import CoordinateNormalizer
            ref_normalizer = None
            for d in mt_versioned:
                if (d / "config.pt").exists():
                    cfg_data = _torch.load(d / "config.pt", map_location="cpu",
                                           weights_only=False)
                    if "normalizer" in cfg_data:
                        ref_normalizer = CoordinateNormalizer.from_state_dict(
                            cfg_data["normalizer"])
                    break

            print(f"\n{'Name':<30s} {'Top1':>7s} {'TopK':>7s} "
                  f"{'Cvg':>6s} {'gn_RMSE':>9s} {'active':>7s} "
                  f"{'fallbk':>7s} {'time':>7s}")
            print("-" * 90)

            for d in all_mt:
                if not (d / "best_model.pt").exists():
                    continue
                try:
                    r = evaluate_multitask(d, pts, gt, infra["ptt_center"],
                                           device=args.device,
                                           ref_normalizer=ref_normalizer)
                    print(f"{d.name:<30s} {r['top1_acc']*100:6.2f}% "
                          f"{r['topk_acc']*100:6.2f}% {r['coverage']*100:5.1f}% "
                          f"{r['gn_rmse']:9.5f} {r['n_active']:7d} "
                          f"{r['n_fallback']:7d} {r['eval_time']:7.3f}s")
                except Exception as e:
                    print(f"{d.name:<30s}  ERROR: {e}")

            print(f"\n  C++ reference time: {cpp_time:.3f}s for {args.n_points} points")


if __name__ == "__main__":
    main()

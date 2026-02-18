"""Hyperparameter sweep for Neural-Pull on reduced data.

Runs multiple configurations on a fraction of the training data (default 10%)
for a short number of epochs (default 80), then reports a comparison table
of SDF and gradient metrics.

Supports: StEik, dual-head, GradNorm, L-BFGS, and all hyperparameter axes.

Usage:
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/sweep_neural_pull.py
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/sweep_neural_pull.py --data_fraction 0.2 --epochs 100
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/sweep_neural_pull.py --configs "steik0.1,dual_g10,gradnorm"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nn_contact.config import DataConfig, NeuralPullConfig, SIRENConfig
from nn_contact.data.loader import make_dataloaders
from nn_contact.models.neural_pull import NeuralPullNet
from nn_contact.training.losses import neural_pull_loss, GradNormBalancer, steik_loss


@dataclass
class SweepConfig:
    """One configuration to sweep."""
    name: str
    omega0: float = 10.0
    hidden_dims: list[int] = field(default_factory=lambda: [512, 512, 512, 512])
    lr: float = 1e-4
    lambda_grad: float = 10.0
    lambda_eikonal: float = 0.01
    lambda_steik: float = 0.0
    lambda_consistency: float = 0.0
    dual_head: bool = False
    gradnorm: bool = False
    lbfgs_epochs: int = 0
    grad_start: int = 5


# ── Define sweep configurations ──
SWEEP_CONFIGS = [
    # === Baseline & gradient weight sweep ===
    SweepConfig(name="baseline",     lambda_grad=1.0),
    SweepConfig(name="grad5",        lambda_grad=5.0),
    SweepConfig(name="grad10",       lambda_grad=10.0),
    SweepConfig(name="grad20",       lambda_grad=20.0),
    SweepConfig(name="grad50",       lambda_grad=50.0),

    # === StEik regularizer ===
    SweepConfig(name="steik0.01",    lambda_grad=10.0, lambda_steik=0.01),
    SweepConfig(name="steik0.1",     lambda_grad=10.0, lambda_steik=0.1),
    SweepConfig(name="steik1.0",     lambda_grad=10.0, lambda_steik=1.0),

    # === Dual-head architecture ===
    SweepConfig(name="dual_g10",     lambda_grad=10.0, dual_head=True),
    SweepConfig(name="dual_g10_c01", lambda_grad=10.0, dual_head=True, lambda_consistency=0.1),
    SweepConfig(name="dual_g50",     lambda_grad=50.0, dual_head=True),

    # === GradNorm adaptive balancing ===
    SweepConfig(name="gradnorm",     lambda_grad=10.0, gradnorm=True),
    SweepConfig(name="gradnorm_st",  lambda_grad=10.0, gradnorm=True, lambda_steik=0.1),

    # === L-BFGS refinement (on top of best baseline) ===
    SweepConfig(name="lbfgs10",      lambda_grad=10.0, lbfgs_epochs=10),
    SweepConfig(name="lbfgs20",      lambda_grad=10.0, lbfgs_epochs=20),

    # === Combined: steik + dual-head ===
    SweepConfig(name="dual_steik",   lambda_grad=10.0, dual_head=True, lambda_steik=0.1),

    # === Omega sweep ===
    SweepConfig(name="w5",           omega0=5,  lambda_grad=10.0),
    SweepConfig(name="w15",          omega0=15, lambda_grad=10.0),
    SweepConfig(name="w20",          omega0=20, lambda_grad=10.0),

    # === Architecture sweep ===
    SweepConfig(name="wide1024",     lambda_grad=10.0, hidden_dims=[1024, 512, 512, 512]),
    SweepConfig(name="deep6",        lambda_grad=10.0, hidden_dims=[512, 512, 512, 512, 512, 512]),
]


def train_one(cfg: SweepConfig, data_cfg: DataConfig, loaders, char_length: float,
              epochs: int, device: str) -> dict:
    """Train one configuration and return metrics."""
    siren_cfg = SIRENConfig(
        omega_0=cfg.omega0, omega_hidden=cfg.omega0, hidden_dims=cfg.hidden_dims,
    )
    model_cfg = NeuralPullConfig(
        architecture="siren",
        siren=siren_cfg,
        lambda_sdf=1.0,
        lambda_grad=cfg.lambda_grad,
        lambda_hess=0.0,
        lambda_eikonal=cfg.lambda_eikonal,
        lambda_steik=cfg.lambda_steik,
        lambda_consistency=cfg.lambda_consistency,
        dual_head=cfg.dual_head,
    )

    torch.manual_seed(42)
    model = NeuralPullNet.from_config(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    warmup = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    def lr_lambda(epoch):
        if epoch < warmup:
            return max(0.01, epoch / warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # GradNorm setup
    gn_balancer = None
    if cfg.gradnorm:
        n_tasks = 3  # sdf, grad, eikonal
        if cfg.lambda_steik > 0:
            n_tasks += 1
        gn_balancer = GradNormBalancer(n_tasks=n_tasks, alpha=1.5, lr=0.025)
        gn_balancer.log_weights = gn_balancer.log_weights.to(device)

    best_combined = float("inf")
    best_sdf = float("inf")
    best_grad = float("inf")
    best_grad_direct = None
    best_epoch = -1
    best_state = None

    t_start = time.time()

    for epoch in range(epochs):
        use_grad = (epoch >= cfg.grad_start)
        lam_grad = cfg.lambda_grad if use_grad else 0.0
        lam_steik = cfg.lambda_steik if use_grad else 0.0

        # ── Train ──
        model.train()
        n = 0

        for batch in loaders.train:
            xyz, patch_id, xi, gn, normal, dndxs = [t.to(device, non_blocking=True) for t in batch]
            gn = gn / char_length
            optimizer.zero_grad(set_to_none=True)
            xyz = xyz.requires_grad_(True)

            g_pred, grad_pred, grad_direct = model.forward_with_grad(xyz)

            if gn_balancer is not None and use_grad:
                loss_sdf = F.mse_loss(g_pred.squeeze(-1), gn)
                loss_grad_val = F.mse_loss(grad_pred, normal)
                grad_norm_val = grad_pred.norm(dim=-1)
                loss_eik = F.mse_loss(grad_norm_val, torch.ones_like(grad_norm_val))
                task_losses = [loss_sdf, loss_grad_val, loss_eik]
                if lam_steik > 0:
                    task_losses.append(steik_loss(grad_pred, xyz))
                shared_params = [p for p in model.parameters() if p.requires_grad]
                gn_weights = gn_balancer.step(task_losses, shared_params)
                loss = sum(w * l for w, l in zip(gn_weights, task_losses))
                if grad_direct is not None:
                    loss = loss + lam_grad * F.mse_loss(grad_direct, normal)
            else:
                loss, _ = neural_pull_loss(
                    g_pred, grad_pred, gn, normal,
                    xyz=xyz, grad_direct=grad_direct,
                    lambda_sdf=1.0, lambda_grad=lam_grad,
                    lambda_eikonal=cfg.lambda_eikonal,
                    lambda_steik=lam_steik,
                    lambda_consistency=cfg.lambda_consistency,
                )

            if not torch.isfinite(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n += 1

        if n == 0:
            continue

        # ── Validate ──
        model.eval()
        val_sdf_sum = 0.0
        val_grad_sum = 0.0
        val_grad_direct_sum = 0.0
        nv = 0
        for batch in loaders.val:
            xyz, patch_id, xi, gn, normal, dndxs = [t.to(device, non_blocking=True) for t in batch]
            gn_norm = gn / char_length
            g_pred, grad_pred, grad_direct = model.forward_with_grad(xyz)
            val_sdf_sum += F.mse_loss(g_pred.squeeze(-1), gn_norm).item()
            val_grad_sum += F.mse_loss(grad_pred, normal).item()
            if grad_direct is not None:
                val_grad_direct_sum += F.mse_loss(grad_direct, normal).item()
            nv += 1

        val_sdf = val_sdf_sum / nv
        val_grad = val_grad_sum / nv
        val_gd = val_grad_direct_sum / nv if cfg.dual_head else None
        # Use best of autodiff and direct gradient for combined metric
        val_grad_best = min(val_grad, val_gd) if val_gd is not None else val_grad
        val_combined = val_sdf + val_grad_best

        scheduler.step()

        if val_combined < best_combined:
            best_combined = val_combined
            best_sdf = val_sdf
            best_grad = val_grad
            best_grad_direct = val_gd
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            star = " *" if val_combined <= best_combined else ""
            gd_str = f" gdir={val_gd:.2e}" if val_gd is not None else ""
            print(f"    [{epoch:3d}/{epochs}] sdf={val_sdf:.2e} grad={val_grad:.2e}{gd_str}{star}")

    # ── L-BFGS refinement ──
    if cfg.lbfgs_epochs > 0 and best_state is not None:
        model.load_state_dict(best_state)
        print(f"    L-BFGS refinement ({cfg.lbfgs_epochs} epochs)...")

        # Collect training data
        all_xyz, all_gn, all_normal = [], [], []
        for batch in loaders.train:
            xyz_b, _, _, gn_b, normal_b, _ = batch
            all_xyz.append(xyz_b)
            all_gn.append(gn_b / char_length)
            all_normal.append(normal_b)
        all_xyz = torch.cat(all_xyz).to(device)
        all_gn = torch.cat(all_gn).to(device)
        all_normal = torch.cat(all_normal).to(device)

        # Subsample for memory
        max_pts = 100_000
        if len(all_xyz) > max_pts:
            idx = np.random.RandomState(42).permutation(len(all_xyz))[:max_pts]
            all_xyz, all_gn, all_normal = all_xyz[idx], all_gn[idx], all_normal[idx]

        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=20,
            history_size=50, line_search_fn="strong_wolfe",
        )

        for le in range(cfg.lbfgs_epochs):
            model.train()

            def closure():
                lbfgs.zero_grad()
                xyz = all_xyz.requires_grad_(True)
                g_pred, grad_pred, grad_direct = model.forward_with_grad(xyz)
                loss, _ = neural_pull_loss(
                    g_pred, grad_pred, all_gn, all_normal,
                    xyz=xyz, grad_direct=grad_direct,
                    lambda_sdf=1.0, lambda_grad=cfg.lambda_grad,
                    lambda_eikonal=cfg.lambda_eikonal,
                    lambda_steik=cfg.lambda_steik,
                    lambda_consistency=cfg.lambda_consistency,
                )
                loss.backward()
                return loss

            lbfgs.step(closure)

            # Re-validate
            model.eval()
            vs, vg, vgd_sum, nv2 = 0.0, 0.0, 0.0, 0
            for batch in loaders.val:
                xyz, _, _, gn, normal, _ = [t.to(device, non_blocking=True) for t in batch]
                g_pred, grad_pred, grad_direct = model.forward_with_grad(xyz)
                vs += F.mse_loss(g_pred.squeeze(-1), gn / char_length).item()
                vg += F.mse_loss(grad_pred, normal).item()
                if grad_direct is not None:
                    vgd_sum += F.mse_loss(grad_direct, normal).item()
                nv2 += 1
            val_sdf = vs / nv2
            val_grad = vg / nv2
            val_gd = vgd_sum / nv2 if cfg.dual_head else None
            val_grad_best = min(val_grad, val_gd) if val_gd is not None else val_grad
            val_combined = val_sdf + val_grad_best

            if val_combined < best_combined:
                best_combined = val_combined
                best_sdf = val_sdf
                best_grad = val_grad
                best_grad_direct = val_gd
                best_epoch = f"L{le}"

            if le % 5 == 0 or le == cfg.lbfgs_epochs - 1:
                print(f"    [LBFGS {le:2d}] sdf={val_sdf:.2e} grad={val_grad:.2e}")

    dt = time.time() - t_start
    # Use best gradient metric (direct head if available and better)
    best_grad_for_metric = best_grad
    if best_grad_direct is not None and best_grad_direct < best_grad:
        best_grad_for_metric = best_grad_direct
    grad_rmse = math.sqrt(best_grad_for_metric * 3)  # MSE per component -> L2 RMSE
    sdf_rmse = math.sqrt(best_sdf)
    angle_approx = math.degrees(math.asin(min(1.0, grad_rmse)))

    return {
        "name": cfg.name,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "val_sdf_mse": best_sdf,
        "val_grad_mse": best_grad,
        "val_grad_direct_mse": best_grad_direct,
        "sdf_rmse_norm": sdf_rmse,
        "grad_l2_rmse": grad_rmse,
        "angle_deg": angle_approx,
        "time_s": dt,
    }


def main():
    parser = argparse.ArgumentParser(description="Neural-Pull hyperparameter sweep")
    parser.add_argument("--data_fraction", type=float, default=0.1,
                        help="Fraction of data (default 0.1 = 10%%)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Epochs per configuration")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names to run (default: all)")
    parser.add_argument("--gn_min", type=float, default=-0.5)
    parser.add_argument("--gn_max", type=float, default=1.5)
    args = parser.parse_args()

    # Select configs
    if args.configs:
        names = set(args.configs.split(","))
        configs = [c for c in SWEEP_CONFIGS if c.name in names]
        if not configs:
            print(f"No configs matched: {names}")
            print(f"Available: {[c.name for c in SWEEP_CONFIGS]}")
            return
    else:
        configs = SWEEP_CONFIGS

    print(f"Sweep: {len(configs)} configs, {args.data_fraction:.0%} data, "
          f"{args.epochs} epochs, device={args.device}")
    print(f"Gap filter: [{args.gn_min}, {args.gn_max}]")
    print()

    # Load data once
    data_cfg = DataConfig(
        batch_size=args.batch_size, num_workers=4, pin_memory=True,
        gn_min=args.gn_min, gn_max=args.gn_max,
    )
    loaders = make_dataloaders(data_cfg, seed=42, verbose=True, data_fraction=args.data_fraction)
    char_length = data_cfg.char_length

    results = []
    for i, cfg in enumerate(configs):
        features = []
        if cfg.dual_head:
            features.append("dual")
        if cfg.gradnorm:
            features.append("GN")
        if cfg.lambda_steik > 0:
            features.append(f"steik={cfg.lambda_steik}")
        if cfg.lbfgs_epochs > 0:
            features.append(f"lbfgs={cfg.lbfgs_epochs}")
        feat_str = f" [{', '.join(features)}]" if features else ""

        print(f"\n[{i+1}/{len(configs)}] {cfg.name}: ω₀={cfg.omega0}, lr={cfg.lr}, "
              f"λ_grad={cfg.lambda_grad}, λ_eik={cfg.lambda_eikonal}, "
              f"dims={cfg.hidden_dims}{feat_str}")

        res = train_one(cfg, data_cfg, loaders, char_length, args.epochs, args.device)
        results.append(res)

        gd_str = ""
        if res["val_grad_direct_mse"] is not None:
            gd_rmse = math.sqrt(res["val_grad_direct_mse"] * 3)
            gd_str = f"  grad_direct_rmse={gd_rmse:.4f}"

        print(f"  -> sdf_rmse={res['sdf_rmse_norm']:.2e}  grad_l2_rmse={res['grad_l2_rmse']:.4f}"
              f"{gd_str}  angle≈{res['angle_deg']:.2f}°  best@{res['best_epoch']}  ({res['time_s']:.0f}s)")

    # ── Summary table ──
    print("\n" + "=" * 110)
    print(f"{'Config':<16} {'Params':>8} {'SDF RMSE':>10} {'Grad L2 RMSE':>13} "
          f"{'GDir RMSE':>10} {'Angle(°)':>9} {'Best@':>6} {'Time':>6}")
    print("-" * 110)

    # Sort by grad_l2_rmse (the metric we care about)
    results.sort(key=lambda r: r["grad_l2_rmse"])

    for r in results:
        gd_col = "—"
        if r["val_grad_direct_mse"] is not None:
            gd_col = f"{math.sqrt(r['val_grad_direct_mse'] * 3):.5f}"
        print(f"{r['name']:<16} {r['n_params']:>8,} {r['sdf_rmse_norm']:>10.2e} "
              f"{r['grad_l2_rmse']:>13.5f} {gd_col:>10} {r['angle_deg']:>9.3f} "
              f"{str(r['best_epoch']):>6} {r['time_s']:>5.0f}s")

    # Save results
    out_path = Path("nn_contact/checkpoints/sweep_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

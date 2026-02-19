"""Train the Neural-Pull SDF network (Phase 2).

Learns g_nn(x_norm) = g_phys(x_raw) / L such that:
  - g_nn ≈ signed distance / char_length  (dimensionless)
  - ∇g_nn ≈ outward surface normal n*     (autodiff in normalized coords)
  - ∇²g_nn ≈ L * dn*/dx_s                 (Hessian in normalized coords)

COORDINATE NORMALIZATION MATH:
  x_norm = (x_raw - center) / L,  g_nn = g_phys / L
  ∂g_nn/∂x_norm = (∂x_raw/∂x_norm) * (1/L) * ∂g_phys/∂x_raw = L * (1/L) * n = n  ✓
  |∇g_nn| = |n| = 1  (eikonal constraint satisfied naturally)
  ∂²g_nn/∂x_norm² = L * ∂n/∂x_raw  (Hessian targets must be scaled by L)

Uses SIREN architecture (sin activations) with curriculum training:
  Phase A (warmup):     L_sdf + L_eikonal only
  Phase B (gradients):  + L_grad (normal supervision)
  Phase C (Hessians):   + L_hess (dn/dx_s supervision) [optional, off by default]

Features:
  - Gradient-focused checkpointing: val_sdf + val_grad_weight * val_grad
  - StEik regularizer: nᵀ∇²gn curvature penalty (NeurIPS 2023)
  - Dual-head: explicit gradient output head alongside SDF head
  - GradNorm: adaptive loss balancing (ICML 2018)
  - L-BFGS refinement: quasi-Newton fine-tuning after Adam converges

IMPORTANT: autodiff with create_graph=True requires float32 — no AMP autocast.

Usage:
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/train_neural_pull.py
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/train_neural_pull.py --steik 0.1
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/train_neural_pull.py --dual_head --lambda_grad 10
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/train_neural_pull.py --gradnorm
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/train_neural_pull.py --lbfgs_epochs 20
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nn_contact.config import DataConfig, NeuralPullConfig
from nn_contact.data.loader import make_dataloaders
from nn_contact.models.neural_pull import NeuralPullNet
from nn_contact.training.losses import neural_pull_loss, GradNormBalancer


def _validate(model, loaders, char_length, device):
    """Compute validation SDF and gradient MSE."""
    model.eval()
    val_sdf_sum = 0.0
    val_grad_sum = 0.0
    val_grad_direct_sum = 0.0
    nv = 0
    has_dual = model.dual_head
    for batch in loaders.val:
        xyz, patch_id, xi, gn, normal, dndxs = [t.to(device, non_blocking=True) for t in batch]
        gn_norm = gn / char_length

        g_pred, grad_pred, grad_direct = model.forward_with_grad(xyz)
        val_sdf_sum += F.mse_loss(g_pred.squeeze(-1), gn_norm).item()
        val_grad_sum += F.mse_loss(grad_pred, normal).item()
        if has_dual and grad_direct is not None:
            val_grad_direct_sum += F.mse_loss(grad_direct, normal).item()
        nv += 1

    val_sdf = val_sdf_sum / nv if nv > 0 else float("inf")
    val_grad = val_grad_sum / nv if nv > 0 else float("inf")
    val_grad_direct = val_grad_direct_sum / nv if nv > 0 and has_dual else None
    return val_sdf, val_grad, val_grad_direct


def train(args):
    device = args.device
    epochs = args.epochs
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    # ── Config ──
    from nn_contact.config import SIRENConfig
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")] if args.hidden_dims else [512, 512, 512, 512]
    siren_cfg = SIRENConfig(
        omega_0=args.omega0, omega_hidden=args.omega0,
        hidden_dims=hidden_dims, h_siren=args.h_siren,
    )

    model_cfg = NeuralPullConfig(
        architecture=args.arch,
        siren=siren_cfg,
        lambda_sdf=1.0,
        lambda_grad=args.lambda_grad,
        lambda_hess=args.lambda_hess,
        lambda_eikonal=args.lambda_eikonal,
        lambda_steik=args.steik,
        lambda_gh_align=args.gh_align,
        lambda_consistency=args.consistency,
        dual_head=args.dual_head,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    data_cfg = DataConfig(
        batch_size=args.batch_size, num_workers=4, pin_memory=True,
        gn_min=args.gn_min, gn_max=args.gn_max,
    )
    loaders = make_dataloaders(data_cfg, seed=42, verbose=True, data_fraction=args.data_fraction)
    char_length = data_cfg.char_length  # 8.0

    # ── Target normalization ──
    print(f"Target normalization: gn/={char_length}, dndxs*={char_length}", flush=True)

    # ── Model ──
    model = NeuralPullNet.from_config(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # ── Resume ──
    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        ckpt_path = ckpt_dir / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            start_epoch = ckpt["epoch"] + 1
            best_val = ckpt["best_val_loss"]
            print(f"Resumed from epoch {start_epoch}, best val={best_val:.6f}", flush=True)
        else:
            print(f"No checkpoint at {ckpt_path}, starting fresh", flush=True)

    print(f"Model ({args.arch}): {n_params:,} params, device={device}", flush=True)
    if args.dual_head:
        print(f"Dual-head mode: explicit gradient output head enabled", flush=True)

    # ── Curriculum schedule ──
    grad_start = args.grad_start
    hess_start = args.hess_start
    print(f"Curriculum: grad at epoch {grad_start}, hess at epoch {hess_start}", flush=True)
    sched_str = f" → {args.lambda_grad_final}" if args.lambda_grad_final is not None else ""
    print(f"Weights: sdf={model_cfg.lambda_sdf}, grad={model_cfg.lambda_grad}{sched_str}, "
          f"hess={model_cfg.lambda_hess}, eik={model_cfg.lambda_eikonal}, "
          f"steik={model_cfg.lambda_steik}, gh_align={model_cfg.lambda_gh_align}", flush=True)
    if args.h_siren:
        print("H-SIREN: sin(sinh(2ωx)) first layer enabled", flush=True)

    # ── GradNorm adaptive balancing ──
    gradnorm = None
    if args.gradnorm:
        n_tasks = 3  # sdf, grad, eikonal
        if model_cfg.lambda_steik > 0:
            n_tasks += 1
        if model_cfg.lambda_gh_align > 0:
            n_tasks += 1
        gradnorm = GradNormBalancer(n_tasks=n_tasks, alpha=1.5, lr=0.025)
        gradnorm.log_weights = gradnorm.log_weights.to(device)
        print(f"GradNorm enabled: {n_tasks} tasks, alpha=1.5", flush=True)

    # ── Optimizer ──
    warmup = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def lr_lambda(epoch):
        if epoch < warmup:
            return max(0.01, epoch / warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for _ in range(start_epoch):
        scheduler.step()

    patience_counter = 0
    patience = args.patience

    # ── Training loop ──
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        use_grad = (epoch >= grad_start)
        use_hess = (epoch >= hess_start) and (model_cfg.lambda_hess > 0)

        # Loss scheduling: decay lambda_grad from initial to final
        if args.lambda_grad_final is not None and use_grad:
            t = (epoch - grad_start) / max(1, epochs - 1 - grad_start)
            t = min(1.0, max(0.0, t))
            s = 6*t**5 - 15*t**4 + 10*t**3  # smooth quintic interpolation
            lam_grad = model_cfg.lambda_grad * (1 - s) + args.lambda_grad_final * s
        else:
            lam_grad = model_cfg.lambda_grad if use_grad else 0.0

        lam_hess = model_cfg.lambda_hess if use_hess else 0.0
        lam_steik = model_cfg.lambda_steik if use_grad else 0.0
        lam_gh_align = model_cfg.lambda_gh_align if use_grad else 0.0

        # ── Train ──
        model.train()
        train_total = 0.0
        train_bd = {}
        n = 0

        for batch in loaders.train:
            xyz, patch_id, xi, gn, normal, dndxs = [t.to(device, non_blocking=True) for t in batch]

            gn = gn / char_length
            dndxs = dndxs * char_length

            optimizer.zero_grad(set_to_none=True)
            xyz = xyz.requires_grad_(True)

            if use_hess:
                g_pred, grad_pred, hess_pred = model.forward_with_hessian(xyz)
                grad_direct = None
            else:
                g_pred, grad_pred, grad_direct = model.forward_with_grad(xyz)
                hess_pred = None

            if gradnorm is not None and use_grad:
                # GradNorm: compute individual losses, let balancer set weights
                loss_sdf = F.mse_loss(g_pred.squeeze(-1), gn)
                loss_grad_val = F.mse_loss(grad_pred, normal)
                grad_norm_val = grad_pred.norm(dim=-1)
                loss_eik = F.mse_loss(grad_norm_val, torch.ones_like(grad_norm_val))

                task_losses = [loss_sdf, loss_grad_val, loss_eik]
                if lam_steik > 0 and xyz is not None:
                    from nn_contact.training.losses import steik_loss as _steik
                    task_losses.append(_steik(grad_pred, xyz))
                if lam_gh_align > 0 and xyz is not None:
                    from nn_contact.training.losses import gh_alignment_loss as _gh
                    task_losses.append(_gh(grad_pred, xyz))

                shared_params = [p for p in model.parameters() if p.requires_grad]
                gn_weights = gradnorm.step(task_losses, shared_params)

                loss = sum(w * l for w, l in zip(gn_weights, task_losses))
                if grad_direct is not None:
                    loss = loss + lam_grad * F.mse_loss(grad_direct, normal)
                    if model_cfg.lambda_consistency > 0:
                        loss = loss + model_cfg.lambda_consistency * F.mse_loss(grad_pred.detach(), grad_direct)

                bd = {
                    "loss_sdf": loss_sdf.item(),
                    "loss_grad": loss_grad_val.item(),
                    "loss_eikonal": loss_eik.item(),
                    "loss_total": loss.item(),
                    "gn_w_sdf": gn_weights[0].item(),
                    "gn_w_grad": gn_weights[1].item(),
                }
            else:
                loss, bd = neural_pull_loss(
                    g_pred, grad_pred, gn, normal,
                    xyz=xyz,
                    grad_direct=grad_direct,
                    hess_pred=hess_pred,
                    dndxs_target=dndxs if use_hess else None,
                    lambda_sdf=model_cfg.lambda_sdf,
                    lambda_grad=lam_grad,
                    lambda_hess=lam_hess,
                    lambda_eikonal=model_cfg.lambda_eikonal,
                    lambda_steik=lam_steik,
                    lambda_gh_align=lam_gh_align,
                    lambda_consistency=model_cfg.lambda_consistency,
                )

            if not torch.isfinite(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_total += loss.item()
            for k, v in bd.items():
                train_bd[k] = train_bd.get(k, 0) + v
            n += 1

        if n == 0:
            print(f"[{epoch:3d}/{epochs}] ALL BATCHES NaN", flush=True)
            continue

        train_total /= n
        train_bd = {k: v / n for k, v in train_bd.items()}

        # ── Validate ──
        val_sdf, val_grad, val_grad_direct = _validate(model, loaders, char_length, device)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # Checkpoint on combined metric
        # For dual-head, use the best of autodiff and direct gradient
        val_grad_best = min(val_grad, val_grad_direct) if val_grad_direct is not None else val_grad
        val_combined = val_sdf + args.val_grad_weight * val_grad_best
        is_best = val_combined < best_val
        if is_best:
            best_val = val_combined
            patience_counter = 0
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val,
                "best_val_sdf": val_sdf,
                "best_val_grad": val_grad,
                "best_val_grad_direct": val_grad_direct,
                "config": model_cfg,
                "char_length": char_length,
            }
            torch.save(state, ckpt_dir / "best_model.pt")
            torch.save(state, ckpt_dir / "checkpoint.pt")
        else:
            patience_counter += 1

        dt = time.time() - t0
        star = " *" if is_best else ""
        phase = "C(hess)" if use_hess else ("B(grad)" if use_grad else "A(sdf)")

        if epoch % 10 == 0 or is_best or epoch < start_epoch + 5 or epoch in (grad_start, hess_start):
            extras = []
            if use_grad:
                extras.append(f"grad={train_bd.get('loss_grad', 0):.3e}")
            if use_hess:
                extras.append(f"hess={train_bd.get('loss_hess', 0):.3e}")
            if lam_steik > 0:
                extras.append(f"steik={train_bd.get('loss_steik', 0):.3e}")
            if lam_gh_align > 0:
                extras.append(f"gh={train_bd.get('loss_gh_align', 0):.3e}")
            if "loss_grad_direct" in train_bd:
                extras.append(f"gdir={train_bd['loss_grad_direct']:.3e}")
            if "gn_w_sdf" in train_bd:
                extras.append(f"GN[{train_bd['gn_w_sdf']:.2f},{train_bd['gn_w_grad']:.2f}]")
            extra_str = " " + " ".join(extras) if extras else ""

            vgd = f" vgd={val_grad_direct:.4e}" if val_grad_direct is not None else ""
            print(
                f"[{epoch:3d}/{epochs}] [{phase}] "
                f"train={train_total:.4e} (sdf={train_bd.get('loss_sdf', 0):.3e} "
                f"eik={train_bd.get('loss_eikonal', 0):.3e}{extra_str}) "
                f"val_sdf={val_sdf:.4e} val_grad={val_grad:.4e}{vgd} "
                f"lr={lr:.2e} dt={dt:.0f}s{star}",
                flush=True,
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})", flush=True)
            break

    # ── L-BFGS refinement phase ──
    if args.lbfgs_epochs > 0:
        print(f"\n── L-BFGS refinement ({args.lbfgs_epochs} epochs) ──", flush=True)

        # Reload best model
        ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        best_val = ckpt["best_val_loss"]
        print(f"Loaded best model (epoch {ckpt['epoch']}, val={best_val:.6e})", flush=True)

        # Collect all training data into a single batch (or large chunks)
        all_xyz, all_gn, all_normal = [], [], []
        for batch in loaders.train:
            xyz_b, _, _, gn_b, normal_b, _ = batch
            all_xyz.append(xyz_b)
            all_gn.append(gn_b / char_length)
            all_normal.append(normal_b)
        all_xyz = torch.cat(all_xyz).to(device)
        all_gn = torch.cat(all_gn).to(device)
        all_normal = torch.cat(all_normal).to(device)
        print(f"L-BFGS data: {len(all_xyz):,} points", flush=True)

        # Split into chunks if too large for memory (max ~200k points)
        max_lbfgs_pts = 200_000
        if len(all_xyz) > max_lbfgs_pts:
            rng = np.random.RandomState(42)
            idx = rng.permutation(len(all_xyz))[:max_lbfgs_pts]
            all_xyz = all_xyz[idx]
            all_gn = all_gn[idx]
            all_normal = all_normal[idx]
            print(f"  subsampled to {max_lbfgs_pts:,} for L-BFGS memory", flush=True)

        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=20,
            history_size=50, line_search_fn="strong_wolfe",
        )

        for lbfgs_epoch in range(args.lbfgs_epochs):
            t0 = time.time()
            model.train()

            def closure():
                lbfgs.zero_grad()
                xyz = all_xyz.requires_grad_(True)
                g_pred, grad_pred, grad_direct = model.forward_with_grad(xyz)
                loss, _ = neural_pull_loss(
                    g_pred, grad_pred, all_gn, all_normal,
                    xyz=xyz,
                    grad_direct=grad_direct,
                    lambda_sdf=model_cfg.lambda_sdf,
                    lambda_grad=model_cfg.lambda_grad,
                    lambda_eikonal=model_cfg.lambda_eikonal,
                    lambda_steik=model_cfg.lambda_steik,
                    lambda_gh_align=model_cfg.lambda_gh_align,
                    lambda_consistency=model_cfg.lambda_consistency,
                )
                loss.backward()
                return loss

            loss_val = lbfgs.step(closure)

            val_sdf, val_grad, val_grad_direct = _validate(model, loaders, char_length, device)
            val_grad_best = min(val_grad, val_grad_direct) if val_grad_direct is not None else val_grad
            val_combined = val_sdf + args.val_grad_weight * val_grad_best

            dt = time.time() - t0
            star = ""
            if val_combined < best_val:
                best_val = val_combined
                star = " *"
                state = {
                    "epoch": f"lbfgs_{lbfgs_epoch}",
                    "model_state": model.state_dict(),
                    "best_val_loss": best_val,
                    "best_val_sdf": val_sdf,
                    "best_val_grad": val_grad,
                    "best_val_grad_direct": val_grad_direct,
                    "config": model_cfg,
                    "char_length": char_length,
                }
                torch.save(state, ckpt_dir / "best_model.pt")

            vgd = f" vgd={val_grad_direct:.4e}" if val_grad_direct is not None else ""
            print(
                f"[LBFGS {lbfgs_epoch:2d}/{args.lbfgs_epochs}] "
                f"loss={loss_val:.4e} val_sdf={val_sdf:.4e} val_grad={val_grad:.4e}{vgd} "
                f"dt={dt:.0f}s{star}",
                flush=True,
            )

    # Save config
    torch.save(
        {"config": model_cfg, "data_config": data_cfg, "normalizer": loaders.normalizer.state_dict()},
        ckpt_dir / "config.pt",
    )
    print(f"\nDone ({args.arch}). Best val combined: {best_val:.6e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train Neural-Pull SDF network (Phase 2)")
    # Architecture
    parser.add_argument("--arch", default="siren", choices=["siren", "fourier_mlp", "mlp"])
    parser.add_argument("--omega0", type=float, default=10.0,
                        help="SIREN omega_0 frequency (default 10; 30 is unstable)")
    parser.add_argument("--hidden_dims", type=str, default=None,
                        help="Comma-separated hidden dims (e.g. '1024,512,512,512')")
    parser.add_argument("--dual_head", action="store_true",
                        help="Enable dual-head: explicit gradient output alongside SDF")
    parser.add_argument("--h_siren", action="store_true",
                        help="Use H-SIREN: sin(sinh(2ωx)) first layer for broader frequency support")
    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Loss weights — gradient-focused defaults
    parser.add_argument("--lambda_grad", type=float, default=10.0,
                        help="Gradient supervision weight (high to prioritize normals)")
    parser.add_argument("--lambda_grad_final", type=float, default=None,
                        help="If set, schedule lambda_grad from initial to this value (e.g. 10)")
    parser.add_argument("--lambda_hess", type=float, default=0.0,
                        help="Hessian supervision weight (0 = disabled)")
    parser.add_argument("--lambda_eikonal", type=float, default=0.01,
                        help="Eikonal regularization weight")
    parser.add_argument("--steik", type=float, default=0.0,
                        help="StEik curvature regularizer weight (e.g. 0.1)")
    parser.add_argument("--gh_align", type=float, default=0.0,
                        help="GH-alignment ‖Hn‖²=0 regularizer weight (e.g. 0.1)")
    parser.add_argument("--consistency", type=float, default=0.0,
                        help="Dual-head consistency loss weight (e.g. 0.1)")
    # GradNorm
    parser.add_argument("--gradnorm", action="store_true",
                        help="Enable GradNorm adaptive loss balancing")
    # Curriculum
    parser.add_argument("--grad_start", type=int, default=5,
                        help="Epoch to start gradient supervision")
    parser.add_argument("--hess_start", type=int, default=9999,
                        help="Epoch to start Hessian supervision (9999 = disabled)")
    # Data filtering
    parser.add_argument("--gn_min", type=float, default=-0.5,
                        help="Min gap for data filtering")
    parser.add_argument("--gn_max", type=float, default=1.5,
                        help="Max gap for data filtering")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of data to use (e.g. 0.1 for 10%% fast iteration)")
    # L-BFGS refinement
    parser.add_argument("--lbfgs_epochs", type=int, default=0,
                        help="Number of L-BFGS refinement epochs after Adam (0 = disabled)")
    # Validation / checkpointing
    parser.add_argument("--val_grad_weight", type=float, default=1.0,
                        help="Weight for val_grad in combined checkpoint metric")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--checkpoint_dir", default="nn_contact/checkpoints/neural_pull")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

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
  Phase C (Hessians):   + L_hess (dn/dx_s supervision)

IMPORTANT: autodiff with create_graph=True requires float32 — no AMP autocast.

Usage:
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/train_neural_pull.py
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/train_neural_pull.py --omega0 10 --lr 5e-4
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/train_neural_pull.py --resume
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
from nn_contact.training.losses import neural_pull_loss


def train(args):
    device = args.device
    epochs = args.epochs
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    # ── Config ──
    # Override SIREN omega if specified
    from nn_contact.config import SIRENConfig
    siren_cfg = SIRENConfig(omega_0=args.omega0, omega_hidden=args.omega0)

    model_cfg = NeuralPullConfig(
        architecture=args.arch,
        siren=siren_cfg,
        lambda_sdf=1.0,
        lambda_grad=args.lambda_grad,
        lambda_hess=args.lambda_hess,
        lambda_eikonal=args.lambda_eikonal,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    # Smaller batch for Hessian (create_graph=True is memory-hungry)
    data_cfg = DataConfig(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    loaders = make_dataloaders(data_cfg, seed=42, verbose=True)
    char_length = data_cfg.char_length  # 8.0

    # ── Target normalization ──
    # g_nn(x_norm) = g_phys / L  → SDF target divided by char_length
    # ∇g_nn = n                  → normal target unchanged
    # ∇²g_nn = L * dn/dx        → dndxs target multiplied by char_length
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

    # ── Curriculum schedule ──
    # Phase A: epochs [0, grad_start)    — SDF + eikonal only
    # Phase B: epochs [grad_start, hess_start) — + gradient supervision
    # Phase C: epochs [hess_start, ...)  — + Hessian supervision
    grad_start = args.grad_start
    hess_start = args.hess_start
    print(f"Curriculum: grad at epoch {grad_start}, hess at epoch {hess_start}", flush=True)
    print(f"Weights: sdf={model_cfg.lambda_sdf}, grad={model_cfg.lambda_grad}, "
          f"hess={model_cfg.lambda_hess}, eik={model_cfg.lambda_eikonal}", flush=True)

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

    # No AMP scaler — autodiff with create_graph=True needs float32
    patience_counter = 0
    patience = args.patience

    # ── Training loop ──
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Determine active losses for this epoch
        use_grad = (epoch >= grad_start)
        use_hess = (epoch >= hess_start) and (model_cfg.lambda_hess > 0)

        # Current effective weights
        lam_grad = model_cfg.lambda_grad if use_grad else 0.0
        lam_hess = model_cfg.lambda_hess if use_hess else 0.0

        # ── Train ──
        model.train()
        train_total = 0.0
        train_bd = {}
        n = 0

        for batch in loaders.train:
            xyz, patch_id, xi, gn, normal, dndxs = [t.to(device, non_blocking=True) for t in batch]

            # Normalize targets to match network output in normalized coordinates
            gn = gn / char_length          # g_nn = g_phys / L
            # normal stays as-is            # ∇g_nn = n (exact)
            dndxs = dndxs * char_length    # ∇²g_nn = L * dn/dx

            optimizer.zero_grad(set_to_none=True)

            # Forward with autodiff — MUST be float32, no AMP autocast
            xyz = xyz.requires_grad_(True)

            if use_hess:
                g_pred, grad_pred, hess_pred = model.forward_with_hessian(xyz)
                loss, bd = neural_pull_loss(
                    g_pred, grad_pred, gn, normal,
                    hess_pred=hess_pred, dndxs_target=dndxs,
                    lambda_sdf=model_cfg.lambda_sdf,
                    lambda_grad=lam_grad,
                    lambda_hess=lam_hess,
                    lambda_eikonal=model_cfg.lambda_eikonal,
                )
            elif use_grad:
                g_pred, grad_pred = model.forward_with_grad(xyz)
                loss, bd = neural_pull_loss(
                    g_pred, grad_pred, gn, normal,
                    lambda_sdf=model_cfg.lambda_sdf,
                    lambda_grad=lam_grad,
                    lambda_eikonal=model_cfg.lambda_eikonal,
                )
            else:
                # Phase A: SDF + eikonal only (still need grad for eikonal)
                g_pred, grad_pred = model.forward_with_grad(xyz)
                loss, bd = neural_pull_loss(
                    g_pred, grad_pred, gn, normal,
                    lambda_sdf=model_cfg.lambda_sdf,
                    lambda_grad=0.0,
                    lambda_eikonal=model_cfg.lambda_eikonal,
                )

            # NaN guard
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
        model.eval()
        val_sdf_sum = 0.0
        val_grad_sum = 0.0
        nv = 0
        do_grad_val = (epoch % 10 == 0) or (epoch == grad_start) or (epoch == hess_start)
        for batch in loaders.val:
            xyz, patch_id, xi, gn, normal, dndxs = [t.to(device, non_blocking=True) for t in batch]
            gn_norm = gn / char_length

            if do_grad_val:
                # Every 10 epochs: also check gradient accuracy
                g_pred, grad_pred = model.forward_with_grad(xyz)
                val_sdf_sum += F.mse_loss(g_pred.squeeze(-1), gn_norm).item()
                val_grad_sum += F.mse_loss(grad_pred, normal).item()
            else:
                with torch.no_grad():
                    g_pred = model(xyz)
                    val_sdf_sum += F.mse_loss(g_pred.squeeze(-1), gn_norm).item()

            nv += 1
        val_sdf = val_sdf_sum / nv if nv > 0 else float("inf")
        val_grad = val_grad_sum / nv if nv > 0 and do_grad_val else None

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # Checkpoint
        is_best = val_sdf < best_val
        if is_best:
            best_val = val_sdf
            patience_counter = 0
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val,
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
            hess_info = f" hess={train_bd.get('loss_hess', 0):.3e}" if use_hess else ""
            grad_info = f" grad={train_bd.get('loss_grad', 0):.3e}" if use_grad else ""
            vg_info = f" val_grad={val_grad:.4e}" if val_grad is not None else ""
            print(
                f"[{epoch:3d}/{epochs}] [{phase}] "
                f"train={train_total:.4e} (sdf={train_bd.get('loss_sdf', 0):.3e} "
                f"eik={train_bd.get('loss_eikonal', 0):.3e}{grad_info}{hess_info}) "
                f"val_sdf={val_sdf:.4e}{vg_info} lr={lr:.2e} dt={dt:.0f}s{star}",
                flush=True,
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})", flush=True)
            break

    # Save config
    torch.save(
        {"config": model_cfg, "data_config": data_cfg, "normalizer": loaders.normalizer.state_dict()},
        ckpt_dir / "config.pt",
    )
    print(f"\nDone ({args.arch}). Best val SDF loss: {best_val:.6e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train Neural-Pull SDF network (Phase 2)")
    parser.add_argument("--arch", default="siren", choices=["siren", "fourier_mlp", "mlp"])
    parser.add_argument("--omega0", type=float, default=30.0,
                        help="SIREN omega_0 frequency (default 30, try 5-10 for smooth SDF)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda_grad", type=float, default=1.0)
    parser.add_argument("--lambda_hess", type=float, default=0.1)
    parser.add_argument("--lambda_eikonal", type=float, default=0.1)
    parser.add_argument("--grad_start", type=int, default=20,
                        help="Epoch to start gradient supervision")
    parser.add_argument("--hess_start", type=int, default=60,
                        help="Epoch to start Hessian supervision")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--checkpoint_dir", default="nn_contact/checkpoints/neural_pull")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

"""Unified training script for multi-task contact NN.

Variants:
  v1  — Baseline: ReLU + trunk [512,512,256,256], no Fourier, equal weights
  v2  — Fourier encoding (128 freq) + SiLU + trunk [512,512,512,256], equal weights
  v2b — Bigger heads + more frequencies (256) + hand-tuned loss weights
  v3  — v2 architecture + GradNorm adaptive loss balancing

Features: AMP mixed precision, warmup+cosine LR, NaN guards, resume support.

Usage:
    # Fresh training
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/train_multitask_v2.py --variant v3

    # Resume from checkpoint
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/train_multitask_v2.py --variant v3 --resume

    # Custom settings
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/train_multitask_v2.py --variant v2b --epochs 300 --batch_size 4096
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nn_contact.config import DataConfig, FourierMLPConfig, MLPConfig, MultiTaskConfig
from nn_contact.data.loader import make_dataloaders
from nn_contact.models.multitask import MultiTaskContactNet
from nn_contact.training.losses import segmented_regression_loss


# ── Variant configs ──────────────────────────────────────────────────────────

def make_v1_config() -> MultiTaskConfig:
    """Baseline: ReLU + smaller trunk, no Fourier encoding."""
    return MultiTaskConfig()  # all defaults


def make_v2_config() -> MultiTaskConfig:
    """Fourier + SiLU + bigger trunk, equal loss weights."""
    return MultiTaskConfig(
        trunk=MLPConfig(hidden_dims=[512, 512, 512, 256], activation="silu", skip_connections=True),
        gn_head_dims=[128, 64],
        patch_head_dims=[256, 128],
        proj_head_dims=[256, 128],
        input_encoding="fourier",
        fourier_config=FourierMLPConfig(n_frequencies=128, frequency_scale=10.0),
    )


def make_v2b_config() -> MultiTaskConfig:
    """Bigger heads + more frequencies + hand-tuned loss weights."""
    return MultiTaskConfig(
        trunk=MLPConfig(hidden_dims=[512, 512, 512, 256], activation="silu", skip_connections=True),
        gn_head_dims=[256, 128],
        patch_head_dims=[512, 256],
        proj_head_dims=[512, 256],
        input_encoding="fourier",
        fourier_config=FourierMLPConfig(n_frequencies=256, frequency_scale=15.0),
        lambda_gn=1.0,
        lambda_patch=2.0,
        lambda_proj=1.5,
    )


def make_v3_config() -> MultiTaskConfig:
    """v2 architecture + GradNorm adaptive loss balancing."""
    return make_v2_config()


VARIANT_CONFIGS = {
    "v1": make_v1_config,
    "v2": make_v2_config,
    "v2b": make_v2b_config,
    "v3": make_v3_config,
}


# ── Loss computation ─────────────────────────────────────────────────────────

def compute_individual_losses(outputs, gn, patch_id, xi):
    """Return 3 individual task losses. Clamps logits to prevent AMP overflow."""
    loss_gn = F.mse_loss(outputs["gn_pred"].squeeze(-1), gn)
    logits = outputs["patch_logits"].float().clamp(-30, 30)
    loss_patch = F.cross_entropy(logits, patch_id)
    loss_proj = segmented_regression_loss(outputs["xi_pred"], xi, patch_id)
    return loss_gn, loss_patch, loss_proj


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = args.device
    variant = args.variant
    epochs = args.epochs
    use_gradnorm = (variant == "v3")

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    # Config
    model_cfg = VARIANT_CONFIGS[variant]()
    ckpt_dir = Path(f"nn_contact/checkpoints/multitask_{variant}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data
    data_cfg = DataConfig(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    loaders = make_dataloaders(data_cfg, seed=42, verbose=True)

    # Model
    model = MultiTaskContactNet.from_config(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # GradNorm state (v3 only)
    log_weights = nn.Parameter(torch.zeros(3, device=device)) if use_gradnorm else None
    gradnorm_alpha = 1.5
    gradnorm_lr = 0.025
    initial_losses = None

    # Resume
    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        ckpt_path = ckpt_dir / "best_model.pt"
        if not ckpt_path.exists():
            print(f"No checkpoint at {ckpt_path}, starting fresh", flush=True)
        else:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            start_epoch = ckpt["epoch"] + 1
            best_val = ckpt["best_val_loss"]
            if use_gradnorm and "log_weights" in ckpt:
                log_weights.data.copy_(ckpt["log_weights"].to(device))
                print(f"  Restored GradNorm log_weights: {log_weights.data.tolist()}", flush=True)
            print(f"Resumed from epoch {start_epoch}, best val={best_val:.6f}", flush=True)

    print(f"Model ({variant}): {n_params:,} params, device={device}", flush=True)
    if use_gradnorm:
        print(f"  GradNorm enabled (alpha={gradnorm_alpha}, lr={gradnorm_lr})", flush=True)

    # Optimizer
    warmup = 10
    if use_gradnorm:
        optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
            {"params": [log_weights], "lr": gradnorm_lr, "weight_decay": 0},
        ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < warmup:
            return max(0.01, epoch / warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for _ in range(start_epoch):
        scheduler.step()

    scaler = torch.amp.GradScaler("cuda")
    patience_counter = 0
    patience = 40

    # ── Epoch loop ──
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_total = 0.0
        train_bd = {"loss_gn": 0, "loss_patch": 0, "loss_proj": 0}
        if use_gradnorm:
            train_bd.update({"w_gn": 0, "w_patch": 0, "w_proj": 0})
        n = 0
        nan_batches = 0

        for batch in loaders.train:
            xyz, patch_id, xi, gn, normal, dndxs = [t.to(device, non_blocking=True) for t in batch]
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                outputs = model(xyz)
                loss_gn, loss_patch, loss_proj = compute_individual_losses(outputs, gn, patch_id, xi)

            # NaN guard: skip batch if any loss is NaN
            if not (torch.isfinite(loss_gn) and torch.isfinite(loss_patch) and torch.isfinite(loss_proj)):
                nan_batches += 1
                continue

            # Compute weighted total
            if use_gradnorm:
                weights = torch.softmax(log_weights, dim=0) * 3.0
                total = weights[0] * loss_gn + weights[1] * loss_patch + weights[2] * loss_proj
            else:
                lam_gn = getattr(model_cfg, "lambda_gn", 1.0)
                lam_patch = getattr(model_cfg, "lambda_patch", 1.0)
                lam_proj = getattr(model_cfg, "lambda_proj", 1.0)
                total = lam_gn * loss_gn + lam_patch * loss_patch + lam_proj * loss_proj

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # GradNorm weight update (v3 only)
            if use_gradnorm:
                with torch.no_grad():
                    losses_det = torch.tensor(
                        [loss_gn.item(), loss_patch.item(), loss_proj.item()], device=device
                    )
                    if torch.isfinite(losses_det).all():
                        if initial_losses is None:
                            initial_losses = losses_det.clone()
                        loss_ratios = losses_det / initial_losses.clamp(min=1e-8)
                        mean_ratio = loss_ratios.mean()
                        target_w = (loss_ratios / mean_ratio.clamp(min=1e-8)).pow(gradnorm_alpha)
                        target_w = target_w / target_w.sum() * 3.0
                        current_w = torch.softmax(log_weights, dim=0) * 3.0
                        log_weights.data -= gradnorm_lr * (current_w - target_w)
                        log_weights.data.clamp_(-5.0, 5.0)

            # Accumulate metrics
            train_total += total.item()
            train_bd["loss_gn"] += loss_gn.item()
            train_bd["loss_patch"] += loss_patch.item()
            train_bd["loss_proj"] += loss_proj.item()
            if use_gradnorm:
                train_bd["w_gn"] += weights[0].item()
                train_bd["w_patch"] += weights[1].item()
                train_bd["w_proj"] += weights[2].item()
            n += 1

        if n == 0:
            print(f"[{epoch:3d}/{epochs}] ALL BATCHES NaN — skipping epoch", flush=True)
            continue

        train_total /= n
        train_bd = {k: v / n for k, v in train_bd.items()}

        # ── Validate (unweighted sum for fair comparison) ──
        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for batch in loaders.val:
                xyz, patch_id, xi, gn, normal, dndxs = [t.to(device, non_blocking=True) for t in batch]
                outputs = model(xyz)
                l_gn, l_patch, l_proj = compute_individual_losses(outputs, gn, patch_id, xi)
                v = (l_gn + l_patch + l_proj).item()
                if math.isfinite(v):
                    val_loss += v
                    nv += 1
        val_loss = val_loss / nv if nv > 0 else float("inf")

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # Checkpoint
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            patience_counter = 0
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val,
                "config": model_cfg,
            }
            if use_gradnorm:
                state["log_weights"] = log_weights.data.clone()
            torch.save(state, ckpt_dir / "best_model.pt")
            torch.save(state, ckpt_dir / "checkpoint.pt")
        else:
            patience_counter += 1

        dt = time.time() - t0
        star = " *" if is_best else ""
        nan_info = f" nan_skip={nan_batches}" if nan_batches > 0 else ""

        if epoch % 10 == 0 or is_best or epoch < start_epoch + 5:
            w_info = ""
            if use_gradnorm:
                w_info = f" w=[{train_bd['w_gn']:.2f},{train_bd['w_patch']:.2f},{train_bd['w_proj']:.2f}]"
            print(
                f"[{epoch:3d}/{epochs}] "
                f"train={train_total:.4e} (gn={train_bd['loss_gn']:.3e} cls={train_bd['loss_patch']:.3e} proj={train_bd['loss_proj']:.3e}) "
                f"val={val_loss:.4e} lr={lr:.2e}{w_info} "
                f"dt={dt:.0f}s{star}{nan_info}",
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
    print(f"\nDone ({variant}). Best val loss: {best_val:.6f}", flush=True)
    if use_gradnorm:
        w = torch.softmax(log_weights, 0) * 3
        print(f"Final weights: gn={w[0]:.2f}, patch={w[1]:.2f}, proj={w[2]:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train multi-task contact NN")
    parser.add_argument("--variant", default="v2", choices=["v1", "v2", "v2b", "v3"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

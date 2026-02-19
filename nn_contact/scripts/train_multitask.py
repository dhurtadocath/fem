"""Unified training script for multi-task contact NN.

Variants:
  v1  — Baseline: ReLU + trunk [512,512,256,256], no Fourier, equal weights
  v2  — Fourier encoding (128 freq) + SiLU + trunk [512,512,512,256], equal weights
  v2b — Bigger heads + more frequencies (256) + hand-tuned loss weights
  v3  — v2 architecture + GradNorm adaptive loss balancing

Extended features (sweep-ready):
  --focal_gamma     — Focal loss for patch classification (0 = standard CE)
  --label_smoothing — Label smoothing epsilon for patch classification
  --ohem_ratio      — OHEM: keep only top-K% hardest samples per batch
  --task_attention   — MTAN-style task attention gates
  --patch_conditioned — FiLM-conditioned regression (replaces segmented)
  --manifold_mixup  — Manifold Mixup at trunk output level
  --mixup_alpha     — Mixup Beta distribution parameter
  --uncertainty_wt  — Kendall-Gal uncertainty weighting (learns log-variance per task)
  --fourier         — Enable Fourier features for v1 architecture
  --fourier_freq    — Number of Fourier frequencies
  --fourier_scale   — Fourier frequency scale (sigma)
  --lambda_patch    — Loss weight for patch classification
  --lambda_proj     — Loss weight for projection regression

Features: AMP mixed precision (GPU), CPU thread tuning, warmup+cosine LR,
          NaN guards, resume support.

Usage:
    # GPU training
    python3 nn_contact/scripts/train_multitask.py --variant v1 --focal_gamma 2 --label_smoothing 0.05

    # CPU/HPC training
    python3 nn_contact/scripts/train_multitask.py --variant v1 --device cpu --num_threads 16

    # Full sweep via sweep_multitask.py
    python3 nn_contact/scripts/sweep_multitask.py --index 0
"""

from __future__ import annotations

import argparse
import math
import os
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

from nn_contact.config import DataConfig, FourierMLPConfig, MLPConfig, MultiTaskConfig
from nn_contact.data.loader import make_dataloaders
from nn_contact.models.multitask import MultiTaskContactNet
from nn_contact.training.losses import (
    focal_loss,
    segmented_regression_loss,
)


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

def compute_individual_losses(outputs, gn, patch_id, xi,
                              focal_gamma=0.0, label_smoothing=0.0,
                              patch_conditioned=False):
    """Return 3 individual task losses. Clamps logits to prevent AMP overflow."""
    loss_gn = F.mse_loss(outputs["gn_pred"].squeeze(-1), gn)
    logits = outputs["patch_logits"].float().clamp(-30, 30)

    if focal_gamma > 0:
        loss_patch = focal_loss(logits, patch_id, gamma=focal_gamma,
                                label_smoothing=label_smoothing)
    else:
        loss_patch = F.cross_entropy(logits, patch_id,
                                     label_smoothing=label_smoothing)

    # Patch-conditioned: use xi_cond if available (direct 2-output from FiLM head)
    if patch_conditioned and "xi_cond" in outputs:
        loss_proj = F.mse_loss(outputs["xi_cond"], xi)
    else:
        loss_proj = segmented_regression_loss(outputs["xi_pred"], xi, patch_id)

    return loss_gn, loss_patch, loss_proj


# ── Manifold Mixup utility ───────────────────────────────────────────────────

def manifold_mixup(features, targets_tuple, alpha=0.4):
    """Mix features and targets at the trunk output level.

    Parameters
    ----------
    features : (B, D) — trunk features
    targets_tuple : (gn, patch_id, xi) — target tensors
    alpha : float — Beta distribution parameter

    Returns
    -------
    mixed_features, (mixed_gn, patch_id_a, patch_id_b, xi_a, xi_b, lam)
    """
    B = features.shape[0]
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 (keep dominant sample identity)

    idx = torch.randperm(B, device=features.device)

    gn, patch_id, xi = targets_tuple
    mixed_features = lam * features + (1 - lam) * features[idx]
    mixed_gn = lam * gn + (1 - lam) * gn[idx]

    return mixed_features, (mixed_gn, patch_id, patch_id[idx], xi, xi[idx], lam)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = args.device
    variant = args.variant
    epochs = args.epochs
    use_gradnorm = (variant == "v3") or args.uncertainty_wt
    is_cpu = (device == "cpu")

    # ── CPU/HPC thread setup ──
    if is_cpu:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
        if args.num_threads > 0:
            n_threads = args.num_threads
        else:
            n_threads = min(16, slurm_cpus)
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(min(4, n_threads))
        print(f"CPU threads: {n_threads} (SLURM_CPUS={slurm_cpus})", flush=True)

    torch.manual_seed(42)
    if not is_cpu:
        torch.backends.cudnn.benchmark = True

    # Config
    model_cfg = VARIANT_CONFIGS[variant]()

    # Override loss weights from CLI
    if args.lambda_patch != 1.0:
        model_cfg.lambda_patch = args.lambda_patch
    if args.lambda_proj != 1.0:
        model_cfg.lambda_proj = args.lambda_proj

    # Override Fourier features from CLI for v1
    if args.fourier and model_cfg.input_encoding == "none":
        model_cfg.input_encoding = "fourier"
        model_cfg.fourier_config = FourierMLPConfig(
            n_frequencies=args.fourier_freq,
            frequency_scale=args.fourier_scale,
        )

    ckpt_tag = args.ckpt_tag or f"multitask_{variant}"
    ckpt_dir = Path(f"nn_contact/checkpoints/{ckpt_tag}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data
    n_workers = min(4, max(1, (os.cpu_count() or 4) - 2)) if is_cpu else 4
    data_cfg = DataConfig(
        batch_size=args.batch_size,
        num_workers=n_workers,
        pin_memory=not is_cpu,
    )
    loaders = make_dataloaders(data_cfg, seed=42, verbose=True)

    # Model
    model = MultiTaskContactNet.from_config(
        model_cfg,
        task_attention=args.task_attention,
        patch_conditioned=args.patch_conditioned,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Uncertainty weighting: learn log-variance per task (Kendall-Gal)
    log_vars = None
    if args.uncertainty_wt:
        log_vars = nn.Parameter(torch.zeros(3, device=device))

    # GradNorm state (v3 only)
    log_weights = nn.Parameter(torch.zeros(3, device=device)) if (variant == "v3") else None
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
            if log_weights is not None and "log_weights" in ckpt:
                log_weights.data.copy_(ckpt["log_weights"].to(device))
            print(f"Resumed from epoch {start_epoch}, best val={best_val:.6f}", flush=True)

    # Print config
    features_str = f"variant={variant}"
    if args.focal_gamma > 0:
        features_str += f" focal={args.focal_gamma}"
    if args.label_smoothing > 0:
        features_str += f" ls={args.label_smoothing}"
    if args.ohem_ratio < 1.0:
        features_str += f" ohem={args.ohem_ratio}"
    if args.task_attention:
        features_str += " attn"
    if args.patch_conditioned:
        features_str += " pcond"
    if args.manifold_mixup:
        features_str += f" mixup={args.mixup_alpha}"
    if args.uncertainty_wt:
        features_str += " uncwt"
    if model_cfg.input_encoding == "fourier":
        features_str += f" fourier({args.fourier_freq},{args.fourier_scale})"

    print(f"Model ({features_str}): {n_params:,} params, device={device}", flush=True)

    # Optimizer
    warmup = 10
    param_groups = [{"params": model.parameters(), "lr": args.lr, "weight_decay": 1e-4}]
    if log_weights is not None:
        param_groups.append({"params": [log_weights], "lr": gradnorm_lr, "weight_decay": 0})
    if log_vars is not None:
        param_groups.append({"params": [log_vars], "lr": args.lr, "weight_decay": 0})
    optimizer = torch.optim.AdamW(param_groups)

    def lr_lambda(epoch):
        if epoch < warmup:
            return max(0.01, epoch / warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for _ in range(start_epoch):
        scheduler.step()

    use_amp = not is_cpu
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    patience_counter = 0
    patience = args.patience

    # ── Epoch loop ──
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_total = 0.0
        train_bd = {"loss_gn": 0, "loss_patch": 0, "loss_proj": 0}
        n = 0
        nan_batches = 0

        for batch in loaders.train:
            xyz, patch_id, xi, gn, normal, dndxs = [t.to(device, non_blocking=True) for t in batch]
            optimizer.zero_grad(set_to_none=True)

            amp_ctx = torch.amp.autocast("cuda") if use_amp else torch.amp.autocast("cpu", enabled=False)
            with amp_ctx:
                # Forward pass (pass true patch_id for patch-conditioned head)
                outputs = model(xyz, patch_id_true=patch_id if args.patch_conditioned else None)

                loss_gn, loss_patch, loss_proj = compute_individual_losses(
                    outputs, gn, patch_id, xi,
                    focal_gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
                    patch_conditioned=args.patch_conditioned,
                )

            # NaN guard
            if not (torch.isfinite(loss_gn) and torch.isfinite(loss_patch) and torch.isfinite(loss_proj)):
                nan_batches += 1
                continue

            # OHEM: recompute with hard sample selection
            if args.ohem_ratio < 1.0:
                from nn_contact.training.losses import multitask_loss
                total, _ = multitask_loss(
                    outputs,
                    {"gn": gn, "patch_id": patch_id, "xi": xi},
                    lambda_gn=model_cfg.lambda_gn,
                    lambda_patch=model_cfg.lambda_patch,
                    lambda_proj=model_cfg.lambda_proj,
                    focal_gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
                    ohem_ratio=args.ohem_ratio,
                )
            elif args.uncertainty_wt and log_vars is not None:
                # Kendall-Gal uncertainty weighting: L_i / (2*sigma_i^2) + log(sigma_i)
                precision = torch.exp(-log_vars)  # 1/sigma^2
                total = (precision[0] * loss_gn + 0.5 * log_vars[0]
                         + precision[1] * loss_patch + 0.5 * log_vars[1]
                         + precision[2] * loss_proj + 0.5 * log_vars[2])
            elif log_weights is not None:
                # GradNorm weights
                weights = torch.softmax(log_weights, dim=0) * 3.0
                total = weights[0] * loss_gn + weights[1] * loss_patch + weights[2] * loss_proj
            else:
                total = (model_cfg.lambda_gn * loss_gn
                         + model_cfg.lambda_patch * loss_patch
                         + model_cfg.lambda_proj * loss_proj)

            if use_amp:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # GradNorm weight update (v3 only)
            if log_weights is not None:
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
            n += 1

        if n == 0:
            print(f"[{epoch:3d}/{epochs}] ALL BATCHES NaN — skipping epoch", flush=True)
            continue

        train_total /= n
        train_bd = {k: v / n for k, v in train_bd.items()}

        # ── Validate (unweighted sum for fair comparison) ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total_samples = 0
        nv = 0
        amp_ctx_val = torch.amp.autocast("cuda") if use_amp else torch.amp.autocast("cpu", enabled=False)
        with torch.no_grad(), amp_ctx_val:
            for batch in loaders.val:
                xyz, patch_id, xi, gn, normal, dndxs = [t.to(device, non_blocking=True) for t in batch]
                outputs = model(xyz)
                l_gn, l_patch, l_proj = compute_individual_losses(outputs, gn, patch_id, xi)
                v = (l_gn + l_patch + l_proj).item()
                if math.isfinite(v):
                    val_loss += v
                    nv += 1
                # Track accuracy
                pred_patch = outputs["patch_logits"].argmax(dim=-1)
                val_correct += (pred_patch == patch_id).sum().item()
                val_total_samples += patch_id.shape[0]

        val_loss = val_loss / nv if nv > 0 else float("inf")
        val_acc = val_correct / max(1, val_total_samples) * 100

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
                "features": features_str,
                "task_attention": args.task_attention,
                "patch_conditioned": args.patch_conditioned,
            }
            if log_weights is not None:
                state["log_weights"] = log_weights.data.clone()
            if log_vars is not None:
                state["log_vars"] = log_vars.data.clone()
            torch.save(state, ckpt_dir / "best_model.pt")
        else:
            patience_counter += 1

        dt = time.time() - t0
        star = " *" if is_best else ""
        nan_info = f" nan_skip={nan_batches}" if nan_batches > 0 else ""

        if epoch % 10 == 0 or is_best or epoch < start_epoch + 5:
            extra = ""
            if log_vars is not None:
                w = torch.exp(-log_vars).detach()
                extra = f" uw=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]"
            print(
                f"[{epoch:3d}/{epochs}] "
                f"train={train_total:.4e} (gn={train_bd['loss_gn']:.3e} "
                f"cls={train_bd['loss_patch']:.3e} proj={train_bd['loss_proj']:.3e}) "
                f"val={val_loss:.4e} acc={val_acc:.2f}% lr={lr:.2e}{extra} "
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
    print(f"\nDone ({features_str}). Best val loss: {best_val:.6f}", flush=True)
    if log_vars is not None:
        w = torch.exp(-log_vars).detach()
        print(f"Final uncertainty weights: gn={w[0]:.2f}, patch={w[1]:.2f}, proj={w[2]:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train multi-task contact NN")
    parser.add_argument("--variant", default="v1", choices=["v1", "v2", "v2b", "v3"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt_tag", type=str, default=None, help="Checkpoint subdirectory name")
    parser.add_argument("--num_threads", type=int, default=0, help="CPU threads (0=auto)")

    # Loss improvements
    parser.add_argument("--focal_gamma", type=float, default=0.0, help="Focal loss gamma (0=off)")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing eps")
    parser.add_argument("--ohem_ratio", type=float, default=1.0, help="OHEM: keep top ratio (1=off)")
    parser.add_argument("--lambda_patch", type=float, default=1.0, help="Classification loss weight")
    parser.add_argument("--lambda_proj", type=float, default=1.0, help="Projection loss weight")
    parser.add_argument("--uncertainty_wt", action="store_true", help="Kendall-Gal uncertainty weighting")

    # Architecture improvements
    parser.add_argument("--task_attention", action="store_true", help="MTAN-style task attention")
    parser.add_argument("--patch_conditioned", action="store_true", help="FiLM patch-conditioned regression")
    parser.add_argument("--fourier", action="store_true", help="Enable Fourier features for v1")
    parser.add_argument("--fourier_freq", type=int, default=128, help="Fourier frequencies")
    parser.add_argument("--fourier_scale", type=float, default=10.0, help="Fourier scale sigma")

    # Data augmentation
    parser.add_argument("--manifold_mixup", action="store_true", help="Manifold Mixup at trunk")
    parser.add_argument("--mixup_alpha", type=float, default=0.4, help="Mixup Beta param")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

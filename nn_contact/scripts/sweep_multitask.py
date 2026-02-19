#!/usr/bin/env python3
"""Hyperparameter sweep for multi-task contact NN improvements.

Designed for HPC array jobs:
    # Run single config
    python3 nn_contact/scripts/sweep_multitask.py --index 0

    # SLURM array job (runs all configs in parallel)
    sbatch --array=0-15 sweep_multitask.slurm

    # List all configs
    python3 nn_contact/scripts/sweep_multitask.py --list

Each config is a dict of CLI args passed to train_multitask.py logic.
Results are saved to nn_contact/checkpoints/mt_sweep_<name>/.

Improvement strategies tested:
  1. Focal loss (gamma=2) — focus on hard boundary patches
  2. Label smoothing (eps=0.05, 0.1) — better calibration
  3. OHEM (ratio=0.5, 0.7) — hard example mining
  4. Focal + LS + OHEM combo — all Tier-1 together
  5. Fourier features on v1 — input encoding
  6. Task attention (MTAN) — per-task feature gating
  7. Patch-conditioned regression — FiLM-modulated xi head
  8. Uncertainty weighting (Kendall-Gal) — learned loss balancing
  9. Higher lambda_patch — upweight classification
  10. Higher lambda_proj — upweight regression
  11. Best combo candidates
"""

from __future__ import annotations

import argparse
import math
import os
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

from nn_contact.config import DataConfig, FourierMLPConfig, MLPConfig, MultiTaskConfig
from nn_contact.data.loader import make_dataloaders
from nn_contact.models.multitask import MultiTaskContactNet
from nn_contact.training.losses import (
    focal_loss,
    segmented_regression_loss,
)


# ── Sweep configuration ─────────────────────────────────────────────────────

@dataclass
class SweepConfig:
    """One sweep experiment configuration."""
    name: str
    variant: str = "v1"
    epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 2048
    patience: int = 40

    # Loss improvements
    focal_gamma: float = 0.0
    label_smoothing: float = 0.0
    ohem_ratio: float = 1.0
    lambda_patch: float = 1.0
    lambda_proj: float = 1.0
    uncertainty_wt: bool = False

    # Architecture
    task_attention: bool = False
    patch_conditioned: bool = False
    fourier: bool = False
    fourier_freq: int = 128
    fourier_scale: float = 10.0


# ── Define all sweep configs ─────────────────────────────────────────────────

CONFIGS: list[SweepConfig] = [
    # ── Baselines ──
    SweepConfig(name="baseline_v1"),
    SweepConfig(name="baseline_v2", variant="v2"),

    # ── Tier 1: Easy wins ──
    SweepConfig(name="focal_g2", focal_gamma=2.0),
    SweepConfig(name="focal_g1", focal_gamma=1.0),
    SweepConfig(name="ls_005", label_smoothing=0.05),
    SweepConfig(name="ls_010", label_smoothing=0.10),
    SweepConfig(name="ohem_50", ohem_ratio=0.5),
    SweepConfig(name="ohem_70", ohem_ratio=0.7),

    # ── Tier 1 combos ──
    SweepConfig(name="focal_ls", focal_gamma=2.0, label_smoothing=0.05),
    SweepConfig(name="focal_ls_ohem", focal_gamma=2.0, label_smoothing=0.05, ohem_ratio=0.5),
    SweepConfig(name="focal_ohem70", focal_gamma=2.0, ohem_ratio=0.7),

    # ── Tier 2: Architecture ──
    SweepConfig(name="fourier_v1", fourier=True, fourier_freq=128, fourier_scale=10.0),
    SweepConfig(name="fourier_hi", fourier=True, fourier_freq=256, fourier_scale=15.0),
    SweepConfig(name="task_attn", task_attention=True),
    SweepConfig(name="patch_cond", patch_conditioned=True),
    SweepConfig(name="unc_wt", uncertainty_wt=True),

    # ── Tier 2: Loss weights ──
    SweepConfig(name="lam_patch2", lambda_patch=2.0),
    SweepConfig(name="lam_proj2", lambda_proj=2.0),
    SweepConfig(name="lam_patch2_proj2", lambda_patch=2.0, lambda_proj=2.0),

    # ── Tier 2 combos with best Tier 1 ──
    SweepConfig(name="focal_ls_attn", focal_gamma=2.0, label_smoothing=0.05, task_attention=True),
    SweepConfig(name="focal_ls_fourier", focal_gamma=2.0, label_smoothing=0.05,
                fourier=True, fourier_freq=128, fourier_scale=10.0),
    SweepConfig(name="focal_ls_pcond", focal_gamma=2.0, label_smoothing=0.05,
                patch_conditioned=True),
    SweepConfig(name="full_combo", focal_gamma=2.0, label_smoothing=0.05, ohem_ratio=0.7,
                task_attention=True, fourier=True),
]


# ── Training logic (mirrors train_multitask.py) ─────────────────────────────

def compute_individual_losses(outputs, gn, patch_id, xi,
                              focal_gamma=0.0, label_smoothing=0.0,
                              patch_conditioned=False):
    """Return 3 individual task losses."""
    loss_gn = F.mse_loss(outputs["gn_pred"].squeeze(-1), gn)
    logits = outputs["patch_logits"].float().clamp(-30, 30)

    if focal_gamma > 0:
        loss_patch = focal_loss(logits, patch_id, gamma=focal_gamma,
                                label_smoothing=label_smoothing)
    else:
        loss_patch = F.cross_entropy(logits, patch_id,
                                     label_smoothing=label_smoothing)

    if patch_conditioned and "xi_cond" in outputs:
        loss_proj = F.mse_loss(outputs["xi_cond"], xi)
    else:
        loss_proj = segmented_regression_loss(outputs["xi_pred"], xi, patch_id)

    return loss_gn, loss_patch, loss_proj


def train_one(cfg: SweepConfig, device: str = "cpu", num_threads: int = 0):
    """Train a single configuration and return final metrics."""
    is_cpu = (device == "cpu")
    t_start = time.time()

    # CPU thread setup
    if is_cpu:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
        n_threads = num_threads if num_threads > 0 else min(16, slurm_cpus)
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(min(4, n_threads))

    torch.manual_seed(42)
    if not is_cpu:
        torch.backends.cudnn.benchmark = True

    # ── Build model config ──
    if cfg.variant == "v1":
        model_cfg = MultiTaskConfig()
    elif cfg.variant == "v2":
        model_cfg = MultiTaskConfig(
            trunk=MLPConfig(hidden_dims=[512, 512, 512, 256], activation="silu",
                            skip_connections=True),
            gn_head_dims=[128, 64],
            patch_head_dims=[256, 128],
            proj_head_dims=[256, 128],
            input_encoding="fourier",
            fourier_config=FourierMLPConfig(n_frequencies=128, frequency_scale=10.0),
        )
    else:
        model_cfg = MultiTaskConfig()

    model_cfg.lambda_patch = cfg.lambda_patch
    model_cfg.lambda_proj = cfg.lambda_proj

    if cfg.fourier and model_cfg.input_encoding == "none":
        model_cfg.input_encoding = "fourier"
        model_cfg.fourier_config = FourierMLPConfig(
            n_frequencies=cfg.fourier_freq,
            frequency_scale=cfg.fourier_scale,
        )

    ckpt_dir = Path(f"nn_contact/checkpoints/mt_sweep_{cfg.name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    n_workers = min(4, max(1, (os.cpu_count() or 4) - 2)) if is_cpu else 4
    data_cfg = DataConfig(
        batch_size=cfg.batch_size,
        num_workers=n_workers,
        pin_memory=not is_cpu,
    )
    loaders = make_dataloaders(data_cfg, seed=42, verbose=True)

    # ── Model ──
    model = MultiTaskContactNet.from_config(
        model_cfg,
        task_attention=cfg.task_attention,
        patch_conditioned=cfg.patch_conditioned,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Uncertainty weighting
    log_vars = nn.Parameter(torch.zeros(3, device=device)) if cfg.uncertainty_wt else None

    # Print config
    features = []
    if cfg.focal_gamma > 0:
        features.append(f"focal={cfg.focal_gamma}")
    if cfg.label_smoothing > 0:
        features.append(f"ls={cfg.label_smoothing}")
    if cfg.ohem_ratio < 1.0:
        features.append(f"ohem={cfg.ohem_ratio}")
    if cfg.task_attention:
        features.append("attn")
    if cfg.patch_conditioned:
        features.append("pcond")
    if cfg.fourier or model_cfg.input_encoding == "fourier":
        features.append(f"fourier({cfg.fourier_freq},{cfg.fourier_scale})")
    if cfg.uncertainty_wt:
        features.append("uncwt")
    if cfg.lambda_patch != 1.0:
        features.append(f"lp={cfg.lambda_patch}")
    if cfg.lambda_proj != 1.0:
        features.append(f"lr={cfg.lambda_proj}")
    feat_str = " ".join(features) if features else "baseline"

    print(f"\n{'='*70}", flush=True)
    print(f"Config: {cfg.name} [{feat_str}]", flush=True)
    print(f"Model: {n_params:,} params, variant={cfg.variant}, device={device}", flush=True)
    print(f"{'='*70}", flush=True)

    # ── Optimizer ──
    param_groups = [{"params": model.parameters(), "lr": cfg.lr, "weight_decay": 1e-4}]
    if log_vars is not None:
        param_groups.append({"params": [log_vars], "lr": cfg.lr, "weight_decay": 0})
    optimizer = torch.optim.AdamW(param_groups)

    warmup = 10

    def lr_lambda(epoch):
        if epoch < warmup:
            return max(0.01, epoch / warmup)
        progress = (epoch - warmup) / max(1, cfg.epochs - warmup)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = not is_cpu
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val = float("inf")
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    # ── Epoch loop ──
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        train_total = 0.0
        train_n = 0

        for batch in loaders.train:
            xyz, patch_id, xi, gn, normal, dndxs = [t.to(device, non_blocking=True) for t in batch]
            optimizer.zero_grad(set_to_none=True)

            amp_ctx = torch.amp.autocast("cuda") if use_amp else torch.amp.autocast("cpu", enabled=False)
            with amp_ctx:
                outputs = model(xyz, patch_id_true=patch_id if cfg.patch_conditioned else None)
                loss_gn, loss_patch, loss_proj = compute_individual_losses(
                    outputs, gn, patch_id, xi,
                    focal_gamma=cfg.focal_gamma,
                    label_smoothing=cfg.label_smoothing,
                    patch_conditioned=cfg.patch_conditioned,
                )

            if not (torch.isfinite(loss_gn) and torch.isfinite(loss_patch) and torch.isfinite(loss_proj)):
                continue

            # OHEM: recompute on hard subset
            if cfg.ohem_ratio < 1.0:
                B = gn.shape[0]
                k = max(1, int(B * cfg.ohem_ratio))
                with torch.no_grad():
                    per_sample = (outputs["gn_pred"].squeeze(-1) - gn).pow(2)
                    per_cls = F.cross_entropy(outputs["patch_logits"], patch_id, reduction="none")
                    difficulty = per_sample + per_cls
                    _, hard_idx = difficulty.topk(k)

                # Re-forward on hard subset only
                with amp_ctx:
                    out_hard = model(xyz[hard_idx],
                                     patch_id_true=patch_id[hard_idx] if cfg.patch_conditioned else None)
                    loss_gn, loss_patch, loss_proj = compute_individual_losses(
                        out_hard, gn[hard_idx], patch_id[hard_idx], xi[hard_idx],
                        focal_gamma=cfg.focal_gamma,
                        label_smoothing=cfg.label_smoothing,
                        patch_conditioned=cfg.patch_conditioned,
                    )

            if cfg.uncertainty_wt and log_vars is not None:
                precision = torch.exp(-log_vars)
                total = (precision[0] * loss_gn + 0.5 * log_vars[0]
                         + precision[1] * loss_patch + 0.5 * log_vars[1]
                         + precision[2] * loss_proj + 0.5 * log_vars[2])
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

            train_total += total.item()
            train_n += 1

        if train_n == 0:
            continue
        train_total /= train_n

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total_samples = 0
        val_xi_err = 0.0
        val_gn_err = 0.0
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

                # Detailed metrics
                B = patch_id.shape[0]
                pred_patch = outputs["patch_logits"].argmax(dim=-1)
                val_correct += (pred_patch == patch_id).sum().item()
                val_total_samples += B

                # xi error for correctly classified
                correct_mask = (pred_patch == patch_id)
                if correct_mask.any():
                    idx_xi = pred_patch[correct_mask].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
                    xi_sel = outputs["xi_pred"][correct_mask].gather(1, idx_xi).squeeze(1)
                    val_xi_err += (xi_sel - xi[correct_mask]).abs().mean().item() * correct_mask.sum().item()

                # gn RMSE
                val_gn_err += (outputs["gn_pred"].squeeze(-1) - gn).pow(2).sum().item()

        val_loss = val_loss / nv if nv > 0 else float("inf")
        val_acc = val_correct / max(1, val_total_samples) * 100
        val_xi_mean = val_xi_err / max(1, val_correct)
        val_gn_rmse = math.sqrt(val_gn_err / max(1, val_total_samples))

        scheduler.step()
        dt = time.time() - t0

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_val_loss": best_val,
                "config": model_cfg,
                "sweep_config": cfg.name,
            }, ckpt_dir / "best_model.pt")
        else:
            patience_counter += 1

        star = " *" if is_best else ""
        if epoch % 20 == 0 or is_best or epoch < 5:
            print(
                f"  [{epoch:3d}/{cfg.epochs}] train={train_total:.4e} "
                f"val={val_loss:.4e} acc={val_acc:.2f}% "
                f"xi_err={val_xi_mean:.5f} gn_rmse={val_gn_rmse:.5f} "
                f"dt={dt:.0f}s{star}",
                flush=True,
            )

        if patience_counter >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}", flush=True)
            break

    elapsed = time.time() - t_start
    results = {
        "name": cfg.name,
        "best_val_loss": best_val,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "n_params": n_params,
        "elapsed_min": elapsed / 60,
    }
    print(f"\n  RESULT [{cfg.name}]: val_loss={best_val:.6f} acc={best_acc:.2f}% "
          f"epoch={best_epoch} time={elapsed/60:.1f}min", flush=True)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sweep multitask improvements")
    parser.add_argument("--index", type=int, default=-1,
                        help="Config index (0-based). -1 = use SLURM_ARRAY_TASK_ID")
    parser.add_argument("--list", action="store_true", help="List all configs and exit")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_threads", type=int, default=0, help="CPU threads (0=auto)")
    parser.add_argument("--all", action="store_true", help="Run ALL configs sequentially")
    args = parser.parse_args()

    if args.list:
        print(f"{'Idx':>3}  {'Name':<25}  Description")
        print("-" * 70)
        for i, c in enumerate(CONFIGS):
            feats = []
            if c.focal_gamma > 0:
                feats.append(f"focal={c.focal_gamma}")
            if c.label_smoothing > 0:
                feats.append(f"ls={c.label_smoothing}")
            if c.ohem_ratio < 1.0:
                feats.append(f"ohem={c.ohem_ratio}")
            if c.task_attention:
                feats.append("attn")
            if c.patch_conditioned:
                feats.append("pcond")
            if c.fourier:
                feats.append("fourier")
            if c.uncertainty_wt:
                feats.append("uncwt")
            if c.lambda_patch != 1.0:
                feats.append(f"lp={c.lambda_patch}")
            if c.lambda_proj != 1.0:
                feats.append(f"lr={c.lambda_proj}")
            desc = " ".join(feats) if feats else f"baseline ({c.variant})"
            print(f"{i:3d}  {c.name:<25}  {desc}")
        print(f"\nTotal: {len(CONFIGS)} configs")
        return

    if args.all:
        results = []
        for i, cfg in enumerate(CONFIGS):
            print(f"\n{'#'*70}")
            print(f"# Running config {i}/{len(CONFIGS)}: {cfg.name}")
            print(f"{'#'*70}")
            r = train_one(cfg, device=args.device, num_threads=args.num_threads)
            results.append(r)

        # Final summary
        print(f"\n{'='*70}")
        print("SWEEP SUMMARY")
        print(f"{'='*70}")
        print(f"{'Name':<25} {'Val Loss':>10} {'Acc%':>7} {'Epoch':>6}")
        print("-" * 55)
        for r in sorted(results, key=lambda x: x["best_val_loss"]):
            print(f"{r['name']:<25} {r['best_val_loss']:10.6f} {r['best_acc']:6.2f}% {r['best_epoch']:6d}")
        return

    # Single config (array job or explicit index)
    idx = args.index
    if idx < 0:
        idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    if idx >= len(CONFIGS):
        print(f"Index {idx} out of range (max {len(CONFIGS)-1})")
        sys.exit(1)

    cfg = CONFIGS[idx]
    train_one(cfg, device=args.device, num_threads=args.num_threads)


if __name__ == "__main__":
    main()

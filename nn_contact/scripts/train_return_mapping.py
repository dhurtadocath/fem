"""Train the learned return mapping surrogate (Phase 3).

Three-phase training protocol:
  Phase 1: Full dataset, standard LR — learn overall mapping
  Phase 2: Hard example mining — weight by |delta_ep| to focus on yielding GPs
  Phase 3: Fine-tune on plastically active GPs only, low LR

Usage:
    python -m nn_contact.scripts.train_return_mapping --data_dir data/rm_train

    # HPC CPU optimized:
    python -m nn_contact.scripts.train_return_mapping --data_dir data/rm_train \
        --device cpu --batch_size 4096 --num_threads 16
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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nn_contact.config import ReturnMappingConfig
from nn_contact.models.return_mapping import ReturnMappingNet
from nn_contact.training.losses import return_mapping_loss


def load_rm_data(data_dir: str, val_fraction: float = 0.2, seed: int = 42):
    """Load return mapping training data.

    Expected files in data_dir:
      - rm_inputs.npy:  (N, 19) — [F(9), Fp_old(9), epcum(1)]
      - rm_Fp_new.npy:  (N, 9)  — target Fp_new
      - rm_dep.npy:     (N,)    — target delta_epcum
      - rm_yielding.npy (optional): (N,) — 0/1 yielding flag
    """
    data_dir = Path(data_dir)
    inputs = np.load(data_dir / "rm_inputs.npy").astype(np.float32)
    Fp_target = np.load(data_dir / "rm_Fp_new.npy").astype(np.float32)
    dep_target = np.load(data_dir / "rm_dep.npy").astype(np.float32)

    # Optional: yielding flags for phase 2/3 weighting
    yield_path = data_dir / "rm_yielding.npy"
    if yield_path.exists():
        yielding = np.load(yield_path).astype(np.float32)
    else:
        yielding = (dep_target > 1e-10).astype(np.float32)

    N = len(inputs)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    n_val = int(N * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    n_yield_total = int(yielding.sum())
    print(f"  Data: {N:,} total, {n_yield_total:,} yielding "
          f"({100*n_yield_total/N:.1f}%)")

    return (inputs, Fp_target, dep_target, yielding,
            train_idx, val_idx)


def make_datasets(inputs, Fp_target, dep_target, yielding, idx):
    """Create TensorDataset from index array."""
    return TensorDataset(
        torch.from_numpy(inputs[idx]),
        torch.from_numpy(Fp_target[idx]),
        torch.from_numpy(dep_target[idx]),
        torch.from_numpy(yielding[idx]),
    )


@torch.no_grad()
def validate(model, val_loader, device, cfg):
    """Run validation and report physics metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_Fp_err = []
    all_dep_err = []
    all_det = []
    n_elastic_correct = 0
    n_elastic_total = 0

    for batch in val_loader:
        x, Fp_tgt, dep_tgt, yld = [b.to(device) for b in batch]
        out = model(x)

        loss, _ = return_mapping_loss(
            out, Fp_tgt, dep_tgt,
            lambda_Fp=cfg.lambda_Fp, lambda_ep=cfg.lambda_epcum,
            lambda_det=cfg.lambda_det)
        total_loss += loss.item()
        n_batches += 1

        # Per-sample errors
        Fp_err = (out["Fp_new"] - Fp_tgt).pow(2).sum(dim=-1).sqrt()
        dep_err = (out["delta_ep"] - dep_tgt).abs()
        all_Fp_err.append(Fp_err.cpu())
        all_dep_err.append(dep_err.cpu())

        # det(Fp)
        from nn_contact.training.losses import _det3x3
        det = _det3x3(out["Fp_new"])
        all_det.append(det.cpu())

        # Elastic classification: if true dep=0, is predicted dep < threshold?
        elastic_mask = dep_tgt < 1e-10
        if elastic_mask.any():
            pred_elastic = out["delta_ep"][elastic_mask] < 1e-6
            n_elastic_correct += pred_elastic.sum().item()
            n_elastic_total += elastic_mask.sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    Fp_err_all = torch.cat(all_Fp_err)
    dep_err_all = torch.cat(all_dep_err)
    det_all = torch.cat(all_det)

    elastic_acc = n_elastic_correct / max(n_elastic_total, 1)

    return {
        "val_loss": avg_loss,
        "Fp_rmse": Fp_err_all.mean().item(),
        "Fp_max_err": Fp_err_all.max().item(),
        "dep_rmse": dep_err_all.mean().item(),
        "dep_max_err": dep_err_all.max().item(),
        "det_mean": det_all.mean().item(),
        "det_min": det_all.min().item(),
        "det_max": det_all.max().item(),
        "elastic_acc": elastic_acc,
    }


def train_one_epoch(model, loader, optimizer, device, cfg):
    """Train for one epoch, return average loss and breakdown."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x, Fp_tgt, dep_tgt, yld = [b.to(device) for b in batch]
        out = model(x)

        loss, breakdown = return_mapping_loss(
            out, Fp_tgt, dep_tgt,
            lambda_Fp=cfg.lambda_Fp, lambda_ep=cfg.lambda_epcum,
            lambda_det=cfg.lambda_det)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train return mapping surrogate")
    parser.add_argument("--data_dir", required=True,
                        help="Directory with rm_*.npy files")
    # Phase 1 settings
    parser.add_argument("--epochs_p1", type=int, default=200,
                        help="Phase 1 epochs (full dataset)")
    parser.add_argument("--epochs_p2", type=int, default=100,
                        help="Phase 2 epochs (hard example mining)")
    parser.add_argument("--epochs_p3", type=int, default=50,
                        help="Phase 3 epochs (plastic-only fine-tune)")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_p3", type=float, default=1e-5,
                        help="Phase 3 learning rate")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience per phase")
    # Architecture
    parser.add_argument("--hidden_dims", type=int, nargs="+",
                        default=[256, 256, 256, 256])
    parser.add_argument("--activation", default="silu",
                        choices=["relu", "silu", "gelu"])
    # Loss weights
    parser.add_argument("--lambda_det", type=float, default=10.0)
    parser.add_argument("--lambda_iso", type=float, default=5.0)
    parser.add_argument("--lambda_elastic", type=float, default=5.0)
    parser.add_argument("--lambda_inc", type=float, default=1.0)
    # Infra
    parser.add_argument("--checkpoint_dir",
                        default="nn_contact/checkpoints/return_mapping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_threads", type=int, default=0,
                        help="CPU thread count (0 = auto)")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_tag", default="",
                        help="Tag appended to checkpoint filename")
    args = parser.parse_args()

    # CPU thread tuning
    if args.device == "cpu" and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(max(1, args.num_threads // 4))
        print(f"CPU threads: {args.num_threads} compute, "
              f"{max(1, args.num_threads // 4)} interop")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ── Load data ──
    inputs, Fp_target, dep_target, yielding, train_idx, val_idx = \
        load_rm_data(args.data_dir, seed=args.seed)

    val_ds = make_datasets(inputs, Fp_target, dep_target, yielding, val_idx)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)

    # ── Model ──
    cfg = ReturnMappingConfig(
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        lambda_det=args.lambda_det,
    )
    model = ReturnMappingNet.from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters, device={device}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.ckpt_tag}" if args.ckpt_tag else ""
    best_path = ckpt_dir / f"best{tag}.pt"
    best_val_loss = float("inf")

    def save_best(val_metrics, phase):
        nonlocal best_val_loss
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "val_metrics": val_metrics,
                "phase": phase,
            }, best_path)
            return True
        return False

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1: Full dataset training
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Phase 1: Full dataset ({args.epochs_p1} epochs, lr={args.lr})")
    print(f"{'='*60}")

    train_ds_p1 = make_datasets(inputs, Fp_target, dep_target, yielding, train_idx)
    train_loader_p1 = DataLoader(train_ds_p1, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6)

    patience_counter = 0
    for epoch in range(args.epochs_p1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader_p1, optimizer, device, cfg)
        val_metrics = validate(model, val_loader, device, cfg)
        scheduler.step(val_metrics["val_loss"])
        dt = time.time() - t0

        improved = save_best(val_metrics, phase=1)
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % args.log_every == 0 or improved:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  P1 ep {epoch+1:4d}  train={train_loss:.4e}  "
                  f"val={val_metrics['val_loss']:.4e}  "
                  f"Fp_rmse={val_metrics['Fp_rmse']:.2e}  "
                  f"dep_rmse={val_metrics['dep_rmse']:.2e}  "
                  f"det={val_metrics['det_mean']:.4f} "
                  f"[{val_metrics['det_min']:.3f},{val_metrics['det_max']:.3f}]  "
                  f"el_acc={val_metrics['elastic_acc']:.4f}  "
                  f"lr={lr_now:.1e}  {dt:.1f}s"
                  f"{'  *' if improved else ''}")

        if patience_counter >= args.patience:
            print(f"  Phase 1 early stop at epoch {epoch+1}")
            break

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2: Hard example mining — weight by |delta_ep|
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Phase 2: Hard example mining ({args.epochs_p2} epochs)")
    print(f"{'='*60}")

    # Load best model from phase 1
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Build weighted sampler: weight = 1 + 100 * |delta_ep|
    # This upsamples yielding GPs proportionally to plastic strain magnitude
    train_dep = dep_target[train_idx]
    weights = 1.0 + 100.0 * np.abs(train_dep)
    weights = weights / weights.sum()
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights).double(),
        num_samples=len(train_idx),
        replacement=True)
    train_loader_p2 = DataLoader(train_ds_p1, batch_size=args.batch_size,
                                  sampler=sampler, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-6)

    patience_counter = 0
    for epoch in range(args.epochs_p2):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader_p2, optimizer, device, cfg)
        val_metrics = validate(model, val_loader, device, cfg)
        scheduler.step(val_metrics["val_loss"])
        dt = time.time() - t0

        improved = save_best(val_metrics, phase=2)
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % args.log_every == 0 or improved:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  P2 ep {epoch+1:4d}  train={train_loss:.4e}  "
                  f"val={val_metrics['val_loss']:.4e}  "
                  f"Fp_rmse={val_metrics['Fp_rmse']:.2e}  "
                  f"dep_rmse={val_metrics['dep_rmse']:.2e}  "
                  f"det={val_metrics['det_mean']:.4f}  "
                  f"el_acc={val_metrics['elastic_acc']:.4f}  "
                  f"lr={lr_now:.1e}  {dt:.1f}s"
                  f"{'  *' if improved else ''}")

        if patience_counter >= args.patience:
            print(f"  Phase 2 early stop at epoch {epoch+1}")
            break

    # ════════════════════════════════════════════════════════════════════
    # PHASE 3: Fine-tune on plastic GPs only
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Phase 3: Plastic-only fine-tune ({args.epochs_p3} epochs, lr={args.lr_p3})")
    print(f"{'='*60}")

    # Load best model from phase 2
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Filter to plastic GPs only
    plastic_mask = yielding[train_idx] > 0.5
    plastic_train_idx = train_idx[plastic_mask]
    n_plastic = len(plastic_train_idx)
    print(f"  Plastic training samples: {n_plastic:,} / {len(train_idx):,}")

    if n_plastic > 0:
        train_ds_p3 = make_datasets(
            inputs, Fp_target, dep_target, yielding, plastic_train_idx)
        train_loader_p3 = DataLoader(
            train_ds_p3, batch_size=min(args.batch_size, n_plastic),
            shuffle=True, drop_last=n_plastic > args.batch_size)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr_p3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_p3, eta_min=1e-7)

        patience_counter = 0
        for epoch in range(args.epochs_p3):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader_p3, optimizer, device, cfg)
            val_metrics = validate(model, val_loader, device, cfg)
            scheduler.step()
            dt = time.time() - t0

            improved = save_best(val_metrics, phase=3)
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % args.log_every == 0 or improved:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  P3 ep {epoch+1:4d}  train={train_loss:.4e}  "
                      f"val={val_metrics['val_loss']:.4e}  "
                      f"Fp_rmse={val_metrics['Fp_rmse']:.2e}  "
                      f"dep_rmse={val_metrics['dep_rmse']:.2e}  "
                      f"det={val_metrics['det_mean']:.4f}  "
                      f"el_acc={val_metrics['elastic_acc']:.4f}  "
                      f"lr={lr_now:.1e}  {dt:.1f}s"
                      f"{'  *' if improved else ''}")

            if patience_counter >= args.patience:
                print(f"  Phase 3 early stop at epoch {epoch+1}")
                break
    else:
        print("  No plastic samples — skipping phase 3")

    # ── Final summary ──
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final_metrics = validate(model, val_loader, device, cfg)

    print(f"\n{'='*60}")
    print(f"  Training complete — best model from phase {ckpt['phase']}")
    print(f"  val_loss:     {final_metrics['val_loss']:.4e}")
    print(f"  Fp RMSE:      {final_metrics['Fp_rmse']:.2e}  (max {final_metrics['Fp_max_err']:.2e})")
    print(f"  dep RMSE:     {final_metrics['dep_rmse']:.2e}  (max {final_metrics['dep_max_err']:.2e})")
    print(f"  det(Fp):      mean={final_metrics['det_mean']:.4f}  "
          f"[{final_metrics['det_min']:.4f}, {final_metrics['det_max']:.4f}]")
    print(f"  Elastic acc:  {final_metrics['elastic_acc']:.4f}")
    print(f"  Checkpoint:   {best_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

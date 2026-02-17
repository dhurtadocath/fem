"""Evaluate a trained multi-task model on the validation set.

Usage:
    python -m nn_contact.scripts.evaluate_multitask --checkpoint nn_contact/checkpoints/multitask_v1
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/train_multitask.py --variant v1
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/train_multitask.py --variant v3 --resume
    PYTHONUNBUFFERED=1 python3 nn_contact/scripts/train_multitask.py --variant v2b
Produces:
  - Per-task accuracy/error metrics (matching paper Tables/Figures)
  - Error distributions (histograms)
  - Confusion matrix for patch classification
  - Patch-wise error analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nn_contact.config import DataConfig, MultiTaskConfig
from nn_contact.data.loader import make_dataloaders
from nn_contact.data.normalization import CoordinateNormalizer
from nn_contact.models.multitask import MultiTaskContactNet
from nn_contact.evaluation.metrics import compute_multitask_metrics


def evaluate(checkpoint_dir: str, device: str = "cuda"):
    ckpt_dir = Path(checkpoint_dir)

    # Load config
    config_data = torch.load(ckpt_dir / "config.pt", map_location="cpu", weights_only=False)
    model_cfg = config_data.get("config", MultiTaskConfig())
    data_cfg = config_data.get("data_config", DataConfig())

    # Load model
    best_ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
    model = MultiTaskContactNet.from_config(model_cfg).to(device)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()
    print(f"Loaded model from epoch {best_ckpt['epoch']}, val loss {best_ckpt['best_val_loss']:.5e}")

    # Load val data
    data_cfg.num_workers = 0
    loaders = make_dataloaders(data_cfg, seed=42, verbose=False)

    # Collect predictions on entire validation set
    all_gn_pred, all_gn_true = [], []
    all_patch_pred, all_patch_true = [], []
    all_xi_pred, all_xi_true = [], []
    all_patch_probs = []

    with torch.no_grad():
        for batch in loaders.val:
            xyz, patch_id, xi, gn, normal, dndxs = [t.to(device) for t in batch]
            out = model.predict(xyz, topk=3)

            all_gn_pred.append(out["gn"].cpu().numpy())
            all_gn_true.append(gn.cpu().numpy())
            all_patch_pred.append(out["patch_ids"][:, 0].cpu().numpy())  # top-1
            all_patch_true.append(patch_id.cpu().numpy())

            # For xi: use the top-1 predicted patch's xi
            all_xi_pred.append(out["xi"][:, 0, :].cpu().numpy())
            all_xi_true.append(xi.cpu().numpy())

            # Full softmax for top-3 analysis
            full_out = model(xyz)
            probs = torch.softmax(full_out["patch_logits"], dim=-1)
            all_patch_probs.append(probs.cpu().numpy())

    gn_pred = np.concatenate(all_gn_pred)
    gn_true = np.concatenate(all_gn_true)
    patch_pred = np.concatenate(all_patch_pred)
    patch_true = np.concatenate(all_patch_true)
    xi_pred = np.concatenate(all_xi_pred)
    xi_true = np.concatenate(all_xi_true)
    patch_probs = np.concatenate(all_patch_probs)

    print(f"\nEvaluation on {len(gn_pred):,} validation samples")

    # Compute metrics
    metrics = compute_multitask_metrics(
        patch_pred, patch_true, xi_pred, xi_true, gn_pred, gn_true,
        patch_probs=patch_probs,
    )

    print(f"\n{'='*60}")
    print(f"PATCH CLASSIFICATION")
    print(f"{'='*60}")
    print(f"  Top-1 accuracy:    {metrics.patch_accuracy*100:.2f}%  (paper target: >98.7%)")
    print(f"  Top-3 accuracy:    {metrics.patch_top3_accuracy*100:.2f}%")

    print(f"\n{'='*60}")
    print(f"PARAMETRIC PROJECTION")
    print(f"{'='*60}")
    print(f"  Mean ||xi_err||:   {metrics.xi_mean_error:.5f}  (paper target: <0.025)")
    print(f"  Median:            {metrics.xi_p50_error:.5f}")
    print(f"  95th percentile:   {metrics.xi_p95_error:.5f}")
    print(f"  99th percentile:   {metrics.xi_p99_error:.5f}")

    print(f"\n{'='*60}")
    print(f"SIGNED DISTANCE")
    print(f"{'='*60}")
    print(f"  RMSE:              {metrics.gn_rmse:.5f}")
    print(f"  Mean |error|:      {metrics.gn_mean_error:.5f}")
    print(f"  95th percentile:   {metrics.gn_p95_error:.5f}")

    print(f"\n{'='*60}")
    print(f"HYBRID CDA")
    print(f"{'='*60}")
    print(f"  Failure rate:      {metrics.cda_failure_rate*100:.3f}%  (paper target: <0.3%)")

    # Per-patch accuracy breakdown
    print(f"\n{'='*60}")
    print(f"PER-PATCH ACCURACY (worst 10)")
    print(f"{'='*60}")
    for pid in range(96):
        mask = patch_true == pid
        if mask.sum() == 0:
            continue
        acc = (patch_pred[mask] == pid).mean()
        if acc < 1.0:
            n = mask.sum()
            print(f"  Patch {pid:2d}: {acc*100:5.1f}%  (n={n:,})")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-task model")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.device)


if __name__ == "__main__":
    main()

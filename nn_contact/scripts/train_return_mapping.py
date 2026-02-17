"""Train the learned return mapping surrogate (Phase 3).

Usage:
    python -m nn_contact.scripts.train_return_mapping --data_dir <path>

Training data must be generated first by running ContactPotato_NGSolve.py
with plastic=True and data_logging=True (generates per-GP return mapping
input/output pairs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nn_contact.config import ReturnMappingConfig, TrainingConfig
from nn_contact.models.return_mapping import ReturnMappingNet
from nn_contact.training.losses import return_mapping_loss
from nn_contact.training.trainer import Trainer


def load_rm_data(data_dir: str, val_fraction: float = 0.2, seed: int = 42):
    """Load return mapping training data.

    Expected files in data_dir:
      - rm_inputs.npy:  (N, 19) — [F(9), Fp_old(9), epcum(1)]
      - rm_Fp_new.npy:  (N, 9)  — target Fp_new
      - rm_dep.npy:     (N,)    — target delta_epcum
    """
    data_dir = Path(data_dir)
    inputs = np.load(data_dir / "rm_inputs.npy").astype(np.float32)
    Fp_target = np.load(data_dir / "rm_Fp_new.npy").astype(np.float32)
    dep_target = np.load(data_dir / "rm_dep.npy").astype(np.float32)

    N = len(inputs)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    n_val = int(N * val_fraction)

    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def make_ds(idx):
        return TensorDataset(
            torch.from_numpy(inputs[idx]),
            torch.from_numpy(Fp_target[idx]),
            torch.from_numpy(dep_target[idx]),
        )

    return make_ds(train_idx), make_ds(val_idx), N - n_val, n_val


def make_loss_fn(cfg: ReturnMappingConfig):
    def loss_fn(model, batch):
        x, Fp_target, dep_target = batch
        outputs = model(x)
        return return_mapping_loss(
            outputs, Fp_target, dep_target,
            lambda_Fp=cfg.lambda_Fp,
            lambda_ep=cfg.lambda_epcum,
            lambda_det=cfg.lambda_det,
        )
    return loss_fn


def main():
    parser = argparse.ArgumentParser(description="Train return mapping surrogate")
    parser.add_argument("--data_dir", required=True, help="Directory with rm_*.npy files")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--checkpoint_dir", default="nn_contact/checkpoints/return_mapping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_cfg = ReturnMappingConfig()
    train_cfg = TrainingConfig(
        optimizer="adamw",
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

    torch.manual_seed(train_cfg.seed)

    train_ds, val_ds, n_train, n_val = load_rm_data(args.data_dir, seed=train_cfg.seed)
    print(f"Return mapping data: {n_train:,} train, {n_val:,} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)

    model = ReturnMappingNet.from_config(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    trainer = Trainer(model, make_loss_fn(model_cfg), train_cfg, device=args.device)
    trainer.fit(train_loader, val_loader)

    ckpt_dir = Path(args.checkpoint_dir)
    torch.save({"config": model_cfg}, ckpt_dir / "config.pt")
    print(f"\nBest validation loss: {trainer.best_val_loss:.5e}")


if __name__ == "__main__":
    main()

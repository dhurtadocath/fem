"""Train the multi-task contact detection network (Phase 1).

Usage:
    python -m nn_contact.scripts.train_multitask [--epochs 1000] [--batch_size 64] [--lr 1e-3]

This script:
  1. Loads the 8M-point Feather dataset from for_HPC/
  2. Filters by gap range, splits train/val
  3. Trains the multi-task NN (patch class + projection + signed distance)
  4. Saves checkpoints and logs metrics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure project root is importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nn_contact.config import DataConfig, MultiTaskConfig, TrainingConfig
from nn_contact.data.loader import make_dataloaders
from nn_contact.models.multitask import MultiTaskContactNet
from nn_contact.training.losses import multitask_loss
from nn_contact.training.trainer import Trainer


def make_loss_fn(cfg: MultiTaskConfig):
    """Build the multi-task loss function matching the Trainer API."""
    def loss_fn(model, batch):
        xyz, patch_id, xi, gn, normal, dndxs = batch
        outputs = model(xyz)
        targets = {"gn": gn, "patch_id": patch_id, "xi": xi}
        return multitask_loss(
            outputs, targets,
            lambda_gn=cfg.lambda_gn,
            lambda_patch=cfg.lambda_patch,
            lambda_proj=cfg.lambda_proj,
        )
    return loss_fn


def main():
    parser = argparse.ArgumentParser(description="Train multi-task contact NN")
    parser.add_argument("--feather", default="for_HPC/Points4Train_TR_LV.ft")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "adadelta"])
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--checkpoint_dir", default="nn_contact/checkpoints/multitask")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encoding", default="none", choices=["none", "fourier"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # --- Configs ---
    data_cfg = DataConfig(
        feather_path=args.feather,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model_cfg = MultiTaskConfig(input_encoding=args.encoding)
    train_cfg = TrainingConfig(
        optimizer=args.optimizer,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

    # --- Data ---
    torch.manual_seed(train_cfg.seed)
    loaders = make_dataloaders(data_cfg, seed=train_cfg.seed)

    # --- Model ---
    model = MultiTaskContactNet.from_config(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # --- Train ---
    trainer = Trainer(model, make_loss_fn(model_cfg), train_cfg, device=args.device)

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(Path(args.resume))

    history = trainer.fit(loaders.train, loaders.val)

    # Save final config alongside checkpoint
    ckpt_dir = Path(args.checkpoint_dir)
    torch.save({
        "config": model_cfg,
        "data_config": data_cfg,
        "train_config": train_cfg,
        "normalizer": loaders.normalizer.state_dict(),
    }, ckpt_dir / "config.pt")

    print(f"\nCheckpoints saved to {ckpt_dir}/")
    print(f"Best validation loss: {trainer.best_val_loss:.5e}")


if __name__ == "__main__":
    main()

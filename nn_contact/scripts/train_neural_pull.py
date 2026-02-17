"""Train the Neural-Pull SDF network (Phase 2).

Usage:
    python -m nn_contact.scripts.train_neural_pull [--arch siren] [--epochs 2000]

This script trains a network to approximate the signed distance field and its
derivatives (normal, Hessian) using autodiff through the network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nn_contact.config import DataConfig, NeuralPullConfig, TrainingConfig
from nn_contact.data.loader import make_dataloaders
from nn_contact.models.neural_pull import NeuralPullNet
from nn_contact.training.losses import neural_pull_loss
from nn_contact.training.trainer import Trainer


def make_loss_fn(cfg: NeuralPullConfig):
    """Build loss function for Neural-Pull with autodiff derivatives."""
    def loss_fn(model, batch):
        xyz, patch_id, xi, gn, normal, dndxs = batch
        xyz = xyz.requires_grad_(True)

        if cfg.lambda_hess > 0:
            g_pred, grad_pred, hess_pred = model.forward_with_hessian(xyz)
            return neural_pull_loss(
                g_pred, grad_pred, gn, normal,
                hess_pred=hess_pred, dndxs_target=dndxs,
                lambda_sdf=cfg.lambda_sdf,
                lambda_grad=cfg.lambda_grad,
                lambda_hess=cfg.lambda_hess,
                lambda_eikonal=cfg.lambda_eikonal,
            )
        else:
            g_pred, grad_pred = model.forward_with_grad(xyz)
            return neural_pull_loss(
                g_pred, grad_pred, gn, normal,
                lambda_sdf=cfg.lambda_sdf,
                lambda_grad=cfg.lambda_grad,
                lambda_eikonal=cfg.lambda_eikonal,
            )
    return loss_fn


def main():
    parser = argparse.ArgumentParser(description="Train Neural-Pull SDF network")
    parser.add_argument("--feather", default="for_HPC/Points4Train_TR_LV.ft")
    parser.add_argument("--arch", default="siren", choices=["siren", "fourier_mlp", "mlp"])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_grad", type=float, default=10.0)
    parser.add_argument("--lambda_hess", type=float, default=1.0)
    parser.add_argument("--lambda_eikonal", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--checkpoint_dir", default="nn_contact/checkpoints/neural_pull")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_cfg = DataConfig(feather_path=args.feather, batch_size=args.batch_size)
    model_cfg = NeuralPullConfig(
        architecture=args.arch,
        lambda_grad=args.lambda_grad,
        lambda_hess=args.lambda_hess,
        lambda_eikonal=args.lambda_eikonal,
    )
    train_cfg = TrainingConfig(
        optimizer="adam",
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

    torch.manual_seed(train_cfg.seed)
    loaders = make_dataloaders(data_cfg, seed=train_cfg.seed)

    model = NeuralPullNet.from_config(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model ({args.arch}): {n_params:,} parameters")

    trainer = Trainer(model, make_loss_fn(model_cfg), train_cfg, device=args.device)
    history = trainer.fit(loaders.train, loaders.val)

    ckpt_dir = Path(args.checkpoint_dir)
    torch.save({
        "config": model_cfg,
        "data_config": data_cfg,
        "normalizer": loaders.normalizer.state_dict(),
    }, ckpt_dir / "config.pt")

    print(f"\nBest validation loss: {trainer.best_val_loss:.5e}")


if __name__ == "__main__":
    main()

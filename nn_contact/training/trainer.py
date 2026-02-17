"""Generic training loop with logging, early stopping, and checkpointing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nn_contact.config import TrainingConfig


class Trainer:
    """Training loop for any nn_contact model.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    loss_fn : callable
        ``loss_fn(model, batch) -> (loss_tensor, breakdown_dict)``
    cfg : TrainingConfig
        Training hyperparameters.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        cfg: TrainingConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # State
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: list[dict] = []

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.cfg
        if cfg.optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "adadelta":
            return torch.optim.Adadelta(self.model.parameters(), lr=cfg.adadelta_lr, rho=cfg.adadelta_rho)
        elif cfg.optimizer == "lbfgs":
            return torch.optim.LBFGS(self.model.parameters(), lr=cfg.lr, max_iter=20)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _build_scheduler(self):
        cfg = self.cfg
        if cfg.scheduler == "none":
            return None
        elif cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs)
        elif cfg.scheduler == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=cfg.scheduler_patience,
                factor=cfg.scheduler_factor)
        elif cfg.scheduler == "warmup_cosine":
            def lr_lambda(epoch):
                if epoch < cfg.warmup_epochs:
                    return epoch / max(1, cfg.warmup_epochs)
                progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
                return 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

    def _to_device(self, batch: tuple) -> tuple:
        return tuple(
            t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
            for t in batch
        )

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_breakdowns: dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            loss, breakdown = self.loss_fn(self.model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in breakdown.items():
                all_breakdowns[k] = all_breakdowns.get(k, 0.0) + v
            n_batches += 1

        avg = {k: v / n_batches for k, v in all_breakdowns.items()}
        avg["loss_avg"] = total_loss / n_batches
        return avg

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_breakdowns: dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            batch = self._to_device(batch)
            loss, breakdown = self.loss_fn(self.model, batch)
            total_loss += loss.item()
            for k, v in breakdown.items():
                all_breakdowns[k] = all_breakdowns.get(k, 0.0) + v
            n_batches += 1

        avg = {k: v / n_batches for k, v in all_breakdowns.items()}
        avg["loss_avg"] = total_loss / n_batches
        return avg

    def save_checkpoint(self, path: Path, is_best: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        if self.scheduler is not None:
            state["scheduler_state"] = self.scheduler.state_dict()
        torch.save(state, path)
        if is_best:
            best_path = path.parent / "best_model.pt"
            torch.save(state, best_path)

    def load_checkpoint(self, path: Path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.epoch = state["epoch"]
        self.best_val_loss = state["best_val_loss"]
        self.history = state.get("history", [])
        if self.scheduler is not None and "scheduler_state" in state:
            self.scheduler.load_state_dict(state["scheduler_state"])

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> list[dict]:
        """Run full training loop with early stopping.

        Returns the training history (list of per-epoch metric dicts).
        """
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        t_start = time.time()

        for epoch in range(self.epoch, self.cfg.epochs):
            self.epoch = epoch
            t_epoch = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss_avg"])
                else:
                    self.scheduler.step()

            # Checkpointing
            is_best = val_metrics["loss_avg"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss_avg"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.cfg.save_best_only:
                if is_best:
                    self.save_checkpoint(ckpt_dir / "checkpoint.pt", is_best=True)
            else:
                self.save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", is_best=is_best)

            # Log
            lr = self.optimizer.param_groups[0]["lr"]
            record = {
                "epoch": epoch,
                "lr": lr,
                "train": train_metrics,
                "val": val_metrics,
                "best_val_loss": self.best_val_loss,
                "dt": time.time() - t_epoch,
            }
            self.history.append(record)

            if verbose and (epoch % self.cfg.log_every == 0 or is_best):
                star = " *" if is_best else ""
                print(
                    f"[{epoch:4d}/{self.cfg.epochs}] "
                    f"train={train_metrics['loss_avg']:.5e}  "
                    f"val={val_metrics['loss_avg']:.5e}  "
                    f"lr={lr:.2e}  "
                    f"dt={record['dt']:.1f}s{star}"
                )

            # Early stopping
            if self.patience_counter >= self.cfg.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (patience={self.cfg.patience})")
                break

        elapsed = time.time() - t_start
        if verbose:
            print(f"Training complete: {elapsed:.0f}s, best val loss: {self.best_val_loss:.5e}")

        return self.history

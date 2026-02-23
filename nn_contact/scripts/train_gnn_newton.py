"""Train the GNN Newton Step Predictor (Phase 4).

Trains an Encode-Process-Decode GNN to predict the first Newton displacement
increment Δu from the current FEM state (displacement, residual, contact).

Training protocol:
  - Curriculum: first 100 epochs on early load steps (≤30% load), then full data
  - Loss: normalized MSE on free DOFs only
  - Optimizer: AdamW + CosineAnnealing
  - Validation: relative du error + energy decrease rate

Usage:
    # Local training
    python -m nn_contact.scripts.train_gnn_newton --data-root nn_contact/data/gnn_newton_processed

    # HPC CPU-optimized
    python -m nn_contact.scripts.train_gnn_newton --data-root nn_contact/data/gnn_newton_processed \
        --epochs 500 --batch-size 32 --num-threads 16

    # Custom architecture
    python -m nn_contact.scripts.train_gnn_newton --hidden 64 --n-layers 6 --epochs 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nn_contact.config import GNNNewtonConfig
from nn_contact.data.gnn_newton_dataset import GNNNewtonDataset, split_by_simulation
from nn_contact.models.gnn_newton import GNNNewtonPredictor, GCNNewtonPredictor, newton_step_loss


class SizeGroupedSampler(Sampler):
    """Groups graphs by node count for efficient GPU batching.

    Avoids mixing n=5 (216 nodes) and n=10 (1331 nodes) in the same batch,
    preventing GPU underutilization from padding/variable scatter sizes.
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True,
                 seed: int = 42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

        # Group indices by node count
        groups: dict[int, list[int]] = {}
        for i in range(len(dataset)):
            nv = dataset[i].x.shape[0]
            groups.setdefault(nv, []).append(i)
        self.groups = groups

    def __iter__(self):
        all_batches = []
        for nv, indices in self.groups.items():
            idx = np.array(indices)
            if self.shuffle:
                self.rng.shuffle(idx)
            # Chunk into batches
            for start in range(0, len(idx), self.batch_size):
                all_batches.append(idx[start:start + self.batch_size].tolist())

        if self.shuffle:
            self.rng.shuffle(all_batches)

        for batch in all_batches:
            yield from batch

    def __len__(self):
        return sum(len(v) for v in self.groups.values())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GNN Newton Step Predictor"
    )
    # Data
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root dir for PyG dataset (contains raw/ with .npz files)")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)

    # Architecture
    parser.add_argument("--model-type", type=str, default="mpn",
                        choices=["mpn", "gcn"],
                        help="mpn = edge-conditioned MPN (accurate, needs GPU); "
                             "gcn = GCN-based (fast on CPU)")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--activation", type=str, default="silu",
                        choices=["silu", "relu", "gelu"])

    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--curriculum-epochs", type=int, default=100,
                        help="Epochs to train on easy samples only (≤30%% load)")
    parser.add_argument("--curriculum-frac", type=float, default=0.3,
                        help="Load fraction cutoff for curriculum phase")

    # System
    parser.add_argument("--num-threads", type=int, default=None,
                        help="PyTorch CPU threads (default: auto)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (4 for GPU, 0 for CPU)")
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--checkpoint-dir", type=str,
                        default="nn_contact/checkpoints/gnn_newton")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience")

    return parser.parse_args()


def train_epoch(
    model: GNNNewtonPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Train for one epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = newton_step_loss(out, batch.y, batch.free_mask, batch.batch)

        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: GNNNewtonPredictor,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Validate model. Returns dict of metrics."""
    model.eval()
    total_loss = 0.0
    total_rel_error = 0.0
    n_batches = 0
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = newton_step_loss(out, batch.y, batch.free_mask, batch.batch)

        total_loss += loss.item()

        # Relative error on free DOFs only (consistent with loss)
        masked_diff = ((out - batch.y) * batch.free_mask)
        masked_true = (batch.y * batch.free_mask)
        rel = masked_diff.norm() / (masked_true.norm() + 1e-10)
        total_rel_error += rel.item()

        n_batches += 1
        if batch.batch is not None:
            n_samples += batch.batch.max().item() + 1
        else:
            n_samples += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "rel_error": total_rel_error / max(n_batches, 1),
        "n_samples": n_samples,
    }


def main():
    args = parse_args()

    # Set threads
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(max(1, args.num_threads // 4))

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    print(f"Loading dataset from {args.data_root}...")
    dataset = GNNNewtonDataset(root=args.data_root, normalize=True)
    print(f"  Total samples: {len(dataset)}")

    # Split
    train_idx, val_idx, test_idx = split_by_simulation(
        dataset, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Curriculum: identify early load-step samples
    early_train_idx = []
    for i in train_idx:
        data = dataset[i]
        if data.load_fraction.item() <= args.curriculum_frac:
            early_train_idx.append(i)
    early_val_idx = []
    for i in val_idx:
        data = dataset[i]
        if data.load_fraction.item() <= args.curriculum_frac:
            early_val_idx.append(i)
    print(f"  Curriculum: {len(early_train_idx)} early train, "
          f"{len(early_val_idx)} early val samples "
          f"(load ≤ {args.curriculum_frac})")

    # DataLoaders
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    early_dataset = dataset[early_train_idx] if early_train_idx else train_dataset
    early_val_dataset = dataset[early_val_idx] if early_val_idx else val_dataset

    nw = args.num_workers
    use_gpu = (device != "cpu")

    def _make_loader(ds, shuffle=True):
        if use_gpu and shuffle:
            return DataLoader(
                ds, batch_size=args.batch_size,
                sampler=SizeGroupedSampler(ds, args.batch_size,
                                           shuffle=True, seed=args.seed),
                num_workers=nw, persistent_workers=nw > 0)
        return DataLoader(ds, batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=nw,
                          persistent_workers=nw > 0)

    train_loader = _make_loader(train_dataset)
    early_loader = _make_loader(early_dataset)
    val_loader = _make_loader(val_dataset, shuffle=False)
    early_val_loader = _make_loader(early_val_dataset, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────
    cfg = GNNNewtonConfig(
        hidden=args.hidden,
        n_layers=args.n_layers,
        activation=args.activation,
    )
    model_type = args.model_type
    if model_type == "gcn":
        model = GCNNewtonPredictor(cfg).to(device)
    else:
        model = GNNNewtonPredictor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel ({model_type}): {n_params:,} parameters")
    print(f"  hidden={cfg.hidden}, layers={cfg.n_layers}, "
          f"activation={cfg.activation}")

    # ── Optimizer + scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── Training loop ────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    t_start = time.perf_counter()

    print(f"\nTraining for {args.epochs} epochs "
          f"(curriculum: first {args.curriculum_epochs} on easy samples)")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.perf_counter()

        # Curriculum: use easy data for first N epochs
        in_curriculum = (epoch <= args.curriculum_epochs and early_train_idx)
        if in_curriculum:
            loader = early_loader
            phase = "curriculum"
        else:
            loader = train_loader
            phase = "full"

        train_loss = train_epoch(model, loader, optimizer, device)
        scheduler.step()

        # Validate on matching distribution
        if in_curriculum:
            val_metrics = validate(model, early_val_loader, device)
        else:
            val_metrics = validate(model, val_loader, device)
        val_loss = val_metrics["loss"]
        val_rel = val_metrics["rel_error"]

        dt = time.perf_counter() - t_epoch

        # Logging
        if epoch % args.log_every == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"rel_err={val_rel:.4f}  lr={lr:.2e}  "
                  f"[{phase}]  {dt:.1f}s")

        # Reset best/patience on curriculum→full transition
        if epoch == args.curriculum_epochs + 1:
            best_val_loss = float("inf")
            patience_counter = 0

        # Checkpointing (only on full-data phase for final model)
        if not in_curriculum and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            ckpt = {
                "config": cfg,
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_rel_error": val_rel,
                "args": vars(args),
            }
            torch.save(ckpt, ckpt_dir / "best.pt")
        elif not in_curriculum:
            patience_counter += 1

        # Early stopping (only after curriculum ends)
        if not in_curriculum and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # ── Summary ──────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_start
    print(f"\n{'='*80}")
    print(f"  Training complete: {epoch} epochs in {t_total:.1f}s")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Checkpoint: {ckpt_dir / 'best.pt'}")

    # ── Final test evaluation ────────────────────────────────────────────
    if test_idx:
        test_dataset = dataset[test_idx]
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        # Load best model
        best_ckpt = torch.load(ckpt_dir / "best.pt", weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

        test_metrics = validate(model, test_loader, device)
        print(f"  Test loss: {test_metrics['loss']:.6f}")
        print(f"  Test rel_error: {test_metrics['rel_error']:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

"""Data loading pipeline: Feather -> filtered numpy -> PyTorch DataLoader.

Reads the 8M-point training data produced by ``for_HPC/compute_tr_projection_batch.py``,
applies gap-range filtering, stratified train/val split, coordinate normalization,
and exposes the result as PyTorch ``DataLoader`` objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow.feather as feather
import torch
from torch.utils.data import DataLoader, TensorDataset

from nn_contact.config import DataConfig
from nn_contact.data.normalization import CoordinateNormalizer


class ContactDataArrays(NamedTuple):
    """Raw numpy arrays after filtering and splitting."""

    xyz: np.ndarray           # (N, 3) float32 — normalized coordinates
    patch_id: np.ndarray      # (N,)   int64   — 0..95
    xi: np.ndarray            # (N, 2) float32 — parametric coords
    gn: np.ndarray            # (N,)   float32 — signed distance
    normal: np.ndarray        # (N, 3) float32 — surface normal
    dndxs: np.ndarray         # (N, 9) float32 — dn/dx_s flattened
    normalizer: CoordinateNormalizer


def load_feather(cfg: DataConfig, verbose: bool = True) -> ContactDataArrays:
    """Load and preprocess the Feather training data.

    Steps:
      1. Read only needed columns (avoids loading full 2 GB into memory twice).
      2. Filter by gap range ``[gn_min, gn_max]``.
      3. Normalize xyz by characteristic length.
      4. Return as float32 numpy arrays.
    """
    path = Path(cfg.feather_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path.resolve()}")

    cols = list(cfg.input_cols) + [
        cfg.patch_col,
        *cfg.xi_cols,
        cfg.gn_col,
        *cfg.normal_cols,
        *cfg.dndxs_cols,
    ]
    if verbose:
        print(f"Loading {path} ...")
    df = feather.read_feather(str(path), columns=cols)
    if verbose:
        print(f"  raw: {len(df):,} points, {len(df.columns)} columns")

    # --- gap filtering ---
    mask = (df[cfg.gn_col] >= cfg.gn_min) & (df[cfg.gn_col] <= cfg.gn_max)
    df = df.loc[mask].reset_index(drop=True)
    if verbose:
        print(f"  after gap filter [{cfg.gn_min}, {cfg.gn_max}]: {len(df):,} points")

    # --- extract arrays ---
    xyz_raw = df[list(cfg.input_cols)].values.astype(np.float32)

    normalizer = CoordinateNormalizer(char_length=cfg.char_length)
    normalizer.fit(xyz_raw)
    xyz = normalizer.transform(xyz_raw).astype(np.float32)

    patch_id = df[cfg.patch_col].values.astype(np.int64)
    xi = df[list(cfg.xi_cols)].values.astype(np.float32)
    gn = df[cfg.gn_col].values.astype(np.float32)
    normal = df[list(cfg.normal_cols)].values.astype(np.float32)
    dndxs = df[list(cfg.dndxs_cols)].values.astype(np.float32)

    return ContactDataArrays(
        xyz=xyz, patch_id=patch_id, xi=xi, gn=gn,
        normal=normal, dndxs=dndxs, normalizer=normalizer,
    )


def stratified_split(
    patch_ids: np.ndarray,
    val_fraction: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/val index split by patch ID.

    Returns ``(train_idx, val_idx)`` index arrays.
    """
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for pid in np.unique(patch_ids):
        idx = np.where(patch_ids == pid)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


class ContactDataLoaders(NamedTuple):
    train: DataLoader
    val: DataLoader
    normalizer: CoordinateNormalizer
    n_train: int
    n_val: int


def make_dataloaders(cfg: DataConfig, seed: int = 42, verbose: bool = True) -> ContactDataLoaders:
    """End-to-end: Feather -> train/val DataLoaders.

    Each batch yields ``(xyz, patch_id, xi, gn, normal, dndxs)``.
    """
    data = load_feather(cfg, verbose=verbose)

    # --- stratified split ---
    if cfg.stratify_by_patch:
        train_idx, val_idx = stratified_split(data.patch_id, cfg.val_fraction, seed)
    else:
        n = len(data.xyz)
        idx = np.random.RandomState(seed).permutation(n)
        n_val = int(n * cfg.val_fraction)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

    def _make_dataset(idx: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(data.xyz[idx]),
            torch.from_numpy(data.patch_id[idx]),
            torch.from_numpy(data.xi[idx]),
            torch.from_numpy(data.gn[idx]),
            torch.from_numpy(data.normal[idx]),
            torch.from_numpy(data.dndxs[idx]),
        )

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        _make_dataset(train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        _make_dataset(val_idx),
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    if verbose:
        print(f"  train: {len(train_idx):,}  val: {len(val_idx):,}")
        print(f"  batches/epoch: {len(train_loader):,} train, {len(val_loader):,} val")

    return ContactDataLoaders(
        train=train_loader,
        val=val_loader,
        normalizer=data.normalizer,
        n_train=len(train_idx),
        n_val=len(val_idx),
    )

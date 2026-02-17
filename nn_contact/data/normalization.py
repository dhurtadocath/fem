"""Input/output normalization for contact NN models.

Coordinates are divided by a characteristic length so that the input domain
is O(1).  Output targets (gap, parametric coords) are already O(1) by
construction of the Gregory patch parameterization.
"""

from __future__ import annotations

import numpy as np
import torch


class CoordinateNormalizer:
    """Normalize xyz coordinates by characteristic length and optional centering.

    Parameters
    ----------
    char_length : float
        Divide coordinates by this value (approximate bounding box extent).
    center : ndarray (3,) or None
        If given, subtract before dividing.  Computed from data if ``fit`` is
        called.
    """

    def __init__(self, char_length: float = 8.0, center: np.ndarray | None = None):
        self.char_length = char_length
        self.center = center  # (3,) or None

    def fit(self, xyz: np.ndarray) -> "CoordinateNormalizer":
        """Compute center from data (mean of min/max per axis)."""
        lo = xyz.min(axis=0)
        hi = xyz.max(axis=0)
        self.center = 0.5 * (lo + hi)
        return self

    def transform(self, xyz: np.ndarray) -> np.ndarray:
        out = xyz.copy()
        if self.center is not None:
            out -= self.center
        out /= self.char_length
        return out

    def inverse_transform(self, xyz_norm: np.ndarray) -> np.ndarray:
        out = xyz_norm * self.char_length
        if self.center is not None:
            out += self.center
        return out

    def transform_torch(self, xyz: torch.Tensor) -> torch.Tensor:
        out = xyz.clone()
        if self.center is not None:
            out -= torch.as_tensor(self.center, dtype=out.dtype, device=out.device)
        out /= self.char_length
        return out

    def state_dict(self) -> dict:
        return {"char_length": self.char_length, "center": self.center}

    @classmethod
    def from_state_dict(cls, d: dict) -> "CoordinateNormalizer":
        return cls(char_length=d["char_length"], center=d["center"])

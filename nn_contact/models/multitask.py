"""Multi-task NN for contact detection (Phase 1).

Architecture (from paper2.tex Section 5, Fig. MTL-arch):
  Input (x,y,z) -> Shared trunk -> 3 task heads:
    1. Signed distance (filter):     trunk -> MLP -> 1 scalar
    2. Patch classification:         trunk -> MLP -> softmax(96)
    3. Parametric projection:        trunk -> MLP -> 2*96 outputs (segmented regression)

The segmented regression (Eq. 15) only supervises the (xi1, xi2) pair
corresponding to the true closest patch — all other patch outputs are masked.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_contact.config import MultiTaskConfig
from nn_contact.models.mlp import MLP, FourierFeatures, get_activation


class MultiTaskContactNet(nn.Module):
    """Multi-task network for simultaneous patch ID, projection, and gap prediction.

    Parameters
    ----------
    cfg : MultiTaskConfig
        Full model configuration.
    in_dim : int
        Input dimension (3 for raw xyz, or 2*n_freq for Fourier-encoded).
    """

    def __init__(self, cfg: MultiTaskConfig, in_dim: int = 3):
        super().__init__()
        self.cfg = cfg
        self.n_patches = cfg.n_patches

        # Optional Fourier encoding
        self.fourier = None
        trunk_in = in_dim
        if cfg.input_encoding == "fourier" and cfg.fourier_config is not None:
            fc = cfg.fourier_config
            self.fourier = FourierFeatures(in_dim, fc.n_frequencies, fc.frequency_scale)
            trunk_in = self.fourier.out_dim

        # Shared trunk
        trunk_dims = cfg.trunk.hidden_dims
        trunk_layers: list[nn.Module] = []
        dims = [trunk_in] + trunk_dims
        for i in range(len(dims) - 1):
            trunk_layers.append(nn.Linear(dims[i], dims[i + 1]))
            trunk_layers.append(get_activation(cfg.trunk.activation))
        self.trunk = nn.Sequential(*trunk_layers)
        trunk_out = trunk_dims[-1]

        # Task 1: Signed distance head (scalar regression)
        gn_layers: list[nn.Module] = []
        gn_dims = [trunk_out] + cfg.gn_head_dims
        for i in range(len(gn_dims) - 1):
            gn_layers.append(nn.Linear(gn_dims[i], gn_dims[i + 1]))
            gn_layers.append(nn.ReLU())
        gn_layers.append(nn.Linear(gn_dims[-1], 1))
        self.gn_head = nn.Sequential(*gn_layers)

        # Task 2: Patch classification head (96-class softmax)
        patch_layers: list[nn.Module] = []
        patch_dims = [trunk_out] + cfg.patch_head_dims
        for i in range(len(patch_dims) - 1):
            patch_layers.append(nn.Linear(patch_dims[i], patch_dims[i + 1]))
            patch_layers.append(nn.ReLU())
        patch_layers.append(nn.Linear(patch_dims[-1], cfg.n_patches))
        self.patch_head = nn.Sequential(*patch_layers)

        # Task 3: Parametric projection head (2*n_patches segmented regression)
        proj_layers: list[nn.Module] = []
        proj_dims = [trunk_out] + cfg.proj_head_dims
        for i in range(len(proj_dims) - 1):
            proj_layers.append(nn.Linear(proj_dims[i], proj_dims[i + 1]))
            proj_layers.append(nn.ReLU())
        proj_layers.append(nn.Linear(proj_dims[-1], 2 * cfg.n_patches))
        self.proj_head = nn.Sequential(*proj_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyz: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        xyz : (B, 3) tensor
            Normalized spatial coordinates.

        Returns
        -------
        dict with keys:
            gn_pred    : (B, 1)          — predicted signed distance
            patch_logits : (B, 96)       — raw logits for patch classification
            xi_pred    : (B, 96, 2)      — predicted (xi1, xi2) per patch
        """
        if self.fourier is not None:
            xyz = self.fourier(xyz)

        features = self.trunk(xyz)

        gn_pred = self.gn_head(features)                    # (B, 1)
        patch_logits = self.patch_head(features)             # (B, 96)
        xi_flat = self.proj_head(features)                   # (B, 192)
        xi_pred = xi_flat.view(-1, self.n_patches, 2)        # (B, 96, 2)

        return {
            "gn_pred": gn_pred,
            "patch_logits": patch_logits,
            "xi_pred": xi_pred,
        }

    def predict(self, xyz: torch.Tensor, topk: int = 1) -> dict[str, torch.Tensor]:
        """Inference: return top-K patch candidates with corresponding xi.

        Parameters
        ----------
        xyz : (B, 3)
        topk : int
            Number of candidate patches to return.

        Returns
        -------
        dict with:
            gn       : (B,)         — predicted signed distance
            patch_ids: (B, topk)    — top-K patch IDs
            patch_probs: (B, topk)  — softmax probabilities
            xi       : (B, topk, 2) — parametric coords for each candidate
        """
        out = self.forward(xyz)
        gn = out["gn_pred"].squeeze(-1)

        probs = F.softmax(out["patch_logits"], dim=-1)        # (B, 96)
        top_probs, top_ids = probs.topk(topk, dim=-1)         # (B, topk)

        # Gather xi for top-K patches
        # xi_pred is (B, 96, 2) — index with top_ids (B, topk)
        idx = top_ids.unsqueeze(-1).expand(-1, -1, 2)         # (B, topk, 2)
        xi = torch.gather(out["xi_pred"], 1, idx)             # (B, topk, 2)

        return {
            "gn": gn,
            "patch_ids": top_ids,
            "patch_probs": top_probs,
            "xi": xi,
        }

    @classmethod
    def from_config(cls, cfg: MultiTaskConfig) -> "MultiTaskContactNet":
        return cls(cfg, in_dim=3)

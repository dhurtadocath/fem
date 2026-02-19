"""Multi-task NN for contact detection (Phase 1).

Architecture (from paper2.tex Section 5, Fig. MTL-arch):
  Input (x,y,z) -> Shared trunk -> 3 task heads:
    1. Signed distance (filter):     trunk -> MLP -> 1 scalar
    2. Patch classification:         trunk -> MLP -> softmax(96)
    3. Parametric projection:        trunk -> MLP -> 2*96 outputs (segmented regression)

The segmented regression (Eq. 15) only supervises the (xi1, xi2) pair
corresponding to the true closest patch — all other patch outputs are masked.

Extended with:
  - Optional task-specific attention (MTAN-style): sigmoid gate on trunk features
  - Optional patch-conditioned regression: FiLM-modulated head using patch embeddings
  - Optional manifold mixup hook (applied at trunk output level)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_contact.config import MultiTaskConfig
from nn_contact.models.mlp import MLP, FourierFeatures, get_activation


class TaskAttention(nn.Module):
    """MTAN-style task-specific attention gate.

    Learns a sigmoid mask over shared features so each task can select
    the most relevant trunk dimensions.

    Ref: Liu et al., "End-to-End Multi-Task Learning with Attention",
    CVPR 2019.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class PatchConditionedHead(nn.Module):
    """FiLM-conditioned regression head for parametric coordinates.

    Instead of outputting 2*96 values (segmented regression), this head
    takes (features, patch_embedding) and outputs (xi1, xi2) for one patch.
    The patch embedding modulates features via FiLM: gamma * features + beta.

    At training time: uses true patch_id for conditioning.
    At inference time: uses predicted patch_id(s) for conditioning.
    """

    def __init__(self, trunk_dim: int, n_patches: int, head_dims: list[int],
                 embed_dim: int = 32):
        super().__init__()
        self.n_patches = n_patches
        self.embed_dim = embed_dim

        # Learnable patch embeddings
        self.patch_embed = nn.Embedding(n_patches, embed_dim)

        # FiLM: patch embedding -> (gamma, beta) for trunk features
        self.film = nn.Linear(embed_dim, 2 * trunk_dim)

        # Regression MLP: modulated features -> (xi1, xi2)
        layers: list[nn.Module] = []
        dims = [trunk_dim] + head_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 2))
        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, patch_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        features : (B, D) — trunk features
        patch_ids : (B,) — patch indices for conditioning

        Returns
        -------
        xi_pred : (B, 2) — predicted (xi1, xi2)
        """
        emb = self.patch_embed(patch_ids)  # (B, embed_dim)
        film_params = self.film(emb)  # (B, 2*D)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, D)
        modulated = (1 + gamma) * features + beta  # FiLM modulation
        return self.head(modulated)  # (B, 2)

    def forward_multi(self, features: torch.Tensor, patch_ids: torch.Tensor) -> torch.Tensor:
        """Forward for multiple patch candidates per sample.

        Parameters
        ----------
        features : (B, D) — trunk features
        patch_ids : (B, K) — K patch candidates per sample

        Returns
        -------
        xi_pred : (B, K, 2) — predicted (xi1, xi2) per candidate
        """
        B, K = patch_ids.shape
        D = features.shape[1]

        emb = self.patch_embed(patch_ids)  # (B, K, embed_dim)
        film_params = self.film(emb)  # (B, K, 2*D)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, K, D)

        feat_exp = features.unsqueeze(1).expand(B, K, D)  # (B, K, D)
        modulated = (1 + gamma) * feat_exp + beta  # (B, K, D)

        # Flatten for MLP, then reshape
        xi = self.head(modulated.reshape(B * K, D))  # (B*K, 2)
        return xi.view(B, K, 2)


class MultiTaskContactNet(nn.Module):
    """Multi-task network for simultaneous patch ID, projection, and gap prediction.

    Parameters
    ----------
    cfg : MultiTaskConfig
        Full model configuration.
    in_dim : int
        Input dimension (3 for raw xyz, or 2*n_freq for Fourier-encoded).
    task_attention : bool
        If True, add MTAN-style attention gates per task head.
    patch_conditioned : bool
        If True, use FiLM-conditioned regression instead of segmented regression.
    """

    def __init__(self, cfg: MultiTaskConfig, in_dim: int = 3,
                 task_attention: bool = False,
                 patch_conditioned: bool = False):
        super().__init__()
        self.cfg = cfg
        self.n_patches = cfg.n_patches
        self.use_task_attention = task_attention
        self.use_patch_conditioned = patch_conditioned

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

        # Optional task attention gates
        if task_attention:
            self.attn_gn = TaskAttention(trunk_out)
            self.attn_patch = TaskAttention(trunk_out)
            self.attn_proj = TaskAttention(trunk_out)

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

        # Task 3: Parametric projection
        if patch_conditioned:
            self.proj_head_cond = PatchConditionedHead(
                trunk_out, cfg.n_patches, cfg.proj_head_dims, embed_dim=32,
            )
            self.proj_head = None
        else:
            # Standard segmented regression (2*n_patches outputs)
            proj_layers: list[nn.Module] = []
            proj_dims = [trunk_out] + cfg.proj_head_dims
            for i in range(len(proj_dims) - 1):
                proj_layers.append(nn.Linear(proj_dims[i], proj_dims[i + 1]))
                proj_layers.append(nn.ReLU())
            proj_layers.append(nn.Linear(proj_dims[-1], 2 * cfg.n_patches))
            self.proj_head = nn.Sequential(*proj_layers)
            self.proj_head_cond = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyz: torch.Tensor,
                patch_id_true: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        xyz : (B, 3) tensor
            Normalized spatial coordinates.
        patch_id_true : (B,) tensor, optional
            True patch IDs for patch-conditioned regression during training.

        Returns
        -------
        dict with keys:
            gn_pred      : (B, 1)          — predicted signed distance
            patch_logits : (B, 96)         — raw logits for patch classification
            xi_pred      : (B, 96, 2)      — predicted (xi1, xi2) per patch
                           OR (B, 2) if patch_conditioned and patch_id_true given
            features     : (B, D)          — trunk features (for manifold mixup)
        """
        if self.fourier is not None:
            xyz = self.fourier(xyz)

        features = self.trunk(xyz)

        # Apply task attention if enabled
        feat_gn = self.attn_gn(features) if self.use_task_attention else features
        feat_patch = self.attn_patch(features) if self.use_task_attention else features
        feat_proj = self.attn_proj(features) if self.use_task_attention else features

        gn_pred = self.gn_head(feat_gn)                      # (B, 1)
        patch_logits = self.patch_head(feat_patch)            # (B, 96)

        result = {
            "gn_pred": gn_pred,
            "patch_logits": patch_logits,
            "features": features,
        }

        if self.use_patch_conditioned:
            if patch_id_true is not None:
                # Training: use true patch for conditioning
                xi_cond = self.proj_head_cond(feat_proj, patch_id_true)  # (B, 2)
                result["xi_cond"] = xi_cond
            # Always produce full xi_pred for compatibility with segmented loss
            # by running all 96 patches (only at eval or if needed)
            all_pids = torch.arange(self.n_patches, device=features.device)
            all_pids = all_pids.unsqueeze(0).expand(features.shape[0], -1)  # (B, 96)
            xi_pred = self.proj_head_cond.forward_multi(feat_proj, all_pids)  # (B, 96, 2)
            result["xi_pred"] = xi_pred
        else:
            xi_flat = self.proj_head(feat_proj)                # (B, 192)
            xi_pred = xi_flat.view(-1, self.n_patches, 2)      # (B, 96, 2)
            result["xi_pred"] = xi_pred

        return result

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

        if self.use_patch_conditioned:
            # Use FiLM head for top-K patches
            features = out["features"]
            feat_proj = self.attn_proj(features) if self.use_task_attention else features
            xi = self.proj_head_cond.forward_multi(feat_proj, top_ids)  # (B, topk, 2)
        else:
            # Gather xi from full (B, 96, 2) output
            idx = top_ids.unsqueeze(-1).expand(-1, -1, 2)      # (B, topk, 2)
            xi = torch.gather(out["xi_pred"], 1, idx)          # (B, topk, 2)

        return {
            "gn": gn,
            "patch_ids": top_ids,
            "patch_probs": top_probs,
            "xi": xi,
        }

    @classmethod
    def from_config(cls, cfg: MultiTaskConfig,
                    task_attention: bool = False,
                    patch_conditioned: bool = False) -> "MultiTaskContactNet":
        return cls(cfg, in_dim=3, task_attention=task_attention,
                   patch_conditioned=patch_conditioned)

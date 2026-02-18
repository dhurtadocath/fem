"""Neural-Pull: direct SDF with gradient/Hessian supervision (Phase 2).

The network learns g(x) such that:
  - g(x) ≈ signed distance to the rigid surface
  - ∇g(x) ≈ outward normal n*(x)
  - ∇²g(x) ≈ dn*/dx_s (normal derivative)

Derivatives are obtained via automatic differentiation through the network,
giving exact gradients of the NN approximation.  Supervision on ∇g and ∇²g
forces the network to produce simulation-quality derivatives.

Dual-head mode (cfg.dual_head=True):
  Adds an explicit gradient output head that directly predicts the normal
  vector n(x) as a 3-vector output. The SDF head and gradient head share
  a common trunk. At inference, the gradient head output is used for the
  normal, bypassing autodiff chain-rule amplification.

The key challenge (identified in paper2.tex): the normal field is
discontinuous across Gregory patch boundaries, making it inherently hard
for a smooth NN.  SIREN and Fourier features help capture high-frequency
content near these boundaries.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nn_contact.config import NeuralPullConfig
from nn_contact.models.mlp import SIREN, FourierMLP, MLP, build_backbone


class NeuralPullNet(nn.Module):
    """SDF network with autodiff gradient/Hessian support.

    Parameters
    ----------
    cfg : NeuralPullConfig
        Architecture and loss weight configuration.
    """

    def __init__(self, cfg: NeuralPullConfig):
        super().__init__()
        self.cfg = cfg
        self.dual_head = cfg.dual_head

        if cfg.architecture == "siren":
            self.net = SIREN.from_config(cfg.siren, in_dim=3, out_dim=1)
            trunk_dim = cfg.siren.hidden_dims[-1]
        elif cfg.architecture == "fourier_mlp":
            self.net = FourierMLP.from_config(cfg.fourier_mlp, in_dim=3, out_dim=1)
            trunk_dim = cfg.fourier_mlp.hidden_dims[-1]
        elif cfg.architecture == "mlp":
            self.net = MLP.from_config(cfg.mlp, in_dim=3, out_dim=1)
            trunk_dim = cfg.mlp.hidden_dims[-1]
        else:
            raise ValueError(f"Unknown architecture: {cfg.architecture}")

        # Dual-head: explicit gradient output sharing the trunk
        if self.dual_head:
            self.grad_head = nn.Linear(trunk_dim, 3)
            with torch.no_grad():
                # SIREN-style init for the gradient head
                if cfg.architecture == "siren":
                    bound = math.sqrt(6.0 / trunk_dim) / cfg.siren.omega_hidden
                    self.grad_head.weight.uniform_(-bound, bound)
                else:
                    nn.init.xavier_normal_(self.grad_head.weight)
                nn.init.zeros_(self.grad_head.bias)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Predict signed distance g(x).

        Parameters
        ----------
        xyz : (B, 3) — normalized coordinates.

        Returns
        -------
        g : (B, 1) — predicted signed distance.
        """
        return self.net(xyz)

    def forward_with_grad(
        self, xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Predict g and ∇g via autodiff, plus optional direct gradient head.

        Parameters
        ----------
        xyz : (B, 3) — must have requires_grad=True.

        Returns
        -------
        g           : (B, 1)
        grad        : (B, 3) — ∂g/∂x via autodiff
        grad_direct : (B, 3) or None — direct gradient head output (dual-head only)
        """
        xyz = xyz.requires_grad_(True)

        if self.dual_head:
            # Shared trunk → two heads
            features = self.net.forward_trunk(xyz)
            g = self.net.head(features)
            grad_direct = self.grad_head(features)
        else:
            g = self.net(xyz)
            grad_direct = None

        grad = torch.autograd.grad(
            g, xyz,
            grad_outputs=torch.ones_like(g),
            create_graph=True,
            retain_graph=True,
        )[0]
        return g, grad, grad_direct

    def forward_with_hessian(
        self, xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict g, ∇g, and ∇²g via autodiff.

        Parameters
        ----------
        xyz : (B, 3) — coordinates (grad will be enabled internally).

        Returns
        -------
        g    : (B, 1)
        grad : (B, 3)   — ∂g/∂x
        hess : (B, 3, 3) — ∂²g/∂x²
        """
        xyz = xyz.requires_grad_(True)
        g = self.net(xyz)

        grad = torch.autograd.grad(
            g, xyz,
            grad_outputs=torch.ones_like(g),
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, 3)

        # Hessian: differentiate each component of grad w.r.t. xyz
        hess_rows = []
        for i in range(3):
            row = torch.autograd.grad(
                grad[:, i], xyz,
                grad_outputs=torch.ones(xyz.shape[0], device=xyz.device),
                create_graph=True,
                retain_graph=True,
            )[0]  # (B, 3)
            hess_rows.append(row)

        hess = torch.stack(hess_rows, dim=1)  # (B, 3, 3)
        return g, grad, hess

    def predict(self, xyz: torch.Tensor, compute_hessian: bool = False):
        """Inference mode: detached predictions.

        Returns dict with 'gn', 'normal', and optionally 'dndxs'.
        In dual-head mode, uses the direct gradient head for normals.
        """
        with torch.enable_grad():
            if compute_hessian:
                g, grad, hess = self.forward_with_hessian(xyz)
                return {
                    "gn": g.squeeze(-1).detach(),
                    "normal": grad.detach(),
                    "dndxs": hess.detach(),
                }
            else:
                g, grad, grad_direct = self.forward_with_grad(xyz)
                # Use direct gradient head if available (bypasses chain-rule amplification)
                normal = grad_direct if grad_direct is not None else grad
                return {
                    "gn": g.squeeze(-1).detach(),
                    "normal": normal.detach(),
                }

    @classmethod
    def from_config(cls, cfg: NeuralPullConfig) -> "NeuralPullNet":
        return cls(cfg)

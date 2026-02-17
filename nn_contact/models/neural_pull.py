"""Neural-Pull: direct SDF with gradient/Hessian supervision (Phase 2).

The network learns g(x) such that:
  - g(x) ≈ signed distance to the rigid surface
  - ∇g(x) ≈ outward normal n*(x)
  - ∇²g(x) ≈ dn*/dx_s (normal derivative)

Derivatives are obtained via automatic differentiation through the network,
giving exact gradients of the NN approximation.  Supervision on ∇g and ∇²g
forces the network to produce simulation-quality derivatives.

The key challenge (identified in paper2.tex): the normal field is
discontinuous across Gregory patch boundaries, making it inherently hard
for a smooth NN.  SIREN and Fourier features help capture high-frequency
content near these boundaries.
"""

from __future__ import annotations

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

        if cfg.architecture == "siren":
            self.net = SIREN.from_config(cfg.siren, in_dim=3, out_dim=1)
        elif cfg.architecture == "fourier_mlp":
            self.net = FourierMLP.from_config(cfg.fourier_mlp, in_dim=3, out_dim=1)
        elif cfg.architecture == "mlp":
            self.net = MLP.from_config(cfg.mlp, in_dim=3, out_dim=1)
        else:
            raise ValueError(f"Unknown architecture: {cfg.architecture}")

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

    def forward_with_grad(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict g and ∇g via autodiff.

        Parameters
        ----------
        xyz : (B, 3) — must have requires_grad=True.

        Returns
        -------
        g    : (B, 1)
        grad : (B, 3) — ∂g/∂x
        """
        xyz = xyz.requires_grad_(True)
        g = self.net(xyz)
        grad = torch.autograd.grad(
            g, xyz,
            grad_outputs=torch.ones_like(g),
            create_graph=True,
            retain_graph=True,
        )[0]
        return g, grad

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
                g, grad = self.forward_with_grad(xyz)
                return {
                    "gn": g.squeeze(-1).detach(),
                    "normal": grad.detach(),
                }

    @classmethod
    def from_config(cls, cfg: NeuralPullConfig) -> "NeuralPullNet":
        return cls(cfg)

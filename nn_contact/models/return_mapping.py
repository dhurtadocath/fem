"""Learned return mapping surrogate for J2 plasticity (Phase 3).

Replaces the iterative per-GP return mapping (eigendecomposition + exponential
map + FD Newton) with a single batched forward pass through a trained network.

Input (19 scalars per GP):
    F        (9) — total deformation gradient
    Fp_old   (9) — converged plastic deformation gradient
    epcum    (1) — cumulative plastic strain

Output (10 scalars per GP):
    Fp_new   (9) — updated plastic deformation gradient
    delta_ep (1) — plastic strain increment (>= 0)

Physics constraints enforced via loss terms:
    - det(Fp_new) > 0
    - Yield surface consistency
    - Plastic flow direction
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nn_contact.config import ReturnMappingConfig


class ReturnMappingNet(nn.Module):
    """Neural surrogate for J2 multiplicative return mapping.

    Parameters
    ----------
    cfg : ReturnMappingConfig
        Network and loss weight configuration.
    """

    def __init__(self, cfg: ReturnMappingConfig):
        super().__init__()
        self.cfg = cfg

        dims = [cfg.input_dim] + cfg.hidden_dims
        layers: list[nn.Module] = []
        act_cls = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[cfg.activation]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_cls())

        self.backbone = nn.Sequential(*layers)

        # Separate heads for Fp and delta_epcum
        self.fp_head = nn.Linear(cfg.hidden_dims[-1], 9)
        self.dep_head = nn.Linear(cfg.hidden_dims[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize Fp head bias to identity matrix (common initial state)
        with torch.no_grad():
            self.fp_head.bias.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1.0]))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (B, 19) — concatenated [F(9), Fp_old(9), epcum(1)]

        Returns
        -------
        dict with:
            Fp_new    : (B, 9)  — updated Fp (flattened 3x3)
            delta_ep  : (B,)    — plastic strain increment (softplus-enforced >= 0)
            Fp_old    : (B, 9)  — passthrough of input Fp_old for residual connection
        """
        features = self.backbone(x)

        # Fp prediction: residual form — predict correction to Fp_old
        Fp_old = x[:, 9:18]
        Fp_correction = self.fp_head(features)
        Fp_new = Fp_old + Fp_correction

        # delta_epcum: enforce non-negativity with softplus
        delta_ep_raw = self.dep_head(features).squeeze(-1)
        delta_ep = nn.functional.softplus(delta_ep_raw)

        return {
            "Fp_new": Fp_new,
            "delta_ep": delta_ep,
            "Fp_old": Fp_old,
        }

    def predict_numpy(self, F: "np.ndarray", Fp_old: "np.ndarray", epcum: "np.ndarray"):
        """Convenience method for integration with ContactPotato_NGSolve.py.

        Parameters
        ----------
        F       : (N_gp, 9) or (N_gp*9,)
        Fp_old  : (N_gp, 9) or (N_gp*9,)
        epcum   : (N_gp,)

        Returns
        -------
        Fp_new  : (N_gp*9,)   — flattened for direct write to _Fp_conv
        delta_ep: (N_gp,)
        """
        import numpy as np

        F = F.reshape(-1, 9)
        Fp_old = Fp_old.reshape(-1, 9)
        epcum = epcum.reshape(-1, 1)

        x = torch.from_numpy(
            np.hstack([F, Fp_old, epcum]).astype(np.float32)
        )

        with torch.no_grad():
            out = self.forward(x)

        return (
            out["Fp_new"].numpy().ravel(),
            out["delta_ep"].numpy(),
        )

    @classmethod
    def from_config(cls, cfg: ReturnMappingConfig) -> "ReturnMappingNet":
        return cls(cfg)

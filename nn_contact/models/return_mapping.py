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

        # Pre-allocated buffer for predict_numpy (lazy-sized on first call)
        self._input_buf: torch.Tensor | None = None

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

    def _forward_Fp_only(self, F_flat: torch.Tensor,
                          Fp_old_flat: torch.Tensor,
                          epcum_scalar: torch.Tensor) -> torch.Tensor:
        """Functional forward returning only Fp_new(9) for Jacobian computation.

        All inputs are for a SINGLE GP (no batch dim) to work with vmap.
        """
        x = torch.cat([F_flat, Fp_old_flat, epcum_scalar])  # (19,)
        features = self.backbone(x.unsqueeze(0))  # (1, hidden)
        Fp_correction = self.fp_head(features).squeeze(0)  # (9,)
        return Fp_old_flat + Fp_correction  # (9,)

    def compute_jacobian_dFp_dF(
        self,
        F: torch.Tensor,
        Fp_old: torch.Tensor,
        epcum: torch.Tensor,
    ) -> torch.Tensor:
        """Compute batched Jacobian dFp_new/dF via autodiff.

        Uses torch.func.vmap + jacrev for efficient batched Jacobian:
        one call computes the full (N, 9, 9) Jacobian tensor.

        This replaces 9 × FD perturbations per GP in the classical
        consistent tangent computation.

        Parameters
        ----------
        F      : (N, 9) — deformation gradient at yielding GPs
        Fp_old : (N, 9) — previous converged Fp
        epcum  : (N,)   — cumulative plastic strain

        Returns
        -------
        dFp_dF : (N, 9, 9) — Jacobian dFp_new[i]/dF[j] per GP
        """
        from torch.func import vmap, jacrev

        epcum_1d = epcum.unsqueeze(-1)  # (N, 1)

        # jacrev of _forward_Fp_only w.r.t. arg 0 (F_flat)
        jac_fn = jacrev(self._forward_Fp_only, argnums=0)

        # vmap over the batch dimension
        batched_jac = vmap(jac_fn)(F, Fp_old, epcum_1d)  # (N, 9, 9)

        return batched_jac

    def compute_jacobian_numpy(
        self,
        F: "np.ndarray",
        Fp_old: "np.ndarray",
        epcum: "np.ndarray",
    ) -> "np.ndarray":
        """Compute dFp_new/dF as numpy array (convenience for FEM integration).

        Parameters
        ----------
        F      : (N, 9) or (N*9,)
        Fp_old : (N, 9) or (N*9,)
        epcum  : (N,)

        Returns
        -------
        dFp_dF : (N, 9, 9) numpy array
        """
        import numpy as np

        F_t = torch.from_numpy(F.reshape(-1, 9).astype(np.float32))
        Fp_old_t = torch.from_numpy(Fp_old.reshape(-1, 9).astype(np.float32))
        epcum_t = torch.from_numpy(epcum.astype(np.float32))

        dFp_dF = self.compute_jacobian_dFp_dF(F_t, Fp_old_t, epcum_t)
        return dFp_dF.detach().numpy()

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

        F_np = F.reshape(-1, 9)
        Fp_np = Fp_old.reshape(-1, 9)
        n = F_np.shape[0]

        # Reuse pre-allocated buffer if same batch size
        if self._input_buf is None or self._input_buf.shape[0] != n:
            self._input_buf = torch.empty(n, 19, dtype=torch.float32)

        buf = self._input_buf
        buf[:, :9] = torch.from_numpy(F_np.astype(np.float32))
        buf[:, 9:18] = torch.from_numpy(Fp_np.astype(np.float32))
        buf[:, 18:] = torch.from_numpy(
            epcum.reshape(-1, 1).astype(np.float32))

        with torch.no_grad():
            out = self.forward(buf)

        return (
            out["Fp_new"].numpy().ravel(),
            out["delta_ep"].numpy(),
        )

    def predict_with_jacobian_numpy(
        self,
        F: "np.ndarray",
        Fp_old: "np.ndarray",
        epcum: "np.ndarray",
    ):
        """Predict Fp_new, delta_ep AND dFp/dF in one call.

        For the Newton solver: return mapping + consistent tangent data.

        Returns
        -------
        Fp_new   : (N*9,) flattened
        delta_ep : (N,)
        dFp_dF   : (N, 9, 9) — only for yielding GPs (delta_ep > 0),
                    None entries for elastic GPs
        """
        import numpy as np

        F_np = F.reshape(-1, 9).astype(np.float32)
        Fp_old_np = Fp_old.reshape(-1, 9).astype(np.float32)
        epcum_np = epcum.astype(np.float32)
        N = F_np.shape[0]

        # Forward pass (no grad needed for Fp_new, delta_ep)
        x = torch.from_numpy(
            np.hstack([F_np, Fp_old_np, epcum_np.reshape(-1, 1)]))

        with torch.no_grad():
            out = self.forward(x)

        Fp_new_np = out["Fp_new"].numpy()
        delta_ep_np = out["delta_ep"].numpy()

        # Identify yielding GPs
        yield_mask = delta_ep_np > 1e-10
        n_yield = yield_mask.sum()

        dFp_dF = np.zeros((N, 9, 9), dtype=np.float32)

        if n_yield > 0:
            # Compute Jacobian only for yielding GPs (saves compute)
            F_yield = torch.from_numpy(F_np[yield_mask])
            Fp_old_yield = torch.from_numpy(Fp_old_np[yield_mask])
            epcum_yield = torch.from_numpy(epcum_np[yield_mask])

            jac = self.compute_jacobian_dFp_dF(
                F_yield, Fp_old_yield, epcum_yield)
            dFp_dF[yield_mask] = jac.detach().numpy()

        return Fp_new_np.ravel(), delta_ep_np, dFp_dF

    @classmethod
    def from_config(cls, cfg: ReturnMappingConfig) -> "ReturnMappingNet":
        return cls(cfg)

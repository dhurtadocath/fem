"""Integration hooks for ContactPotato_NGSolve.py.

Provides drop-in replacements for the contact detection pipeline:

1. HybridCDA: NN broad phase + Newton refinement (Phase 1)
2. NeuralPullCDA: Full NN replacement for SDF + derivatives (Phase 2)
3. NeuralReturnMapping: NN surrogate for J2 return mapping (Phase 3)

Each class wraps a trained model and provides numpy-in/numpy-out interfaces
matching the existing ContactCache API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nn_contact.data.normalization import CoordinateNormalizer


class HybridCDA:
    """Multi-task NN for broad phase, with C++ Newton refinement.

    Replaces:  KD-tree → candidate selection → TR projection
    With:      NN(x,y,z) → (g_filter, patch_id, xi_init)

    Newton refinement (narrow phase 2) is kept for machine-precision accuracy.
    """

    def __init__(
        self,
        model_path: str | Path,
        normalizer: CoordinateNormalizer,
        gn_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        topk: int = 1,
        device: str = "cuda",
    ):
        self.device = device
        self.normalizer = normalizer
        self.gn_threshold = gn_threshold
        self.confidence_threshold = confidence_threshold
        self.topk = topk

        # Load model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        from nn_contact.models.multitask import MultiTaskContactNet
        from nn_contact.config import MultiTaskConfig

        # Reconstruct model from checkpoint
        cfg = checkpoint.get("config", MultiTaskConfig())
        self.model = MultiTaskContactNet.from_config(cfg).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, xyz: np.ndarray) -> dict[str, np.ndarray]:
        """Predict contact quantities for slave node positions.

        Parameters
        ----------
        xyz : (N, 3) — current slave node positions.

        Returns
        -------
        dict with:
            active_mask : (N,) bool — True if |g| < threshold
            patch_ids   : (N,) int  — predicted closest patch (-1 if inactive)
            xi_init     : (N, 2)    — initial guess for parametric coords
            gn_approx   : (N,)      — approximate signed distance
        """
        N = xyz.shape[0]
        xyz_norm = self.normalizer.transform(xyz).astype(np.float32)
        xyz_t = torch.from_numpy(xyz_norm).to(self.device)

        out = self.model.predict(xyz_t, topk=self.topk)

        gn = out["gn"].cpu().numpy()
        patch_ids = out["patch_ids"][:, 0].cpu().numpy()  # top-1
        xi = out["xi"][:, 0, :].cpu().numpy()             # top-1 xi
        probs = out["patch_probs"][:, 0].cpu().numpy()    # top-1 confidence

        # Active mask: close to surface AND confident
        active = (np.abs(gn) < self.gn_threshold) & (probs > self.confidence_threshold)

        # Inactive nodes: set to -1
        result_patch = np.full(N, -1, dtype=np.int32)
        result_xi = np.zeros((N, 2), dtype=np.float64)
        result_patch[active] = patch_ids[active]
        result_xi[active] = xi[active]

        return {
            "active_mask": active,
            "patch_ids": result_patch,
            "xi_init": result_xi,
            "gn_approx": gn,
        }


class NeuralPullCDA:
    """Full CDA replacement using Neural-Pull SDF network.

    Replaces the entire projection pipeline with NN(x,y,z) -> (g, n, dn/dx_s).
    Only viable if derivative accuracy is simulation-grade.
    """

    def __init__(
        self,
        model_path: str | Path,
        normalizer: CoordinateNormalizer,
        device: str = "cuda",
    ):
        self.device = device
        self.normalizer = normalizer

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        from nn_contact.models.neural_pull import NeuralPullNet
        from nn_contact.config import NeuralPullConfig

        cfg = checkpoint.get("config", NeuralPullConfig())
        self.model = NeuralPullNet.from_config(cfg).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def evaluate(
        self, xyz: np.ndarray, compute_hessian: bool = True
    ) -> dict[str, np.ndarray]:
        """Evaluate contact quantities for slave node positions.

        Parameters
        ----------
        xyz : (N, 3)

        Returns
        -------
        dict with:
            gn      : (N,)     — signed distance
            normals : (N, 3)   — surface normal
            dndxs   : (N, 3, 3) — dn/dx_s (if compute_hessian)
        """
        xyz_norm = self.normalizer.transform(xyz).astype(np.float32)
        xyz_t = torch.from_numpy(xyz_norm).to(self.device).requires_grad_(True)

        with torch.enable_grad():
            out = self.model.predict(xyz_t, compute_hessian=compute_hessian)

        result = {
            "gn": out["gn"].cpu().numpy(),
            "normals": out["normal"].cpu().numpy(),
        }
        if compute_hessian and "dndxs" in out:
            result["dndxs"] = out["dndxs"].cpu().numpy()
        return result


class NeuralReturnMapping:
    """NN surrogate for J2 return mapping (Phase 3).

    Drop-in replacement for the iterative return_mapping() function.
    """

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        self.device = device

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        from nn_contact.models.return_mapping import ReturnMappingNet
        from nn_contact.config import ReturnMappingConfig

        cfg = checkpoint.get("config", ReturnMappingConfig())
        self.model = ReturnMappingNet.from_config(cfg).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def __call__(
        self,
        F_flat: np.ndarray,
        Fp_conv: np.ndarray,
        epcum_conv: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Replace iterative return mapping.

        Parameters
        ----------
        F_flat    : (N_gp*9,) — current deformation gradient
        Fp_conv   : (N_gp*9,) — converged plastic deformation gradient
        epcum_conv: (N_gp,)   — cumulative plastic strain

        Returns
        -------
        Fp_new    : (N_gp*9,)
        delta_ep  : (N_gp,)
        success   : bool (always True for NN — no convergence failure)
        """
        Fp_new, delta_ep = self.model.predict_numpy(F_flat, Fp_conv, epcum_conv)
        return Fp_new, delta_ep, True

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
        topk: int = 3,
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

    @classmethod
    def from_variant(
        cls,
        variant: str = "v1",
        device: str = "cuda",
        gn_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
    ) -> "HybridCDA":
        """Load a trained model by variant name (v1, v2, v2b, v3).

        Expects checkpoints at nn_contact/checkpoints/multitask_{variant}/
        with best_model.pt and config.pt files.
        """
        ckpt_dir = Path(f"nn_contact/checkpoints/multitask_{variant}")
        model_path = ckpt_dir / "best_model.pt"
        config_path = ckpt_dir / "config.pt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No checkpoint for variant '{variant}' at {model_path}. "
                f"Train with: python3 nn_contact/scripts/train_multitask.py --variant {variant}"
            )

        # Load normalizer from config.pt
        normalizer = CoordinateNormalizer()
        if config_path.exists():
            config_data = torch.load(config_path, map_location="cpu", weights_only=False)
            if "normalizer" in config_data:
                normalizer.load_state_dict(config_data["normalizer"])

        return cls(
            model_path=model_path,
            normalizer=normalizer,
            gn_threshold=gn_threshold,
            confidence_threshold=confidence_threshold,
            device=device,
        )

    @torch.no_grad()
    def predict(self, xyz: np.ndarray) -> dict[str, np.ndarray]:
        """Predict contact quantities with top-K candidates per node.

        Parameters
        ----------
        xyz : (N, 3) — current slave node positions (unnormalized).

        Returns
        -------
        dict with:
            active_mask : (N,) bool        — True if |g| < threshold and confident
            patch_ids   : (N, K) int32     — top-K patch candidates (-1 if inactive)
            xi_init     : (N, K, 2) float64 — parametric coords per candidate
            gn_approx   : (N,) float64     — approximate signed distance (from NN)
            patch_probs : (N, K) float64   — softmax probabilities per candidate
        """
        N = xyz.shape[0]
        K = self.topk
        xyz_norm = self.normalizer.transform(xyz).astype(np.float32)
        xyz_t = torch.from_numpy(xyz_norm).to(self.device)

        out = self.model.predict(xyz_t, topk=K)

        gn = out["gn"].cpu().numpy()
        patch_ids = out["patch_ids"].cpu().numpy().astype(np.int32)  # (N, K)
        xi = out["xi"].cpu().numpy().astype(np.float64)              # (N, K, 2)
        probs = out["patch_probs"].cpu().numpy()                     # (N, K)

        # Active mask: close to surface AND top-1 confident
        active = (np.abs(gn) < self.gn_threshold) & (probs[:, 0] > self.confidence_threshold)

        # Zero out inactive nodes in-place (avoid extra allocation)
        inactive = ~active
        if inactive.any():
            patch_ids[inactive] = -1
            xi[inactive] = 0.0

        return {
            "active_mask": active,
            "patch_ids": patch_ids,
            "xi_init": xi,
            "gn_approx": gn,
            "patch_probs": probs,
        }


class NeuralPullCDA:
    """Full CDA replacement using Neural-Pull SDF network.

    Replaces the entire projection pipeline with NN(x,y,z) -> (g, n, dn/dx_s).

    The network operates in normalized coordinates:
      g_nn = g_phys / L,  ∇g_nn = n,  ∇²g_nn = L * dn/dx_raw
    This class handles the denormalization back to physical units.
    """

    def __init__(
        self,
        model_path: str | Path,
        normalizer: CoordinateNormalizer,
        char_length: float = 8.0,
        device: str = "cuda",
    ):
        self.device = device
        self.normalizer = normalizer
        self.char_length = char_length

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        from nn_contact.models.neural_pull import NeuralPullNet
        from nn_contact.config import NeuralPullConfig

        cfg = checkpoint.get("config", NeuralPullConfig())
        self.model = NeuralPullNet.from_config(cfg).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        # Use char_length from checkpoint if available
        if "char_length" in checkpoint:
            self.char_length = checkpoint["char_length"]

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str | Path = "nn_contact/checkpoints/neural_pull",
        device: str = "cuda",
    ) -> "NeuralPullCDA":
        """Load from checkpoint directory."""
        ckpt_dir = Path(checkpoint_dir)
        model_path = ckpt_dir / "best_model.pt"
        config_path = ckpt_dir / "config.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"No Neural-Pull checkpoint at {model_path}")

        # Load normalizer
        normalizer = CoordinateNormalizer()
        char_length = 8.0
        if config_path.exists():
            config_data = torch.load(config_path, map_location="cpu", weights_only=False)
            if "normalizer" in config_data:
                normalizer = CoordinateNormalizer.from_state_dict(config_data["normalizer"])
            if "data_config" in config_data:
                char_length = config_data["data_config"].char_length

        return cls(
            model_path=model_path,
            normalizer=normalizer,
            char_length=char_length,
            device=device,
        )

    def evaluate(
        self, xyz: np.ndarray, compute_hessian: bool = True
    ) -> dict[str, np.ndarray]:
        """Evaluate contact quantities for slave node positions.

        Parameters
        ----------
        xyz : (N, 3) — slave positions in physical coordinates.

        Returns
        -------
        dict with:
            gn      : (N,)      — signed distance (physical units)
            normals : (N, 3)    — unit surface normal
            xc_surf : (N, 3)    — approximate surface point: xs - gn * n
            dndxs   : (N, 3, 3) — dn/dx_s in physical coords (if compute_hessian)
        """
        L = self.char_length
        xyz_norm = self.normalizer.transform(xyz).astype(np.float32)
        xyz_t = torch.from_numpy(xyz_norm).to(self.device).requires_grad_(True)

        with torch.enable_grad():
            out = self.model.predict(xyz_t, compute_hessian=compute_hessian)

        # Denormalize: g_phys = g_nn * L, normals = ∇g_nn (already unit)
        gn = out["gn"].cpu().numpy() * L
        normals = out["normal"].cpu().numpy()

        # Normalize normals to unit length (eikonal is near-perfect but not exact)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normals = normals / norms

        # Surface point approximation: xc ≈ xs - g * n
        xc_surf = xyz - gn[:, None] * normals

        result = {
            "gn": gn,
            "normals": normals,
            "xc_surf": xc_surf,
        }

        if compute_hessian and "dndxs" in out:
            # Denormalize: ∇²g_nn = L * dn/dx_raw → dn/dx_raw = hess / L
            result["dndxs"] = out["dndxs"].cpu().numpy() / L

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

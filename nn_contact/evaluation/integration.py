"""Integration hooks for ContactPotato_NGSolve.py.

Provides drop-in replacements for the contact detection pipeline:

1. HybridCDA: NN broad phase + Newton refinement (Phase 1)
2. NeuralPullCDA: Full NN replacement for SDF + derivatives (Phase 2)
3. NeuralReturnMapping: NN surrogate for J2 return mapping (Phase 3)
4. GNNNewtonIntegration: GNN Newton step predictor warm-start (Phase 4)

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
        coord_offset: np.ndarray | None = None,
    ):
        self.device = device
        self.normalizer = normalizer
        self.gn_threshold = gn_threshold
        self.confidence_threshold = confidence_threshold
        self.topk = topk
        self.coord_offset = np.asarray(coord_offset, dtype=np.float64) if coord_offset is not None else None

        # Load model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        from nn_contact.models.multitask import MultiTaskContactNet
        from nn_contact.config import MultiTaskConfig

        # Reconstruct model from checkpoint, detecting features from state_dict
        cfg = checkpoint.get("config", MultiTaskConfig())
        state = checkpoint["model_state"]
        # Detect task_attention and patch_conditioned from state_dict keys
        # (some checkpoints don't store these flags in config)
        has_attn = any(k.startswith("attn_gn.") for k in state)
        has_pcond = any(k.startswith("proj_head_cond.") for k in state)
        self.model = MultiTaskContactNet.from_config(
            cfg, task_attention=has_attn, patch_conditioned=has_pcond,
        ).to(device)
        self.model.load_state_dict(state)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str | Path | None = None,
        device: str = "cuda",
    ) -> "HybridCDA":
        """Load from a checkpoint directory containing best_model.pt [+ config.pt].

        Parameters
        ----------
        checkpoint_dir : path to directory with best_model.pt and optionally config.pt.
            None defaults to nn_contact/checkpoints/external/mt_sweep_unc_wt/.
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "external" / "mt_sweep_unc_wt"
        ckpt_dir = Path(checkpoint_dir)
        model_path = ckpt_dir / "best_model.pt"
        config_path = ckpt_dir / "config.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"No multitask checkpoint at {model_path}")

        normalizer = CoordinateNormalizer()
        if config_path.exists():
            config_data = torch.load(config_path, map_location="cpu", weights_only=False)
            if "normalizer" in config_data:
                normalizer = CoordinateNormalizer.from_state_dict(config_data["normalizer"])

        return cls(
            model_path=model_path,
            normalizer=normalizer,
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
        xyz_shifted = xyz + self.coord_offset if self.coord_offset is not None else xyz
        xyz_norm = self.normalizer.transform(xyz_shifted).astype(np.float32)
        xyz_t = torch.from_numpy(xyz_norm).to(self.device)

        out = self.model.predict(xyz_t, topk=K)

        gn = out["gn"].cpu().numpy()
        patch_ids = out["patch_ids"].cpu().numpy().astype(np.int32)  # (N, K)
        xi = out["xi"].cpu().numpy().astype(np.float64)              # (N, K, 2)
        probs = out["patch_probs"].cpu().numpy()                     # (N, K)

        confident = probs[:, 0] > self.confidence_threshold
        near_surface = np.abs(gn) < self.gn_threshold

        # NN-active: confident AND near surface → use NN patch + C++ refinement
        active = confident & near_surface

        # Low-confidence: not confident → need classical TR fallback
        # Far-field: confident but far → skip entirely (gn >> 0, no contact)
        needs_fallback = ~confident

        # Zero out inactive nodes in-place
        inactive = ~active
        if inactive.any():
            patch_ids[inactive] = -1
            xi[inactive] = 0.0

        return {
            "active_mask": active,
            "needs_fallback": needs_fallback,
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
        coord_offset: np.ndarray | None = None,
    ):
        self.device = device
        self.normalizer = normalizer
        self.char_length = char_length
        self.coord_offset = np.asarray(coord_offset, dtype=np.float64) if coord_offset is not None else None

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
        cls, checkpoint_dir: str | Path | None = None,
        device: str = "cuda",
    ) -> "NeuralPullCDA":
        """Load from checkpoint directory.

        Parameters
        ----------
        checkpoint_dir : path to directory with best_model.pt and optionally config.pt.
            None defaults to nn_contact/checkpoints/external/neural_pull_v1/.
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "external" / "neural_pull_v1"
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
        xyz_shifted = xyz + self.coord_offset if self.coord_offset is not None else xyz
        xyz_norm = self.normalizer.transform(xyz_shifted).astype(np.float32)
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
        # Support both old ("model_state") and new ("model_state_dict") key names
        state_key = "model_state_dict" if "model_state_dict" in checkpoint else "model_state"
        self.model.load_state_dict(checkpoint[state_key])
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


class GNNNewtonIntegration:
    """GNN Newton step predictor for warm-starting iteration 0 (Phase 4).

    Predicts the first Newton displacement increment Δu from the current
    FEM state (displacement, residual, contact). Replaces the expensive
    tangent assembly + linear solve at iteration 0 with a ~3ms GNN forward.

    Parameters
    ----------
    checkpoint_path : path to best.pt checkpoint
    mesh : ngsolve.Mesh — hex mesh
    n : int — mesh density (elements per side)
    X_ref : (nv, 3) — reference vertex coordinates
    slave_verts : (n_slave,) int — contact surface vertex indices
    top_verts : (n_top,) int — Dirichlet boundary vertex indices
    free_dofs : (n_free,) int — free DOF indices (block-sequential)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        mesh,
        n: int,
        X_ref: np.ndarray,
        slave_verts: np.ndarray,
        top_verts: np.ndarray,
        free_dofs: np.ndarray,
    ):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No GNN Newton checkpoint at {checkpoint_path}"
            )

        # Load model (auto-detect MPN vs GCN)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_type = ckpt.get("model_type", "mpn")
        if model_type == "gcn":
            from nn_contact.models.gnn_newton import GCNNewtonPredictor
            self.model = GCNNewtonPredictor.from_checkpoint(str(checkpoint_path))
        else:
            from nn_contact.models.gnn_newton import GNNNewtonPredictor
            self.model = GNNNewtonPredictor.from_checkpoint(str(checkpoint_path))

        # Constants
        self.nv = len(X_ref)
        self.n = n
        self.L_char = 4.0
        self.h_elem = 4.0 / n
        self.slave_verts = slave_verts
        self.top_verts = top_verts

        # Precompute normalized reference coords [0,1]³
        x_min = X_ref.min(axis=0)
        x_max = X_ref.max(axis=0)
        self.x_ref_norm = (
            (X_ref - x_min) / (x_max - x_min + 1e-10)
        ).astype(np.float32)

        # Precompute Dirichlet mask
        self._is_dirichlet = np.zeros(self.nv, dtype=np.float32)
        self._is_dirichlet[top_verts] = 1.0

        # Build static graph (once)
        self.edge_index, self.edge_attr = self._build_graph(mesh, X_ref)

        print(f"  GNN Newton: loaded {model_type} model "
              f"({sum(p.numel() for p in self.model.parameters()):,} params), "
              f"{self.edge_index.shape[1]} edges")

    def _build_graph(self, mesh, X_ref):
        """Build bidirectional edge_index and normalized edge_attr from hex mesh."""
        from ngsolve import VOL

        # Extract unique edges from hex elements
        edges = set()
        for el in mesh.Elements(VOL):
            verts = [v.nr for v in el.vertices]
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    a, b = min(verts[i], verts[j]), max(verts[i], verts[j])
                    edges.add((a, b))
        edge_array = np.array(sorted(edges), dtype=np.int32)  # (n_edges, 2)

        # Edge features: dx_ref and ||dx_ref||
        dx_ref = X_ref[edge_array[:, 1]] - X_ref[edge_array[:, 0]]
        edge_len = np.linalg.norm(dx_ref, axis=1, keepdims=True)
        edge_feat = np.hstack([dx_ref, edge_len]).astype(np.float32)

        # Bidirectional edge_index
        fwd = edge_array.T
        bwd = edge_array[:, ::-1].T
        edge_index = torch.tensor(
            np.concatenate([fwd, bwd], axis=1), dtype=torch.long
        )

        # Bidirectional edge_attr (negate direction for reverse, keep length)
        fwd_attr = edge_feat.copy()
        bwd_attr = edge_feat.copy()
        bwd_attr[:, :3] = -bwd_attr[:, :3]
        edge_attr = torch.tensor(
            np.concatenate([fwd_attr, bwd_attr], axis=0), dtype=torch.float32
        )

        # Normalize by element size (matching training pipeline)
        edge_attr[:, :3] /= self.h_elem
        edge_attr[:, 3] /= self.h_elem

        return edge_index, edge_attr

    def predict_step(
        self,
        u_vec: np.ndarray,
        r_vec: np.ndarray,
        contact_state: tuple,
        load_frac: float,
    ) -> np.ndarray:
        """Predict Newton Δu from current FEM state.

        Parameters
        ----------
        u_vec : (3*nv,) — current displacement (block-sequential)
        r_vec : (3*nv,) — current residual (block-sequential)
        contact_state : (gn, normals, active) from contact cache
            gn : (n_slave,), normals : (n_slave, 3), active : (n_slave,) bool
        load_frac : float in [0, 1]

        Returns
        -------
        delta_u : (3*nv,) — predicted Newton increment (block-sequential)
        """
        nv = self.nv
        node_feat = self._build_features(u_vec, r_vec, contact_state, load_frac)
        du_node = self.model.predict_numpy(
            node_feat, self.edge_index, self.edge_attr
        )  # (nv, 3)

        # (nv, 3) per-node → (3*nv,) block-sequential
        return np.concatenate([du_node[:, 0], du_node[:, 1], du_node[:, 2]])

    def _build_features(
        self,
        u_vec: np.ndarray,
        r_vec: np.ndarray,
        contact_state: tuple,
        load_frac: float,
    ) -> np.ndarray:
        """Build (nv, 17) node features matching training pipeline exactly."""
        nv = self.nv
        gn, normals, active = contact_state

        # 1. Displacement (3): block-sequential → per-node, normalize by L_char
        u_node = np.stack([
            u_vec[:nv], u_vec[nv:2*nv], u_vec[2*nv:3*nv]
        ], axis=1).astype(np.float32) / self.L_char

        # 2. Residual (3): per-sample global normalization
        r_node = np.stack([
            r_vec[:nv], r_vec[nv:2*nv], r_vec[2*nv:3*nv]
        ], axis=1).astype(np.float32)
        r_max = np.linalg.norm(r_node, axis=1).max()
        if r_max > 1e-30:
            r_node /= r_max

        # 3. Reference coords (3): precomputed
        # 4-6. Contact features: scatter from slave verts
        contact_flag = np.zeros(nv, dtype=np.float32)
        gap_node = np.zeros(nv, dtype=np.float32)
        normal_node = np.zeros((nv, 3), dtype=np.float32)
        for i, sv in enumerate(self.slave_verts):
            if active[i]:
                contact_flag[sv] = 1.0
                gap_node[sv] = gn[i] / self.h_elem
                normal_node[sv] = normals[i]

        # 7. Load fraction (broadcast)
        load_node = np.full((nv, 1), load_frac, dtype=np.float32)

        # 8. Dirichlet flag (precomputed)
        # 9. BC value: displacement magnitude at top verts
        bc_value = np.zeros(nv, dtype=np.float32)
        tv = self.top_verts
        bc_value[tv] = np.sqrt(
            u_vec[tv]**2 + u_vec[tv + nv]**2 + u_vec[tv + 2*nv]**2
        ).astype(np.float32)
        bc_value /= self.L_char

        # Concatenate: (nv, 17)
        return np.hstack([
            u_node,                                  # 0-2
            r_node,                                  # 3-5
            self.x_ref_norm,                         # 6-8
            contact_flag.reshape(-1, 1),             # 9
            gap_node.reshape(-1, 1),                 # 10
            normal_node,                             # 11-13
            load_node,                               # 14
            self._is_dirichlet.reshape(-1, 1),       # 15
            bc_value.reshape(-1, 1),                 # 16
        ]).astype(np.float32)

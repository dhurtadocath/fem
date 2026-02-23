"""PyG dataset for GNN Newton step prediction training data.

Loads .npz files produced by collect_gnn_newton_data.py and converts them
to torch_geometric Data objects with proper node/edge features and targets.

Node features (17 dims):
    u_i         (3) — current displacement (normalized by L_char)
    r_i         (3) — residual (normalized by max||r|| per sample)
    x_ref_i     (3) — reference coordinates (normalized to [0,1]^3)
    contact_flag(1) — binary: node in contact
    gap         (1) — signed gap (normalized by element size)
    normal      (3) — contact normal (0 for non-contact)
    load_frac   (1) — load fraction t/T, broadcast to all nodes
    is_dirichlet(1) — binary: node has prescribed DOFs
    bc_value    (1) — prescribed displacement magnitude (normalized)

Edge features (4 dims, static):
    dx_ref      (3) — relative reference position
    ||dx_ref||  (1) — edge length

Target: delta_u (3*nv,) — first Newton increment (only free DOFs supervised).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


def build_bidirectional_edge_index(edge_array: np.ndarray) -> torch.Tensor:
    """Convert (n_edges, 2) undirected edges to (2, 2*n_edges) bidirectional.

    Parameters
    ----------
    edge_array : (n_edges, 2) int32 — undirected edge list

    Returns
    -------
    edge_index : (2, 2*n_edges) long tensor — bidirectional for PyG
    """
    fwd = edge_array.T  # (2, n_edges)
    bwd = edge_array[:, ::-1].T  # (2, n_edges) reversed
    return torch.tensor(np.concatenate([fwd, bwd], axis=1), dtype=torch.long)


def build_bidirectional_edge_attr(
    edge_features: np.ndarray,
) -> torch.Tensor:
    """Duplicate edge features for bidirectional edges.

    For reverse edges, negate the direction vector but keep the length.

    Parameters
    ----------
    edge_features : (n_edges, 4) — [dx(3), ||dx||(1)]

    Returns
    -------
    edge_attr : (2*n_edges, 4) float32 tensor
    """
    fwd = edge_features.copy()
    bwd = edge_features.copy()
    bwd[:, :3] = -bwd[:, :3]  # reverse direction
    return torch.tensor(
        np.concatenate([fwd, bwd], axis=0), dtype=torch.float32
    )


class GNNNewtonDataset(InMemoryDataset):
    """PyG InMemoryDataset for GNN Newton step prediction.

    Loads one or more .npz files from the raw directory and converts them
    to PyG Data objects.

    Parameters
    ----------
    root : str
        Root directory. Processed data cached under root/processed/.
    npz_files : list[str | Path] | None
        Explicit list of .npz files to load. If None, loads all .npz files
        found in root/raw/.
    normalize : bool
        Apply feature normalization (recommended).
    """

    def __init__(
        self,
        root: str,
        npz_files: list[str | Path] | None = None,
        normalize: bool = True,
        transform=None,
        pre_transform=None,
    ):
        self._npz_files = npz_files
        self._normalize = normalize
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        if self._npz_files:
            return [Path(f).name for f in self._npz_files]
        raw_dir = Path(self.raw_dir)
        if raw_dir.exists():
            return [f.name for f in sorted(raw_dir.glob("*.npz"))]
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    def process(self):
        data_list = []

        # Determine which npz files to load
        if self._npz_files:
            npz_paths = [Path(f) for f in self._npz_files]
        else:
            npz_paths = sorted(Path(self.raw_dir).glob("*.npz"))

        for npz_path in npz_paths:
            print(f"Loading {npz_path.name}...")
            data_list.extend(self._process_npz(npz_path))

        print(f"Total samples: {len(data_list)}")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])

    def _process_npz(self, npz_path: Path) -> list[Data]:
        """Convert a single .npz file to a list of PyG Data objects."""
        d = np.load(npz_path, allow_pickle=True)

        nv = int(d["nv"])
        n = int(d["n"])
        n_samples = int(d["n_samples"])
        X_ref = d["X_ref"]              # (nv, 3)
        edge_array = d["edge_index"]    # (n_edges, 2) undirected
        edge_feat = d["edge_features"]  # (n_edges, 4)
        free_dofs = d["free_dofs"]      # (n_free,)
        slave_verts = d["slave_verts"]  # (n_slave,)

        # Static graph structure (shared across samples)
        edge_index = build_bidirectional_edge_index(edge_array)
        edge_attr = build_bidirectional_edge_attr(edge_feat)

        # Normalize static features
        h_elem = 4.0 / n  # element edge length
        L_char = 4.0       # block half-extent ([-2,2]^3)

        # Reference coords: normalize to [0,1]^3
        x_ref_norm = (X_ref - X_ref.min(axis=0)) / (
            X_ref.max(axis=0) - X_ref.min(axis=0) + 1e-10
        )

        # Edge features: normalize lengths by element size
        edge_attr_norm = edge_attr.clone()
        edge_attr_norm[:, :3] /= h_elem  # dx_ref / h
        edge_attr_norm[:, 3] /= h_elem   # ||dx_ref|| / h

        # Free DOF mask — stored as (nv, 3) for correct PyG batching
        # Block-sequential: free_dofs[i<nv] → x-comp, [nv..2nv) → y, [2nv..3nv) → z
        free_mask_flat = np.zeros(3 * nv, dtype=bool)
        free_mask_flat[free_dofs] = True
        free_mask_node = np.stack([
            free_mask_flat[:nv],       # x-component free
            free_mask_flat[nv:2*nv],   # y-component free
            free_mask_flat[2*nv:3*nv], # z-component free
        ], axis=1)  # (nv, 3)

        # Slave vertex index map (vertex idx → slave index, -1 if not slave)
        slave_map = -np.ones(nv, dtype=np.int32)
        for i, sv in enumerate(slave_verts):
            slave_map[sv] = i

        data_list = []
        for idx in range(n_samples):
            u = d["u_current"][idx]        # (3*nv,)
            du = d["delta_u"][idx]         # (3*nv,)
            r = d["residual"][idx]         # (3*nv,)
            load_frac = float(d["load_fraction"][idx])
            is_dirichlet = d["is_dirichlet"][idx]  # (nv,)
            bc_value = d["bc_value"][idx]           # (nv,)
            contact_flag = d["contact_flag"][idx]   # (n_slave,)
            gap = d["gap"][idx]                     # (n_slave,)
            normal = d["normal"][idx]               # (n_slave, 3)

            # Build per-node features (nv, 17)
            # 1. Displacement (3): normalized by L_char
            u_x = u[:nv] / L_char
            u_y = u[nv:2*nv] / L_char
            u_z = u[2*nv:3*nv] / L_char
            u_node = np.stack([u_x, u_y, u_z], axis=1)  # (nv, 3)

            # 2. Residual (3): normalized by max||r||
            r_x = r[:nv]
            r_y = r[nv:2*nv]
            r_z = r[2*nv:3*nv]
            r_node = np.stack([r_x, r_y, r_z], axis=1)  # (nv, 3)
            if self._normalize:
                r_norm = np.linalg.norm(r_node, axis=1).max()
                if r_norm > 1e-30:
                    r_node /= r_norm

            # 3. Reference coords (3): already normalized
            # 4. Contact features: scatter slave data to full node array
            contact_flag_node = np.zeros(nv, dtype=np.float32)
            gap_node = np.zeros(nv, dtype=np.float32)
            normal_node = np.zeros((nv, 3), dtype=np.float32)
            for i, sv in enumerate(slave_verts):
                contact_flag_node[sv] = contact_flag[i]
                gap_node[sv] = gap[i] / h_elem if self._normalize else gap[i]
                normal_node[sv] = normal[i]

            # 5. Load fraction (1), broadcast
            load_node = np.full((nv, 1), load_frac, dtype=np.float32)

            # 6. Dirichlet info (2)
            dirichlet_node = is_dirichlet.reshape(-1, 1)
            bc_node = bc_value.reshape(-1, 1)
            if self._normalize:
                bc_node = bc_node / L_char

            # Concatenate all features: (nv, 17)
            node_feat = np.hstack([
                u_node,                          # 3
                r_node,                          # 3
                x_ref_norm,                      # 3
                contact_flag_node.reshape(-1, 1),# 1
                gap_node.reshape(-1, 1),         # 1
                normal_node,                     # 3
                load_node,                       # 1
                dirichlet_node,                  # 1
                bc_node,                         # 1
            ]).astype(np.float32)                 # total: 17

            # Target: du reshaped to (nv, 3) for per-node regression
            du_node = np.stack([
                du[:nv], du[nv:2*nv], du[2*nv:3*nv]
            ], axis=1).astype(np.float32)  # (nv, 3)

            data = Data(
                x=torch.tensor(node_feat),
                edge_index=edge_index,
                edge_attr=edge_attr_norm,
                y=torch.tensor(du_node),
                free_mask=torch.tensor(free_mask_node),
                load_fraction=torch.tensor([load_frac], dtype=torch.float32),
                # Metadata
                sim_n=torch.tensor([n], dtype=torch.int32),
                sim_E=torch.tensor([float(d["E_val"])], dtype=torch.float32),
                sim_nu=torch.tensor([float(d["nu_val"])], dtype=torch.float32),
            )
            data_list.append(data)

        return data_list


def split_by_simulation(
    dataset: GNNNewtonDataset,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split dataset indices ensuring temporal coherence.

    All steps from a single simulation parameter combination go to the
    same split (no leakage from consecutive load steps).

    Returns train_idx, val_idx, test_idx as lists of integer indices.
    """
    rng = np.random.RandomState(seed)
    n_total = len(dataset)

    # Group samples by simulation params (E, nu, n)
    groups: dict[tuple, list[int]] = {}
    for i in range(n_total):
        data = dataset[i]
        key = (
            int(data.sim_n.item()),
            float(data.sim_E.item()),
            float(data.sim_nu.item()),
        )
        groups.setdefault(key, []).append(i)

    # Shuffle group keys
    keys = list(groups.keys())
    rng.shuffle(keys)

    n_groups = len(keys)
    if n_groups <= 2:
        # Too few groups for proper split — fall back to sample-level split
        all_idx = list(range(n_total))
        rng.shuffle(all_idx)
        n_val = max(1, int(n_total * val_frac))
        n_test = max(1, int(n_total * test_frac))
        # Ensure train is non-empty
        n_val = min(n_val, n_total - 2)
        n_test = min(n_test, n_total - n_val - 1)
        val_idx = all_idx[:n_val]
        test_idx = all_idx[n_val:n_val + n_test]
        train_idx = all_idx[n_val + n_test:]
        return train_idx, val_idx, test_idx

    n_val = max(1, int(n_groups * val_frac))
    n_test = max(1, int(n_groups * test_frac))
    # Ensure train has at least 1 group
    while n_val + n_test >= n_groups:
        if n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1

    val_keys = keys[:n_val]
    test_keys = keys[n_val:n_val + n_test]
    train_keys = keys[n_val + n_test:]

    train_idx = [i for k in train_keys for i in groups[k]]
    val_idx = [i for k in val_keys for i in groups[k]]
    test_idx = [i for k in test_keys for i in groups[k]]

    return train_idx, val_idx, test_idx

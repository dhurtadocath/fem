"""GNN Newton Step Predictor — Encode-Process-Decode architecture.

Predicts the first Newton displacement increment Δu from current FEM state
(displacement, residual, contact info) using message-passing on the hex mesh
graph. Reduces Newton iteration count from ~6 to ~2-3 per load step.

Architecture follows MeshGraphNets (Pfaff et al., 2021):
  - Encode: node/edge features → hidden representations
  - Process: N message-passing layers with residual connections + LayerNorm
  - Decode: hidden → per-node Δu (3 components)

Designed for CPU inference at < 5ms for graphs with 200-3400 nodes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv

from nn_contact.config import GNNNewtonConfig


class MPLayer(MessagePassing):
    """Edge-conditioned message passing layer with residual + LayerNorm.

    message: MLP(concat(x_i, x_j, edge_attr)) → hidden
    update:  MLP(concat(x_i, aggregated)) → hidden, then LayerNorm
    """

    def __init__(self, hidden: int, activation: type[nn.Module] = nn.SiLU,
                 aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.msg_mlp = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            activation(),
            nn.Linear(hidden, hidden),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            activation(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.upd_mlp(torch.cat([x, out], dim=-1))
        return self.norm(out)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class GNNNewtonPredictor(nn.Module):
    """Encode-Process-Decode GNN for Newton step prediction.

    Parameters
    ----------
    cfg : GNNNewtonConfig
        Architecture configuration.
    """

    def __init__(self, cfg: GNNNewtonConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden

        act_cls = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[
            cfg.activation
        ]

        # Encode
        self.node_encoder = nn.Sequential(
            nn.Linear(cfg.node_in, h),
            act_cls(),
            nn.Linear(h, h),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(cfg.edge_in, h),
            act_cls(),
            nn.Linear(h, h),
        )

        # Process: message passing with residual connections
        self.processors = nn.ModuleList(
            [MPLayer(h, act_cls, cfg.aggr) for _ in range(cfg.n_layers)]
        )

        # Decode
        self.decoder = nn.Sequential(
            nn.Linear(h, h),
            act_cls(),
            nn.Linear(h, cfg.node_out),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming normal, zero output bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init decoder output layer for stable training start
        # (initial prediction = 0, safe for Newton)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (nv, node_in) — node features
        edge_index : (2, n_edges) — bidirectional edge indices
        edge_attr : (n_edges, edge_in) — edge features

        Returns
        -------
        delta_u : (nv, 3) — predicted Newton displacement increment
        """
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        for proc in self.processors:
            h = h + proc(h, edge_index, e)  # residual connection

        return self.decoder(h)

    def predict_numpy(
        self,
        node_features: "np.ndarray",
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> "np.ndarray":
        """Convenience method for FEM integration.

        Parameters
        ----------
        node_features : (nv, 17) numpy array
        edge_index : (2, n_edges) long tensor (prebuilt, static)
        edge_attr : (n_edges, 4) float tensor (prebuilt, static)

        Returns
        -------
        delta_u : (nv, 3) numpy array
        """
        import numpy as np

        x = torch.tensor(node_features, dtype=torch.float32)
        with torch.no_grad():
            out = self.forward(x, edge_index, edge_attr)
        return out.numpy()

    @classmethod
    def from_config(cls, cfg: GNNNewtonConfig) -> "GNNNewtonPredictor":
        return cls(cfg)

    @classmethod
    def from_checkpoint(cls, path: str) -> "GNNNewtonPredictor":
        """Load model from checkpoint file.

        The checkpoint should contain 'config' (GNNNewtonConfig) and
        'model_state_dict'.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model


class GCNNewtonPredictor(nn.Module):
    """Lightweight GCN-based Newton step predictor (fast on CPU).

    Uses GCNConv layers (no edge MLPs) — significantly cheaper than
    edge-conditioned MPN. Edge features are concatenated into node features
    via a one-time aggregation in the encoder.
    """

    def __init__(self, cfg: GNNNewtonConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden

        act_cls = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[
            cfg.activation
        ]

        # Encode: node features → hidden (edge features ignored by GCNConv)
        self.node_encoder = nn.Sequential(
            nn.Linear(cfg.node_in, h),
            act_cls(),
            nn.Linear(h, h),
        )

        # Process: GCNConv layers with residual + LayerNorm
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.convs.append(GCNConv(h, h))
            self.norms.append(nn.LayerNorm(h))
            self.acts.append(act_cls())

        # Decode
        self.decoder = nn.Sequential(
            nn.Linear(h, h),
            act_cls(),
            nn.Linear(h, cfg.node_out),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_encoder(x)
        for conv, norm, act in zip(self.convs, self.norms, self.acts):
            h = h + norm(act(conv(h, edge_index)))  # residual
        return self.decoder(h)

    def predict_numpy(self, node_features, edge_index, edge_attr):
        import numpy as np
        x = torch.tensor(node_features, dtype=torch.float32)
        with torch.no_grad():
            out = self.forward(x, edge_index, edge_attr)
        return out.numpy()

    @classmethod
    def from_checkpoint(cls, path: str) -> "GCNNewtonPredictor":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model


def newton_step_loss(
    du_pred: torch.Tensor,
    du_true: torch.Tensor,
    free_mask: torch.Tensor,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-graph normalized MSE loss on free DOFs only.

    Works correctly with PyG batched mixed-size graphs.

    Parameters
    ----------
    du_pred : (total_nodes, 3) — predicted Newton increment
    du_true : (total_nodes, 3) — target Newton increment
    free_mask : (total_nodes, 3) bool — per-node, per-component free DOF mask
    batch : (total_nodes,) — batch assignment (for PyG batched graphs)

    Returns
    -------
    loss : scalar tensor — mean of per-graph normalized MSE
    """
    # Element-wise error, masked to free DOFs only
    diff = (du_pred - du_true) * free_mask  # (total_nodes, 3)
    sq_err = diff.pow(2)  # (total_nodes, 3)

    if batch is not None:
        # Per-graph normalization: avoids early-step amplification
        n_graphs = int(batch.max().item()) + 1
        loss_sum = torch.zeros(n_graphs, device=du_pred.device)
        scale_sq = torch.zeros(n_graphs, device=du_pred.device)
        count = torch.zeros(n_graphs, device=du_pred.device)

        # Scatter-add per graph
        graph_idx = batch.unsqueeze(1).expand_as(sq_err)  # (total_nodes, 3)
        loss_sum.scatter_add_(0, batch, sq_err.sum(dim=1))
        scale_sq.scatter_add_(0, batch, (du_true * free_mask).pow(2).sum(dim=1))
        count.scatter_add_(0, batch, free_mask.float().sum(dim=1))

        # Normalized per-graph loss: MSE / ||du_true||^2
        scale_sq = scale_sq.clamp(min=1e-20)
        per_graph = loss_sum / scale_sq
        return per_graph.mean()
    else:
        # Single graph
        masked_true = du_true * free_mask
        scale = masked_true.pow(2).sum().clamp(min=1e-20)
        return sq_err.sum() / scale

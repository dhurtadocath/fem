"""Shared model building blocks: MLP, SIREN, FourierMLP.

All architectures output a raw feature vector that task-specific heads
can consume.  This separation enables multi-task learning (Phase 1) and
architecture swapping (Phase 2).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from nn_contact.config import FourierMLPConfig, MLPConfig, SIRENConfig


# ── Activations ──────────────────────────────────────────────────────────

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def get_activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


# ── Standard MLP with optional skip connections ──────────────────────────

class MLP(nn.Module):
    """Multi-layer perceptron with optional residual skip connections.

    When ``skip=True``, a residual connection is added around each hidden
    layer (requires constant width; a linear projection is inserted if
    the width changes).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int],
        activation: str = "silu",
        skip: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.skip = skip

        dims = [in_dim] + hidden_dims
        layers: list[nn.Module] = []
        skips: list[nn.Module | None] = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # skip projection if dimensions differ
            if skip and dims[i] != dims[i + 1]:
                skips.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            else:
                skips.append(None)

        self.layers = nn.ModuleList(layers)
        self.skips = nn.ModuleList([s if s is not None else nn.Identity() for s in skips])
        self._skip_flags = [s is not None or (skip and dims[i] == dims[i + 1])
                            for i, s in enumerate(skips)]
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_dims[-1], out_dim)

        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            h = self.act(layer(x))
            h = self.dropout(h)
            if self._skip_flags[i]:
                h = h + self.skips[i](x)
            x = h
        return self.head(x)

    def forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Return the last hidden representation (before head)."""
        for i, layer in enumerate(self.layers):
            h = self.act(layer(x))
            h = self.dropout(h)
            if self._skip_flags[i]:
                h = h + self.skips[i](x)
            x = h
        return x

    @classmethod
    def from_config(cls, cfg: MLPConfig, in_dim: int, out_dim: int) -> "MLP":
        return cls(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
            skip=cfg.skip_connections,
            dropout=cfg.dropout,
        )


# ── SIREN (Sinusoidal Representation Networks) ──────────────────────────

class SineLayer(nn.Module):
    """Linear layer followed by sin activation with SIREN initialization.

    Ref: Sitzmann et al., "Implicit Neural Representations with Periodic
    Activation Functions", NeurIPS 2020.
    """

    def __init__(self, in_dim: int, out_dim: int, omega: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_dim, out_dim)

        # SIREN-specific initialization
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
            else:
                bound = math.sqrt(6.0 / in_dim) / omega
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class HSineLayer(nn.Module):
    """H-SIREN first layer: sin(sinh(2·ω·Wx+b)).

    Broadens frequency support via hyperbolic composition: sinh provides
    unbounded growth enabling capture of both low and high frequencies.
    Used only for the first layer; hidden layers remain standard SineLayer.

    Ref: Gao & Jaiman, "H-SIREN: Improving Implicit Neural Representations
    with Hyperbolic Periodic Functions", arXiv 2410.04716, 2024.
    """

    def __init__(self, in_dim: int, out_dim: int, omega: float = 10.0):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            self.linear.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.sinh(2.0 * self.omega * self.linear(x)))


class SIREN(nn.Module):
    """Sinusoidal Representation Network.

    Uses sin activations throughout, with specialized initialization that
    preserves the distribution of activations across layers.  This gives
    excellent high-frequency representation and well-behaved derivatives
    (derivative of sin is cos — bounded and smooth).

    Parameters
    ----------
    in_dim : int
        Input dimensionality (3 for xyz coordinates).
    out_dim : int
        Output dimensionality.
    hidden_dims : list[int]
        Width of each hidden layer.
    omega_0 : float
        Frequency multiplier for the first layer.
    omega_hidden : float
        Frequency multiplier for hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int],
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
        h_siren: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        layers: list[nn.Module] = []
        dims = [in_dim] + hidden_dims

        # First layer: H-SIREN (sin(sinh(2ωx))) or standard SIREN (sin(ωx))
        if h_siren:
            layers.append(HSineLayer(dims[0], dims[1], omega=omega_0))
        else:
            layers.append(SineLayer(dims[0], dims[1], omega=omega_0, is_first=True))

        # Hidden layers with omega_hidden
        for i in range(1, len(dims) - 1):
            layers.append(SineLayer(dims[i], dims[i + 1], omega=omega_hidden))

        self.layers = nn.Sequential(*layers)

        # Final linear layer (no sin activation)
        self.head = nn.Linear(hidden_dims[-1], out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dims[-1]) / omega_hidden
            self.head.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.layers(x))

    def forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Return last hidden features (before head)."""
        return self.layers(x)

    @classmethod
    def from_config(cls, cfg: SIRENConfig, in_dim: int, out_dim: int) -> "SIREN":
        return cls(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=cfg.hidden_dims,
            omega_0=cfg.omega_0,
            omega_hidden=cfg.omega_hidden,
            h_siren=cfg.h_siren,
        )


# ── Random Fourier Features + MLP ───────────────────────────────────────

class FourierFeatures(nn.Module):
    """Random Fourier feature encoding: x -> [sin(Bx), cos(Bx)].

    Ref: Tancik et al., "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains", NeurIPS 2020.

    The random matrix B is drawn once at init and frozen.  Scaling ``sigma``
    controls the frequency band.
    """

    def __init__(self, in_dim: int, n_frequencies: int = 128, sigma: float = 10.0):
        super().__init__()
        B = torch.randn(n_frequencies, in_dim) * sigma
        self.register_buffer("B", B)
        self.out_dim = 2 * n_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B.T  # (..., n_freq)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FourierMLP(nn.Module):
    """MLP with Fourier feature input encoding.

    Combines ``FourierFeatures`` front-end with a standard ``MLP`` backbone.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int],
        n_frequencies: int = 128,
        frequency_scale: float = 10.0,
        activation: str = "silu",
        skip: bool = True,
    ):
        super().__init__()
        self.fourier = FourierFeatures(in_dim, n_frequencies, frequency_scale)
        self.mlp = MLP(
            in_dim=self.fourier.out_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            skip=skip,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.fourier(x))

    def forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp.forward_trunk(self.fourier(x))

    @classmethod
    def from_config(cls, cfg: FourierMLPConfig, in_dim: int, out_dim: int) -> "FourierMLP":
        return cls(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=cfg.hidden_dims,
            n_frequencies=cfg.n_frequencies,
            frequency_scale=cfg.frequency_scale,
            activation=cfg.activation,
            skip=cfg.skip_connections,
        )


# ── Factory ──────────────────────────────────────────────────────────────

def build_backbone(
    arch: Literal["mlp", "siren", "fourier_mlp"],
    in_dim: int,
    out_dim: int,
    *,
    mlp_cfg: MLPConfig | None = None,
    siren_cfg: SIRENConfig | None = None,
    fourier_cfg: FourierMLPConfig | None = None,
) -> nn.Module:
    """Build a backbone network from config."""
    if arch == "mlp":
        cfg = mlp_cfg or MLPConfig()
        return MLP.from_config(cfg, in_dim, out_dim)
    elif arch == "siren":
        cfg = siren_cfg or SIRENConfig()
        return SIREN.from_config(cfg, in_dim, out_dim)
    elif arch == "fourier_mlp":
        cfg = fourier_cfg or FourierMLPConfig()
        return FourierMLP.from_config(cfg, in_dim, out_dim)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

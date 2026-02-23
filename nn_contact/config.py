"""Configuration dataclasses for all nn_contact models and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Configuration for training data loading and preprocessing."""

    feather_path: str = "for_HPC/Points4Train_TR_LV.ft"

    # Gap-based filtering (as in paper Section 6)
    gn_min: float = -0.5
    gn_max: float = 1.5

    # Train/val split
    val_fraction: float = 0.2
    stratify_by_patch: bool = True

    # Normalization: divide xyz by characteristic length
    char_length: float = 8.0  # approximate potato bounding box extent

    # DataLoader
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True

    # Columns
    input_cols: tuple[str, ...] = ("x", "y", "z")
    patch_col: str = "p_id_tr"
    xi_cols: tuple[str, ...] = ("xi1_tr", "xi2_tr")
    gn_col: str = "gn_tr"
    normal_cols: tuple[str, ...] = ("nx_tr", "ny_tr", "nz_tr")
    dndxs_cols: tuple[str, ...] = (
        "dndxs_11_tr", "dndxs_12_tr", "dndxs_13_tr",
        "dndxs_21_tr", "dndxs_22_tr", "dndxs_23_tr",
        "dndxs_31_tr", "dndxs_32_tr", "dndxs_33_tr",
    )

    n_patches: int = 96  # number of Gregory patches


# ---------------------------------------------------------------------------
# Architecture building blocks
# ---------------------------------------------------------------------------
@dataclass
class MLPConfig:
    """Standard MLP with optional skip connections."""

    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: Literal["relu", "silu", "gelu", "tanh"] = "silu"
    skip_connections: bool = True
    dropout: float = 0.0


@dataclass
class SIRENConfig:
    """SIREN network (sin activations with specialized init)."""

    hidden_dims: list[int] = field(default_factory=lambda: [512, 512, 512, 512])
    omega_0: float = 30.0
    omega_hidden: float = 30.0
    h_siren: bool = False  # H-SIREN: sin(sinh(2ωx)) first layer (arXiv 2410.04716)


@dataclass
class FourierMLPConfig:
    """MLP with random Fourier feature input encoding."""

    hidden_dims: list[int] = field(default_factory=lambda: [512, 512, 512, 512])
    n_frequencies: int = 128
    frequency_scale: float = 10.0
    activation: Literal["relu", "silu", "gelu", "tanh"] = "silu"
    skip_connections: bool = True


# ---------------------------------------------------------------------------
# Phase 1: Multi-task NN
# ---------------------------------------------------------------------------
@dataclass
class MultiTaskConfig:
    """Multi-task NN: patch classification + projection + signed distance."""

    # Shared trunk
    trunk: MLPConfig = field(default_factory=lambda: MLPConfig(
        hidden_dims=[512, 512, 256, 256],
        activation="relu",
        skip_connections=True,
    ))

    # Task-specific heads
    gn_head_dims: list[int] = field(default_factory=lambda: [128, 64])
    patch_head_dims: list[int] = field(default_factory=lambda: [256, 128])
    proj_head_dims: list[int] = field(default_factory=lambda: [256, 128])

    n_patches: int = 96

    # Segmented regression: output 2*n_patches parametric coords
    # Only the pair for the true patch is supervised (Eq. 15 in paper)
    segmented_regression: bool = True

    # Loss weights (Eq. 17 in paper)
    lambda_gn: float = 1.0
    lambda_patch: float = 1.0
    lambda_proj: float = 1.0

    # GradNorm adaptive loss balancing
    use_gradnorm: bool = False
    gradnorm_alpha: float = 1.5  # restoring force exponent

    # Input encoding
    input_encoding: Literal["none", "fourier"] = "none"
    fourier_config: FourierMLPConfig | None = None

    # Architecture variants
    task_attention: bool = False       # per-task gating attention
    patch_conditioned: bool = False    # patch-conditioned regression head


# ---------------------------------------------------------------------------
# Phase 2: Neural-Pull (Direct SDF + Derivatives)
# ---------------------------------------------------------------------------
@dataclass
class NeuralPullConfig:
    """Neural-Pull: learn signed distance field with gradient/Hessian supervision."""

    architecture: Literal["siren", "fourier_mlp", "mlp"] = "siren"
    siren: SIRENConfig = field(default_factory=SIRENConfig)
    fourier_mlp: FourierMLPConfig = field(default_factory=FourierMLPConfig)
    mlp: MLPConfig = field(default_factory=lambda: MLPConfig(
        hidden_dims=[512, 512, 512, 512], activation="silu"
    ))

    # Loss weights
    lambda_sdf: float = 1.0
    lambda_grad: float = 10.0    # gradient (normal) supervision
    lambda_hess: float = 0.0     # Hessian (dn/dx_s) supervision (0 = disabled)
    lambda_eikonal: float = 0.01 # |nabla g| = 1 regularization
    lambda_steik: float = 0.0    # StEik: nᵀ∇²gn curvature penalty (NeurIPS 2023)
    lambda_gh_align: float = 0.0 # GH-alignment: ‖Hn‖²=0 (NeurIPS 2023)
    lambda_consistency: float = 0.0  # dual-head consistency loss

    # Whether to use autodiff for gradient/Hessian (True) or direct output (False)
    autodiff_derivatives: bool = True

    # Dual-head: add explicit gradient output head alongside SDF head
    dual_head: bool = False


# ---------------------------------------------------------------------------
# Phase 3: Learned Return Mapping
# ---------------------------------------------------------------------------
@dataclass
class ReturnMappingConfig:
    """Learned J2 return mapping surrogate."""

    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256, 256])
    activation: Literal["relu", "silu", "gelu"] = "silu"

    # Input: F(9) + Fp_old(9) + epcum(1) = 19
    input_dim: int = 19
    # Output: Fp_new(9) + delta_epcum(1) = 10
    output_dim: int = 10

    # Loss weights (must match return_mapping_loss() signature)
    lambda_Fp: float = 1.0
    lambda_epcum: float = 1.0
    lambda_det: float = 10.0       # det(Fp_new) > 0 constraint
    lambda_iso: float = 5.0        # isochoric: det(Fp) ≈ 1
    lambda_elastic: float = 5.0    # elastic regime: delta_ep ≈ 0 when target = 0
    lambda_inc: float = 1.0        # plastic increment accuracy


# ---------------------------------------------------------------------------
# Phase 4: GNN Newton Step Predictor
# ---------------------------------------------------------------------------
@dataclass
class GNNNewtonConfig:
    """Encode-Process-Decode GNN for Newton step prediction.

    Predicts the first Newton displacement increment Δu from current state
    (displacement, residual, contact info) to reduce iteration count.
    """

    # Encoder dims
    node_in: int = 17       # u(3) + r_norm(3) + x_ref(3) + contact(1) + gap(1) + normal(3) + load(1) + is_dirichlet(1) + bc_value(1)
    edge_in: int = 4        # dx_ref(3) + ||dx_ref||(1)
    hidden: int = 64
    node_out: int = 3       # Δu per node (3 components)

    # Message passing
    n_layers: int = 4       # 4 for n=5, 6 for n=10, 8 for n=15
    aggr: Literal["mean", "sum", "max"] = "mean"

    # Activation
    activation: Literal["silu", "relu", "gelu"] = "silu"


@dataclass
class GNNDataCollectionConfig:
    """Configuration for collecting GNN Newton training data."""

    enabled: bool = False
    output_dir: str = "nn_contact/data/gnn_newton_raw"
    record_every_n_steps: int = 1   # record every N-th load step (1 = all)
    normalize_residual: bool = True  # normalize r by max(||r||, eps)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Generic training configuration."""

    optimizer: Literal["adam", "adamw", "adadelta", "lbfgs"] = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 1000
    patience: int = 50  # early stopping patience

    # Adadelta-specific (paper used these)
    adadelta_rho: float = 0.95
    adadelta_lr: float = 0.001

    # Scheduler
    scheduler: Literal["none", "cosine", "reduce_on_plateau", "warmup_cosine"] = "reduce_on_plateau"
    scheduler_patience: int = 20
    scheduler_factor: float = 0.5
    warmup_epochs: int = 10

    # Checkpointing
    checkpoint_dir: str = "nn_contact/checkpoints"
    save_best_only: bool = True
    log_every: int = 10

    # Reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """Evaluation and integration configuration."""

    # For hybrid CDA: gap threshold below which NN prediction is used
    gn_filter_threshold: float = 0.5

    # Minimum confidence for patch classification to trust NN
    patch_confidence_threshold: float = 0.5

    # Whether to use top-K candidates instead of argmax
    use_topk: bool = False
    topk: int = 3

    # Reference data for comparison
    reference_feather: str = "for_HPC/Points4Train_Newton_LV.ft"

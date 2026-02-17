"""Physical error metrics for contact NN evaluation.

Goes beyond ML metrics (MSE, accuracy) to compute physics-relevant quantities:
- Patch classification accuracy and confusion analysis
- Projection error in parametric space
- Signed distance error distribution
- Normal direction error (angular)
- Hessian (dn/dx_s) error
- Hybrid CDA failure rate
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultiTaskMetrics:
    """Evaluation metrics for the multi-task NN (Phase 1)."""

    # Patch classification
    patch_accuracy: float           # top-1 accuracy
    patch_top3_accuracy: float      # top-3 accuracy
    patch_boundary_accuracy: float  # accuracy for points near patch boundaries

    # Parametric projection
    xi_mean_error: float   # mean ‖ξ̂ - ξ*‖₂
    xi_p50_error: float    # median
    xi_p95_error: float    # 95th percentile
    xi_p99_error: float    # 99th percentile

    # Signed distance
    gn_mean_error: float   # mean |ĝ - g*|
    gn_p95_error: float
    gn_rmse: float

    # Composite: CDA failure rate
    # A "failure" = wrong patch AND xi error > tolerance (Newton won't converge)
    cda_failure_rate: float


@dataclass
class NeuralPullMetrics:
    """Evaluation metrics for Neural-Pull (Phase 2)."""

    # SDF
    gn_rmse: float
    gn_p95_error: float

    # Gradient (normal direction)
    normal_angle_mean: float   # mean angle error (degrees)
    normal_angle_p95: float
    grad_rmse: float

    # Hessian
    hess_rmse: float | None
    hess_p95: float | None

    # Eikonal
    eikonal_mean: float  # mean ||∇g| - 1|


def compute_multitask_metrics(
    patch_pred: np.ndarray,     # (N,) int — predicted patch IDs
    patch_true: np.ndarray,     # (N,) int — true patch IDs
    xi_pred: np.ndarray,        # (N, 2)   — predicted (xi1, xi2)
    xi_true: np.ndarray,        # (N, 2)   — true (xi1, xi2)
    gn_pred: np.ndarray,        # (N,)     — predicted signed distance
    gn_true: np.ndarray,        # (N,)     — true signed distance
    patch_probs: np.ndarray | None = None,  # (N, 96) softmax probabilities
    boundary_mask: np.ndarray | None = None,  # (N,) bool — near patch boundaries
    xi_failure_tol: float = 0.1,  # xi error above which Newton may fail
) -> MultiTaskMetrics:
    """Compute all multi-task metrics."""
    N = len(patch_pred)

    # Patch accuracy
    correct = patch_pred == patch_true
    patch_accuracy = correct.mean()

    # Top-3 accuracy
    if patch_probs is not None:
        top3 = np.argsort(patch_probs, axis=-1)[:, -3:]  # (N, 3)
        top3_correct = np.any(top3 == patch_true[:, None], axis=-1)
        patch_top3_accuracy = top3_correct.mean()
    else:
        patch_top3_accuracy = patch_accuracy

    # Boundary accuracy
    if boundary_mask is not None and boundary_mask.any():
        patch_boundary_accuracy = correct[boundary_mask].mean()
    else:
        patch_boundary_accuracy = patch_accuracy

    # Projection error (only meaningful when patch is correct)
    xi_err = np.linalg.norm(xi_pred - xi_true, axis=-1)  # (N,)
    xi_mean_error = xi_err.mean()
    xi_p50_error = np.median(xi_err)
    xi_p95_error = np.percentile(xi_err, 95)
    xi_p99_error = np.percentile(xi_err, 99)

    # Signed distance error
    gn_err = np.abs(gn_pred - gn_true)
    gn_mean_error = gn_err.mean()
    gn_p95_error = np.percentile(gn_err, 95)
    gn_rmse = np.sqrt(np.mean(gn_err ** 2))

    # CDA failure: wrong patch AND projection too far for Newton to recover
    wrong_patch = ~correct
    xi_too_far = xi_err > xi_failure_tol
    failures = wrong_patch & xi_too_far
    cda_failure_rate = failures.mean()

    return MultiTaskMetrics(
        patch_accuracy=float(patch_accuracy),
        patch_top3_accuracy=float(patch_top3_accuracy),
        patch_boundary_accuracy=float(patch_boundary_accuracy),
        xi_mean_error=float(xi_mean_error),
        xi_p50_error=float(xi_p50_error),
        xi_p95_error=float(xi_p95_error),
        xi_p99_error=float(xi_p99_error),
        gn_mean_error=float(gn_mean_error),
        gn_p95_error=float(gn_p95_error),
        gn_rmse=float(gn_rmse),
        cda_failure_rate=float(cda_failure_rate),
    )


def compute_neural_pull_metrics(
    gn_pred: np.ndarray,          # (N,)
    gn_true: np.ndarray,          # (N,)
    grad_pred: np.ndarray,        # (N, 3)
    normal_true: np.ndarray,      # (N, 3)
    hess_pred: np.ndarray | None = None,  # (N, 9)
    dndxs_true: np.ndarray | None = None,  # (N, 9)
) -> NeuralPullMetrics:
    """Compute Neural-Pull evaluation metrics."""
    # SDF
    gn_err = np.abs(gn_pred - gn_true)
    gn_rmse = np.sqrt(np.mean(gn_err ** 2))
    gn_p95_error = np.percentile(gn_err, 95)

    # Normal angle error
    grad_norm = np.linalg.norm(grad_pred, axis=-1, keepdims=True)
    grad_unit = grad_pred / np.clip(grad_norm, 1e-8, None)
    normal_norm = np.linalg.norm(normal_true, axis=-1, keepdims=True)
    normal_unit = normal_true / np.clip(normal_norm, 1e-8, None)

    cos_angle = np.clip(np.sum(grad_unit * normal_unit, axis=-1), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    normal_angle_mean = angle_deg.mean()
    normal_angle_p95 = np.percentile(angle_deg, 95)

    grad_err = np.linalg.norm(grad_pred - normal_true, axis=-1)
    grad_rmse = np.sqrt(np.mean(grad_err ** 2))

    # Eikonal
    eikonal = np.abs(grad_norm.squeeze(-1) - 1.0)
    eikonal_mean = eikonal.mean()

    # Hessian
    hess_rmse = None
    hess_p95 = None
    if hess_pred is not None and dndxs_true is not None:
        hess_err = np.abs(hess_pred - dndxs_true)
        hess_rmse = float(np.sqrt(np.mean(hess_err ** 2)))
        hess_p95 = float(np.percentile(np.linalg.norm(hess_err, axis=-1), 95))

    return NeuralPullMetrics(
        gn_rmse=float(gn_rmse),
        gn_p95_error=float(gn_p95_error),
        normal_angle_mean=float(normal_angle_mean),
        normal_angle_p95=float(normal_angle_p95),
        grad_rmse=float(grad_rmse),
        hess_rmse=hess_rmse,
        hess_p95=hess_p95,
        eikonal_mean=float(eikonal_mean),
    )

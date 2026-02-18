"""Contact-specific loss functions for all phases.

Phase 1 (Multi-task):
  - Segmented regression loss for parametric coords (Eq. 15 in paper)
  - Cross-entropy for patch classification
  - MSE for signed distance
  - Optional GradNorm adaptive weighting

Phase 2 (Neural-Pull):
  - SDF MSE loss
  - Gradient (normal) supervision via autodiff
  - Hessian supervision via autodiff
  - Eikonal regularization: (|∇g| - 1)²

Phase 3 (Return Mapping):
  - Fp reconstruction loss
  - Delta epcum loss
  - det(Fp) > 0 penalty
  - Yield surface consistency
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Phase 1: Multi-task losses ───────────────────────────────────────────

def segmented_regression_loss(
    xi_pred: torch.Tensor,
    xi_target: torch.Tensor,
    patch_id: torch.Tensor,
) -> torch.Tensor:
    """Segmented regression loss (Eq. 15 in paper2.tex).

    Only the (xi1, xi2) pair corresponding to the true closest patch is
    supervised; all other 95 patch outputs are ignored.

    Parameters
    ----------
    xi_pred   : (B, 96, 2) — predicted parametric coords for all patches
    xi_target : (B, 2)     — true parametric coords
    patch_id  : (B,)       — true patch ID (0..95)

    Returns
    -------
    Scalar MSE loss over selected (xi1, xi2) pairs.
    """
    B = xi_pred.shape[0]
    # Index into the correct patch for each sample
    idx = patch_id.unsqueeze(-1).unsqueeze(-1).expand(B, 1, 2)  # (B, 1, 2)
    xi_selected = xi_pred.gather(1, idx).squeeze(1)              # (B, 2)
    return F.mse_loss(xi_selected, xi_target)


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    lambda_gn: float = 1.0,
    lambda_patch: float = 1.0,
    lambda_proj: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined multi-task loss (Eq. 17 in paper).

    Parameters
    ----------
    outputs : dict from MultiTaskContactNet.forward()
    targets : dict with 'gn', 'patch_id', 'xi'

    Returns
    -------
    total_loss : scalar tensor
    breakdown  : dict of individual loss values (for logging)
    """
    # Task 1: signed distance MSE
    loss_gn = F.mse_loss(outputs["gn_pred"].squeeze(-1), targets["gn"])

    # Task 2: patch classification cross-entropy
    loss_patch = F.cross_entropy(outputs["patch_logits"], targets["patch_id"])

    # Task 3: segmented regression
    loss_proj = segmented_regression_loss(
        outputs["xi_pred"], targets["xi"], targets["patch_id"]
    )

    total = lambda_gn * loss_gn + lambda_patch * loss_patch + lambda_proj * loss_proj

    breakdown = {
        "loss_gn": loss_gn.item(),
        "loss_patch": loss_patch.item(),
        "loss_proj": loss_proj.item(),
        "loss_total": total.item(),
    }
    return total, breakdown


# ── Phase 2: Neural-Pull losses ──────────────────────────────────────────

def steik_loss(grad_pred: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    """StEik regularizer: penalize curvature in the normal direction.

    Computes nᵀ∇²g n where n = ∇g/|∇g| (detached). For a true SDF,
    the second directional derivative along the gradient direction should
    be zero (distance increases linearly along normals).

    Uses only ONE extra backward pass (not the full 3x3 Hessian).

    Ref: Yang et al., "StEik: Stabilizing the Optimization of Neural
    Signed Distance Functions", NeurIPS 2023.

    Parameters
    ----------
    grad_pred : (B, 3) — ∂g/∂x with create_graph=True
    xyz       : (B, 3) — input coords with requires_grad=True
    """
    # Detach n to avoid differentiating through normalization
    n = (grad_pred / grad_pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)).detach()

    # Directional derivative of g in direction n: d = ∇g · n (scalar per sample)
    dir_deriv = (grad_pred * n).sum(dim=-1)  # (B,)

    # Second directional derivative via autograd: Hn = ∂(∇g·n)/∂x
    Hn = torch.autograd.grad(
        dir_deriv.sum(), xyz, create_graph=True, retain_graph=True,
    )[0]  # (B, 3)

    # nᵀHn = n · Hn
    nHn = (n * Hn).sum(dim=-1)  # (B,)

    return (nHn ** 2).mean()


def neural_pull_loss(
    g_pred: torch.Tensor,
    grad_pred: torch.Tensor,
    g_target: torch.Tensor,
    normal_target: torch.Tensor,
    xyz: torch.Tensor | None = None,
    grad_direct: torch.Tensor | None = None,
    hess_pred: torch.Tensor | None = None,
    dndxs_target: torch.Tensor | None = None,
    lambda_sdf: float = 1.0,
    lambda_grad: float = 10.0,
    lambda_hess: float = 1.0,
    lambda_eikonal: float = 0.1,
    lambda_steik: float = 0.0,
    lambda_consistency: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Neural-Pull loss with gradient/Hessian supervision + eikonal + StEik.

    Parameters
    ----------
    g_pred        : (B, 1)    — predicted SDF
    grad_pred     : (B, 3)    — ∂g/∂x via autodiff
    g_target      : (B,)      — true signed distance
    normal_target : (B, 3)    — true normal direction
    xyz           : (B, 3)    — input coords (needed for StEik, requires_grad)
    grad_direct   : (B, 3)    — direct gradient head output (optional, dual-head)
    hess_pred     : (B, 3, 3) — ∂²g/∂x² via autodiff (optional)
    dndxs_target  : (B, 9)    — true dn/dx_s flattened (optional)
    lambda_steik  : float     — StEik normal-curvature regularizer weight
    lambda_consistency : float — dual-head consistency loss weight
    """
    # SDF reconstruction
    loss_sdf = F.mse_loss(g_pred.squeeze(-1), g_target)

    # Gradient supervision: ∇g should match the normal
    loss_grad = F.mse_loss(grad_pred, normal_target)

    # Eikonal: |∇g| should be 1 (SDF property)
    grad_norm = grad_pred.norm(dim=-1)
    loss_eik = F.mse_loss(grad_norm, torch.ones_like(grad_norm))

    total = lambda_sdf * loss_sdf + lambda_grad * loss_grad + lambda_eikonal * loss_eik
    breakdown = {
        "loss_sdf": loss_sdf.item(),
        "loss_grad": loss_grad.item(),
        "loss_eikonal": loss_eik.item(),
    }

    # Optional Hessian supervision
    if hess_pred is not None and dndxs_target is not None:
        hess_flat = hess_pred.reshape(-1, 9)
        loss_hess = F.mse_loss(hess_flat, dndxs_target)
        total = total + lambda_hess * loss_hess
        breakdown["loss_hess"] = loss_hess.item()

    # StEik: penalize nᵀ∇²gn (curvature in normal direction)
    if lambda_steik > 0 and xyz is not None:
        loss_st = steik_loss(grad_pred, xyz)
        total = total + lambda_steik * loss_st
        breakdown["loss_steik"] = loss_st.item()

    # Dual-head: direct gradient supervision + consistency
    if grad_direct is not None:
        loss_grad_direct = F.mse_loss(grad_direct, normal_target)
        total = total + lambda_grad * loss_grad_direct
        breakdown["loss_grad_direct"] = loss_grad_direct.item()

        if lambda_consistency > 0:
            loss_cons = F.mse_loss(grad_pred.detach(), grad_direct)
            total = total + lambda_consistency * loss_cons
            breakdown["loss_consistency"] = loss_cons.item()

    breakdown["loss_total"] = total.item()
    return total, breakdown


# ── Phase 3: Return mapping losses ───────────────────────────────────────

def _det3x3(M: torch.Tensor) -> torch.Tensor:
    """Batched determinant of (B, 9) flattened 3x3 matrices."""
    a, b, c, d, e, f, g, h, i = M.unbind(dim=-1)
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def return_mapping_loss(
    outputs: dict[str, torch.Tensor],
    Fp_target: torch.Tensor,
    dep_target: torch.Tensor,
    lambda_Fp: float = 1.0,
    lambda_ep: float = 1.0,
    lambda_det: float = 10.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return mapping surrogate loss with physics constraints.

    Parameters
    ----------
    outputs    : dict from ReturnMappingNet.forward()
    Fp_target  : (B, 9) — true updated Fp (flattened)
    dep_target : (B,)   — true delta epcum
    """
    Fp_new = outputs["Fp_new"]
    delta_ep = outputs["delta_ep"]

    # Reconstruction losses
    loss_Fp = F.mse_loss(Fp_new, Fp_target)
    loss_ep = F.mse_loss(delta_ep, dep_target)

    # Physics: det(Fp_new) > 0 penalty
    det_Fp = _det3x3(Fp_new)
    loss_det = torch.clamp(-det_Fp, min=0.0).pow(2).mean()

    total = lambda_Fp * loss_Fp + lambda_ep * loss_ep + lambda_det * loss_det

    breakdown = {
        "loss_Fp": loss_Fp.item(),
        "loss_ep": loss_ep.item(),
        "loss_det": loss_det.item(),
        "det_Fp_min": det_Fp.min().item(),
        "loss_total": total.item(),
    }
    return total, breakdown


# ── GradNorm adaptive loss balancing ─────────────────────────────────────

class GradNormBalancer:
    """GradNorm: gradient normalization for multi-task learning.

    Ref: Chen et al., "GradNorm: Gradient Normalization for Adaptive
    Loss Balancing in Deep Multitask Networks", ICML 2018.

    Dynamically adjusts loss weights so that all tasks train at similar rates.
    """

    def __init__(self, n_tasks: int, alpha: float = 1.5, lr: float = 0.025):
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.lr = lr
        # Learnable weights (log-space for positivity)
        self.log_weights = torch.zeros(n_tasks, requires_grad=True)
        self._initial_losses: torch.Tensor | None = None

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.log_weights, dim=0) * self.n_tasks

    def step(self, losses: list[torch.Tensor], shared_params: list[nn.Parameter]):
        """Update weights based on gradient norms."""
        weights = self.weights

        # Compute gradient norms for each task
        grad_norms = []
        for i, loss in enumerate(losses):
            grads = torch.autograd.grad(
                weights[i] * loss, shared_params,
                retain_graph=True, allow_unused=True,
            )
            total_norm = sum(g.norm() for g in grads if g is not None)
            grad_norms.append(total_norm)

        grad_norms = torch.stack(grad_norms)
        mean_norm = grad_norms.mean()

        # Relative training rates
        if self._initial_losses is None:
            self._initial_losses = torch.tensor([l.item() for l in losses])

        loss_ratios = torch.tensor([l.item() for l in losses]) / self._initial_losses
        relative_rates = loss_ratios / loss_ratios.mean()

        # Target gradient norms
        target_norms = mean_norm * relative_rates.pow(self.alpha)

        # GradNorm loss
        gn_loss = (grad_norms - target_norms.to(grad_norms.device)).abs().sum()

        # Update log_weights
        self.log_weights.grad = torch.autograd.grad(gn_loss, self.log_weights)[0]
        with torch.no_grad():
            self.log_weights -= self.lr * self.log_weights.grad
            self.log_weights.grad = None

        return self.weights.detach()

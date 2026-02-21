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

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Focal loss for classification: down-weights easy examples.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Ref: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Parameters
    ----------
    logits : (B, C) — raw logits
    targets : (B,) — class indices
    gamma : float — focusing parameter (0 = standard CE, 2 = strong focus)
    label_smoothing : float — label smoothing epsilon (0 = none)
    """
    ce = F.cross_entropy(logits, targets, reduction="none",
                         label_smoothing=label_smoothing)
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


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
    focal_gamma: float = 0.0,
    label_smoothing: float = 0.0,
    ohem_ratio: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined multi-task loss (Eq. 17 in paper).

    Parameters
    ----------
    outputs : dict from MultiTaskContactNet.forward()
    targets : dict with 'gn', 'patch_id', 'xi'
    focal_gamma : float — focal loss gamma (0 = standard CE)
    label_smoothing : float — label smoothing epsilon
    ohem_ratio : float — fraction of hardest samples to keep (1.0 = all)

    Returns
    -------
    total_loss : scalar tensor
    breakdown  : dict of individual loss values (for logging)
    """
    gn_pred = outputs["gn_pred"].squeeze(-1)
    patch_logits = outputs["patch_logits"]
    xi_pred = outputs["xi_pred"]
    gn_target = targets["gn"]
    patch_id = targets["patch_id"]
    xi_target = targets["xi"]

    # OHEM: compute per-sample losses, keep only the hardest fraction
    if ohem_ratio < 1.0:
        B = gn_pred.shape[0]
        k = max(1, int(B * ohem_ratio))

        # Per-sample losses for difficulty scoring
        per_gn = (gn_pred - gn_target).pow(2)  # (B,)
        per_cls = F.cross_entropy(patch_logits, patch_id, reduction="none",
                                  label_smoothing=label_smoothing)  # (B,)
        idx_xi = patch_id.unsqueeze(-1).unsqueeze(-1).expand(B, 1, 2)
        xi_sel = xi_pred.gather(1, idx_xi).squeeze(1)
        per_proj = (xi_sel - xi_target).pow(2).sum(dim=-1)  # (B,)

        # Difficulty = sum of all per-sample losses
        difficulty = per_gn + per_cls + per_proj
        _, hard_idx = difficulty.topk(k)

        # Select hard samples
        gn_pred = gn_pred[hard_idx]
        gn_target = gn_target[hard_idx]
        patch_logits = patch_logits[hard_idx]
        patch_id = patch_id[hard_idx]
        xi_pred = xi_pred[hard_idx]
        xi_target = xi_target[hard_idx]

    # Task 1: signed distance MSE
    loss_gn = F.mse_loss(gn_pred, gn_target)

    # Task 2: patch classification (focal or standard CE)
    if focal_gamma > 0:
        loss_patch = focal_loss(patch_logits, patch_id, gamma=focal_gamma,
                                label_smoothing=label_smoothing)
    else:
        loss_patch = F.cross_entropy(patch_logits, patch_id,
                                     label_smoothing=label_smoothing)

    # Task 3: segmented regression
    loss_proj = segmented_regression_loss(xi_pred, xi_target, patch_id)

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


def gh_alignment_loss(grad_pred: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    """Gradient-Hessian alignment: ‖Hn‖² = 0.

    For a true SDF, the gradient is an eigenvector of the Hessian with
    eigenvalue zero: H·n = 0. This constrains gradient DIRECTION (3 components)
    while StEik only constrains the scalar nᵀHn.

    Ref: Wang et al., "Aligning Gradient and Hessian for Neural SDF
    Optimization", NeurIPS 2023.

    Parameters
    ----------
    grad_pred : (B, 3) — ∂g/∂x with create_graph=True
    xyz       : (B, 3) — input coords with requires_grad=True
    """
    n = (grad_pred / grad_pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)).detach()

    # Hn = ∂(∇g·n)/∂x = H·n (one backward pass, not full 3x3 Hessian)
    dir_deriv = (grad_pred * n).sum(dim=-1)  # (B,)
    Hn = torch.autograd.grad(
        dir_deriv.sum(), xyz, create_graph=True, retain_graph=True,
    )[0]  # (B, 3)

    return (Hn ** 2).sum(dim=-1).mean()


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
    lambda_gh_align: float = 0.0,
    lambda_consistency: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Neural-Pull loss with gradient/Hessian supervision + eikonal + StEik + GH-align.

    Parameters
    ----------
    g_pred        : (B, 1)    — predicted SDF
    grad_pred     : (B, 3)    — ∂g/∂x via autodiff
    g_target      : (B,)      — true signed distance
    normal_target : (B, 3)    — true normal direction
    xyz           : (B, 3)    — input coords (needed for StEik/GH-align, requires_grad)
    grad_direct   : (B, 3)    — direct gradient head output (optional, dual-head)
    hess_pred     : (B, 3, 3) — ∂²g/∂x² via autodiff (optional)
    dndxs_target  : (B, 9)    — true dn/dx_s flattened (optional)
    lambda_steik  : float     — StEik normal-curvature regularizer weight
    lambda_gh_align : float   — GH-alignment ‖Hn‖²=0 regularizer weight
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

    # GH-Alignment: ‖Hn‖² = 0 (gradient is zero-eigenvector of Hessian)
    if lambda_gh_align > 0 and xyz is not None:
        loss_gh = gh_alignment_loss(grad_pred, xyz)
        total = total + lambda_gh_align * loss_gh
        breakdown["loss_gh_align"] = loss_gh.item()

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
    lambda_iso: float = 5.0,
    lambda_elastic: float = 5.0,
    lambda_inc: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return mapping surrogate loss with physics constraints.

    Parameters
    ----------
    outputs    : dict from ReturnMappingNet.forward()
    Fp_target  : (B, 9) — true updated Fp (flattened)
    dep_target : (B,)   — true delta epcum
    lambda_Fp  : weight for Fp reconstruction
    lambda_ep  : weight for delta_ep reconstruction
    lambda_det : weight for det(Fp) > 0 constraint
    lambda_iso : weight for isochoric constraint det(Fp) ≈ 1
    lambda_elastic : weight for elastic regime accuracy (delta_ep should be 0)
    lambda_inc : weight for plastic increment accuracy (Fp_new - Fp_old)
    """
    Fp_new = outputs["Fp_new"]
    delta_ep = outputs["delta_ep"]
    Fp_old = outputs["Fp_old"]

    # Primary: Fp reconstruction (Frobenius)
    loss_Fp = F.mse_loss(Fp_new, Fp_target)

    # Primary: delta_ep reconstruction
    loss_ep = F.mse_loss(delta_ep, dep_target)

    # Physics 1: det(Fp_new) > 0 — hard constraint (prevent inversion)
    det_Fp = _det3x3(Fp_new)
    loss_det = torch.clamp(-det_Fp, min=0.0).pow(2).mean()

    # Physics 2: isochoric plastic flow — det(Fp) ≈ 1 for J2 plasticity
    loss_iso = F.mse_loss(det_Fp, torch.ones_like(det_Fp))

    # Physics 3: elastic regime accuracy — if target delta_ep = 0,
    # prediction should also be ~0 (correct elastic-plastic transition)
    elastic_mask = dep_target < 1e-10
    if elastic_mask.any():
        loss_elastic = F.mse_loss(
            delta_ep[elastic_mask],
            torch.zeros_like(delta_ep[elastic_mask]))
    else:
        loss_elastic = torch.tensor(0.0, device=Fp_new.device)

    # Physics 4: plastic increment accuracy — Fp_new - Fp_old should
    # match the target increment (focus on the actual plastic update)
    loss_inc = F.mse_loss(Fp_new - Fp_old, Fp_target - Fp_old)

    total = (lambda_Fp * loss_Fp
             + lambda_ep * loss_ep
             + lambda_det * loss_det
             + lambda_iso * loss_iso
             + lambda_elastic * loss_elastic
             + lambda_inc * loss_inc)

    breakdown = {
        "loss_Fp": loss_Fp.item(),
        "loss_ep": loss_ep.item(),
        "loss_det": loss_det.item(),
        "loss_iso": loss_iso.item(),
        "loss_elastic": loss_elastic.item(),
        "loss_inc": loss_inc.item(),
        "det_Fp_min": det_Fp.min().item(),
        "det_Fp_mean": det_Fp.mean().item(),
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
        # Ensure log_weights on same device as losses
        dev = losses[0].device
        if self.log_weights.device != dev:
            self.log_weights = self.log_weights.to(dev).requires_grad_(True)
            if self._initial_losses is not None:
                self._initial_losses = self._initial_losses.to(dev)
        weights = self.weights

        # Compute gradient norms for each task
        # create_graph=True so grad_norms are differentiable w.r.t. log_weights
        grad_norms = []
        for i, loss in enumerate(losses):
            grads = torch.autograd.grad(
                weights[i] * loss, shared_params,
                retain_graph=True, allow_unused=True,
                create_graph=True,
            )
            total_norm = sum(g.norm() for g in grads if g is not None)
            grad_norms.append(total_norm)

        grad_norms = torch.stack(grad_norms)

        # Detach mean_norm: target is a fixed reference, not a moving target
        mean_norm = grad_norms.mean().detach()

        # Relative training rates (keep everything on same device)
        if self._initial_losses is None:
            self._initial_losses = torch.tensor([l.item() for l in losses], device=dev)

        loss_ratios = torch.tensor([l.item() for l in losses], device=dev) / self._initial_losses
        relative_rates = loss_ratios / loss_ratios.mean()

        # Target gradient norms (detached — no gradient through targets)
        target_norms = (mean_norm * relative_rates.pow(self.alpha)).detach()

        # GradNorm loss: how far each task's grad norm is from its target
        gn_loss = (grad_norms - target_norms).abs().sum()

        # Update log_weights via manual gradient step
        self.log_weights.grad = torch.autograd.grad(gn_loss, self.log_weights)[0]
        with torch.no_grad():
            self.log_weights -= self.lr * self.log_weights.grad
            self.log_weights.grad = None

        return self.weights.detach()

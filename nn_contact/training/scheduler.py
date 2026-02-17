"""Learning rate schedulers and curriculum learning strategies."""

from __future__ import annotations

import math

import torch


class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr_factor: float = 0.01):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_factor = min_lr_factor

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return max(min_lr_factor, epoch / max(1, warmup_epochs))
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return max(min_lr_factor, 0.5 * (1 + math.cos(math.pi * progress)))

        super().__init__(optimizer, lr_lambda)


class CurriculumScheduler:
    """Curriculum learning: gradually increase data difficulty.

    For contact data, "difficulty" = proximity to surface.  Start training
    on points far from the surface (easy) and progressively include points
    closer to / penetrating the surface (hard — near patch boundaries,
    normal discontinuities).

    Usage::

        curriculum = CurriculumScheduler(gn_min=-0.5, gn_max=1.5, warmup_epochs=100)
        for epoch in range(epochs):
            gn_range = curriculum.get_range(epoch)
            # Filter batch to gn_range...
    """

    def __init__(
        self,
        gn_min: float = -0.5,
        gn_max: float = 1.5,
        warmup_epochs: int = 100,
        start_gn_min: float = 0.0,
        start_gn_max: float = 1.5,
    ):
        self.gn_min = gn_min
        self.gn_max = gn_max
        self.warmup_epochs = warmup_epochs
        self.start_gn_min = start_gn_min
        self.start_gn_max = start_gn_max

    def get_range(self, epoch: int) -> tuple[float, float]:
        if epoch >= self.warmup_epochs:
            return self.gn_min, self.gn_max
        t = epoch / self.warmup_epochs
        current_min = self.start_gn_min + t * (self.gn_min - self.start_gn_min)
        current_max = self.start_gn_max + t * (self.gn_max - self.start_gn_max)
        return current_min, current_max

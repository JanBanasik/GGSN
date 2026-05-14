"""Loss functions for Phase 2 GNN training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """BCE with focal down-weighting of easy examples.

    gamma=0 → plain BCE (with pos_weight).
    gamma=2 → standard focal loss; hard positives dominate.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        if self.gamma == 0.0:
            return bce.mean()
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return (((1 - p_t) ** self.gamma) * bce).mean()

"""Loss functions for Phase 1 contrastive and Phase 2 GNN training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(z_text: torch.Tensor, z_signal: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE (NT-Xent) loss over in-batch positives.

    z_text, z_signal are L2-normalised (N, D).
    Diagonal entries are the positive pairs; off-diagonal are negatives.
    """
    logits = torch.matmul(z_text, z_signal.T) / temperature  # (N, N)
    labels = torch.arange(z_text.size(0), device=z_text.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


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

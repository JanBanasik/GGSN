import torch
import torch.nn.functional as F

from src.training.loss import FocalLoss, info_nce_loss


def test_info_nce_prefers_aligned_pairs() -> None:
    aligned = torch.eye(3)
    reversed_pairs = torch.flip(aligned, dims=[0])

    assert info_nce_loss(aligned, aligned, 0.1) < info_nce_loss(aligned, reversed_pairs, 0.1)


def test_focal_loss_with_zero_gamma_matches_bce() -> None:
    logits = torch.tensor([0.2, -1.0, 1.5])
    targets = torch.tensor([1.0, 0.0, 1.0])
    pos_weight = torch.tensor([2.0])

    actual = FocalLoss(gamma=0.0, pos_weight=pos_weight)(logits, targets)
    expected = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    assert torch.allclose(actual, expected)

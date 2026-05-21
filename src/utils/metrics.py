"""Phase 2 evaluation metrics for mortality prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_score))


def sens_at_spec(y_true: np.ndarray, y_score: np.ndarray, target_spec: float = 0.95) -> float:
    """Sensitivity at a given specificity threshold."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= (1.0 - target_spec) + 1e-6
    return float(tpr[mask][-1]) if mask.any() else 0.0

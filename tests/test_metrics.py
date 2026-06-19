import numpy as np
import pytest

from src.utils.metrics import auprc, auroc, brier_score, sens_at_spec


def test_binary_metrics_for_perfect_ranking() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    assert auroc(labels, scores) == 1.0
    assert auprc(labels, scores) == 1.0
    assert brier_score(labels, scores) == pytest.approx(0.025)
    assert sens_at_spec(labels, scores, target_spec=0.95) == 1.0

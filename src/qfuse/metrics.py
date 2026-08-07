from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    conf = probabilities.max(axis=1)
    pred = probabilities.argmax(axis=1)
    correct = pred == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if np.any(mask):
            ece += mask.mean() * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def selective_risk(y_true: np.ndarray, y_pred: np.ndarray, accepted: np.ndarray) -> float:
    accepted = np.asarray(accepted, dtype=bool)
    if not np.any(accepted):
        return 0.0
    return float(np.mean(np.asarray(y_true)[accepted] != np.asarray(y_pred)[accepted]))


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC from ranks without requiring scikit-learn."""
    from scipy.stats import rankdata

    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) != len(s):
        raise ValueError("y_true and scores must have the same length")
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    auc = (float(ranks[pos].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

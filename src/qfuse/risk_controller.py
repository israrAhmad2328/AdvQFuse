from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta


@dataclass(slots=True)
class ThresholdCalibration:
    action: str
    threshold: float
    coverage: float
    empirical_risk: float
    upper_risk_bound: float
    accepted_count: int
    total_count: int
    valid_groups: tuple[str, ...]


@dataclass(slots=True)
class ActionDecision:
    action: str
    estimated_post_action_risk: float
    estimated_cost: float
    threshold: float | None
    reason: str


def clopper_pearson_upper(errors: int, n: int, delta: float = 0.05) -> float:
    """One-sided exact binomial upper confidence limit."""

    errors = int(errors)
    n = int(n)
    if n < 0 or errors < 0 or errors > n:
        raise ValueError("require 0 <= errors <= n")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0,1)")
    if n == 0:
        return 1.0
    if errors == n:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, n - errors))


def _risk_summary(losses: np.ndarray, delta: float) -> tuple[float, float, int]:
    n = int(losses.size)
    if n == 0:
        return 1.0, 1.0, 0
    errors = int(np.sum(losses > 0.5))
    empirical = float(errors / n)
    upper = clopper_pearson_upper(errors, n, delta=delta)
    return empirical, upper, n


def calibrate_group_robust_threshold(
    scores: Sequence[float],
    losses: Sequence[int | bool | float],
    *,
    action: str,
    groups: Sequence[str] | None = None,
    alpha: float = 0.05,
    delta: float = 0.05,
    min_group_samples: int = 25,
    min_coverage: float = 0.0,
) -> ThresholdCalibration:
    """Find the largest score threshold satisfying global and supported-group risk.

    Lower scores indicate safer samples. Bonferroni allocation across the global
    calibration set and supported groups prevents silently selecting a threshold
    after looking at many group bounds.
    """

    score = np.asarray(scores, dtype=float).reshape(-1)
    loss = np.asarray(losses, dtype=float).reshape(-1)
    if score.shape != loss.shape:
        raise ValueError("scores and losses must have equal length")
    if score.size == 0:
        raise ValueError("calibration data cannot be empty")
    if np.any(~np.isfinite(score)):
        raise ValueError("scores contain non-finite values")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if groups is None:
        group_array = np.full(score.shape, "all", dtype=object)
    else:
        group_array = np.asarray(groups, dtype=object).reshape(-1)
        if group_array.shape != score.shape:
            raise ValueError("groups must have the same length as scores")

    labels, counts = np.unique(group_array, return_counts=True)
    supported = tuple(str(label) for label, count in zip(labels, counts, strict=True) if count >= min_group_samples)
    # One global constraint plus every supported group.
    local_delta = delta / max(1, 1 + len(supported))
    candidate_thresholds = np.unique(np.r_[np.nextafter(score.min(), -np.inf), score])
    best: ThresholdCalibration | None = None

    for threshold in candidate_thresholds:
        accepted = score <= threshold
        coverage = float(np.mean(accepted))
        if coverage < min_coverage or not np.any(accepted):
            continue
        empirical, global_upper, accepted_count = _risk_summary(loss[accepted], local_delta)
        worst_upper = global_upper
        valid = global_upper <= alpha
        for group in supported:
            group_mask = accepted & (group_array.astype(str) == group)
            # A selected subset too small for a group is not considered certified.
            if int(group_mask.sum()) < min_group_samples:
                valid = False
                worst_upper = 1.0
                break
            _, group_upper, _ = _risk_summary(loss[group_mask], local_delta)
            worst_upper = max(worst_upper, group_upper)
            if group_upper > alpha:
                valid = False
                break
        if valid:
            best = ThresholdCalibration(
                action=action,
                threshold=float(threshold),
                coverage=coverage,
                empirical_risk=empirical,
                upper_risk_bound=float(worst_upper),
                accepted_count=accepted_count,
                total_count=int(score.size),
                valid_groups=supported,
            )

    if best is None:
        return ThresholdCalibration(
            action=action,
            threshold=float("-inf"),
            coverage=0.0,
            empirical_risk=0.0,
            upper_risk_bound=1.0,
            accepted_count=0,
            total_count=int(score.size),
            valid_groups=supported,
        )
    return best


class RiskControlledActionPolicy:
    """Select the cheapest calibrated action; abstain when none is certified."""

    def __init__(
        self,
        calibrations: Mapping[str, ThresholdCalibration],
        costs: Mapping[str, float],
        abstain_action: str = "abstain",
    ) -> None:
        self.calibrations = dict(calibrations)
        self.costs = {str(k): float(v) for k, v in costs.items()}
        self.abstain_action = abstain_action

    def decide(self, predicted_post_action_risk: Mapping[str, float]) -> ActionDecision:
        candidates: list[tuple[float, str, float, float]] = []
        for action, calibration in self.calibrations.items():
            if action not in predicted_post_action_risk:
                continue
            score = float(predicted_post_action_risk[action])
            if score <= calibration.threshold:
                candidates.append((self.costs.get(action, float("inf")), action, score, calibration.threshold))
        if not candidates:
            score = float(predicted_post_action_risk.get(self.abstain_action, 0.0))
            return ActionDecision(
                action=self.abstain_action,
                estimated_post_action_risk=score,
                estimated_cost=self.costs.get(self.abstain_action, 0.0),
                threshold=None,
                reason="no non-abstention action satisfies its calibrated risk threshold",
            )
        cost, action, score, threshold = min(candidates, key=lambda item: (item[0], item[2]))
        return ActionDecision(
            action=action,
            estimated_post_action_risk=score,
            estimated_cost=cost,
            threshold=threshold,
            reason="cheapest action satisfying the finite-sample calibrated threshold",
        )

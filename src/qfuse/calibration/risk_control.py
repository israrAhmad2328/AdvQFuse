from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


@dataclass(slots=True)
class RiskCalibrationResult:
    threshold: float
    accepted: int
    calibration_coverage: float
    empirical_risk: float
    upper_risk_bound: float
    target_risk: float
    delta: float


def _clopper_pearson_upper(errors: int, total: int, delta: float) -> float:
    if total <= 0:
        return 1.0
    if errors >= total:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, total - errors))


class SelectiveRiskController:
    """Bonferroni-corrected finite-sample risk calibration.

    Scores are failure probabilities or any nonconformity score where lower is safer.
    The controller chooses the largest-coverage threshold whose one-sided binomial
    upper confidence bound does not exceed the target risk.
    """

    def __init__(self, target_risk: float = 0.10, delta: float = 0.05, min_accepted: int = 50) -> None:
        if not (0 < target_risk < 1 and 0 < delta < 1):
            raise ValueError("target_risk and delta must lie in (0, 1)")
        self.target_risk = float(target_risk)
        self.delta = float(delta)
        self.min_accepted = int(min_accepted)
        self.result_: RiskCalibrationResult | None = None

    def fit(self, scores: np.ndarray, errors: np.ndarray) -> RiskCalibrationResult:
        scores = np.asarray(scores, dtype=float).reshape(-1)
        errors = np.asarray(errors, dtype=int).reshape(-1)
        if len(scores) != len(errors) or len(scores) == 0:
            raise ValueError("scores and errors must have the same non-zero length")
        if not np.all(np.isin(errors, [0, 1])):
            raise ValueError("errors must contain only 0/1 values")
        order = np.argsort(scores, kind="mergesort")
        s = scores[order]
        e = errors[order]
        unique_end = np.r_[np.where(np.diff(s) > 0)[0], len(s) - 1]
        m = max(len(unique_end), 1)
        per_test_delta = self.delta / m
        cumulative_errors = np.cumsum(e)
        best: RiskCalibrationResult | None = None
        for idx in unique_end:
            n = int(idx + 1)
            if n < self.min_accepted:
                continue
            err = int(cumulative_errors[idx])
            empirical = err / n
            upper = _clopper_pearson_upper(err, n, per_test_delta)
            if upper <= self.target_risk:
                candidate = RiskCalibrationResult(
                    threshold=float(s[idx]),
                    accepted=n,
                    calibration_coverage=n / len(s),
                    empirical_risk=empirical,
                    upper_risk_bound=upper,
                    target_risk=self.target_risk,
                    delta=self.delta,
                )
                if best is None or candidate.calibration_coverage > best.calibration_coverage:
                    best = candidate
        if best is None:
            best = RiskCalibrationResult(
                threshold=float("-inf"),
                accepted=0,
                calibration_coverage=0.0,
                empirical_risk=0.0,
                upper_risk_bound=1.0,
                target_risk=self.target_risk,
                delta=self.delta,
            )
        self.result_ = best
        return best

    def accept(self, scores: np.ndarray) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("controller is not fitted")
        return np.asarray(scores, dtype=float) <= self.result_.threshold

import numpy as np

from qfuse.risk_controller import (
    RiskControlledActionPolicy,
    calibrate_group_robust_threshold,
    clopper_pearson_upper,
)


def test_clopper_pearson_is_conservative():
    upper = clopper_pearson_upper(errors=2, n=100, delta=0.05)
    assert upper > 0.02
    assert upper < 0.10


def test_group_robust_calibration_and_policy():
    rng = np.random.default_rng(7)
    scores = np.linspace(0.0, 1.0, 300)
    losses = (scores + rng.normal(0, 0.10, len(scores)) > 0.68).astype(int)
    groups = np.where(np.arange(len(scores)) % 2 == 0, "a", "b")
    calibration = calibrate_group_robust_threshold(
        scores,
        losses,
        action="accept",
        groups=groups,
        alpha=0.10,
        delta=0.05,
        min_group_samples=30,
    )
    assert calibration.coverage > 0
    assert calibration.upper_risk_bound <= 0.10

    policy = RiskControlledActionPolicy(
        calibrations={"accept": calibration},
        costs={"accept": 1.0, "abstain": 0.0},
    )
    safe = policy.decide({"accept": max(0.0, calibration.threshold - 1e-4)})
    unsafe = policy.decide({"accept": calibration.threshold + 0.2})
    assert safe.action == "accept"
    assert unsafe.action == "abstain"

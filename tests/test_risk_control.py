import numpy as np

from qfuse.calibration import SelectiveRiskController


def test_risk_controller_selects_safe_prefix() -> None:
    rng = np.random.default_rng(1)
    scores = np.linspace(0, 1, 1000)
    errors = (rng.random(1000) < (0.01 + 0.5 * scores)).astype(int)
    rc = SelectiveRiskController(target_risk=0.15, delta=0.05, min_accepted=100)
    result = rc.fit(scores, errors)
    accepted = rc.accept(scores)
    assert result.accepted >= 100
    assert result.upper_risk_bound <= 0.15
    assert accepted.sum() == result.accepted

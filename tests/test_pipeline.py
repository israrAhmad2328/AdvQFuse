import numpy as np

from qfuse import QFuseController
from qfuse.calibration import SelectiveRiskController


def test_controller_roundtrip_and_binary_only_inference(tmp_path) -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(900, 5))
    targets = np.column_stack(
        [
            np.clip(0.08 + 0.05 * x[:, 0] + 0.02 * x[:, 1], 0, np.log(2)),
            (x[:, 1] - x[:, 3] > 0.6).astype(float),
        ]
    )
    errors = (x[:, 0] + x[:, 2] > 1.0).astype(int)
    controller = QFuseController(
        risk_controller=SelectiveRiskController(
            target_risk=0.20, delta=0.05, min_accepted=30
        )
    ).fit(x[:500], targets[:500], errors[:500], x[500:700], errors[500:700])
    scores = controller.predict_failure_probability(x[700:])
    assert scores.shape == (200,)
    assert np.all((scores >= 0) & (scores <= 1))
    path = tmp_path / "controller.json"
    controller.save(path)
    restored = QFuseController.load(path)
    assert np.allclose(
        restored.predict_failure_probability(x[700:720]),
        controller.predict_failure_probability(x[700:720]),
    )

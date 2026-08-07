import numpy as np

from qfuse.uncertainty import LogisticFailureModel, decompose_uncertainty


def test_decomposition_detects_precision_flip() -> None:
    f = decompose_uncertainty(
        binary_modalities=[np.array([0.8, 0.2])],
        ternary_modalities=[np.array([0.2, 0.8])],
        qualities=[0.8],
        fused_uncertainty=0.2,
    )
    assert f.precision_label_flip == 1.0
    assert f.quantization_disagreement > 0


def test_logistic_failure_model_learns_signal() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    model = LogisticFailureModel().fit(x, y, steps=1200)
    pred = model.predict_proba(x)[:, 1] > 0.5
    assert np.mean(pred == y) > 0.85


def test_precision_sensitivity_is_deployable_and_serializable() -> None:
    from qfuse.uncertainty import (
        PrecisionSensitivityModel,
        build_deployable_features,
    )

    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 5))
    disagreement = np.clip(0.12 + 0.08 * x[:, 0] - 0.05 * x[:, 3], 0, np.log(2))
    flip = (x[:, 1] + x[:, 2] > 0.5).astype(float)
    targets = np.column_stack([disagreement, flip])
    model = PrecisionSensitivityModel().fit(x, targets)
    pred = model.predict(x)
    assert pred.shape == (500, 2)
    assert np.mean(np.abs(pred[:, 0] - disagreement)) < 0.03
    deployable = build_deployable_features(x, pred)
    assert deployable.shape == (500, 7)
    restored = PrecisionSensitivityModel.from_dict(model.to_dict())
    assert np.allclose(restored.predict(x[:20]), pred[:20])

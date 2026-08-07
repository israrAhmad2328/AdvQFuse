from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from qfuse.calibration import RiskCalibrationResult, SelectiveRiskController
from qfuse.uncertainty import (
    LogisticFailureModel,
    PrecisionSensitivityModel,
    build_deployable_features,
)


class QFuseController:
    """Deployable QFuse controller trained from paired-precision calibration data.

    Paired binary/ternary targets are consumed only while fitting the precision
    sensitivity model. At inference, the controller requires only features from
    the binary pass.
    """

    def __init__(
        self,
        sensitivity_model: PrecisionSensitivityModel | None = None,
        failure_model: LogisticFailureModel | None = None,
        risk_controller: SelectiveRiskController | None = None,
    ) -> None:
        self.sensitivity_model = sensitivity_model or PrecisionSensitivityModel()
        self.failure_model = failure_model or LogisticFailureModel(l2=5e-3)
        self.risk_controller = risk_controller or SelectiveRiskController(
            target_risk=0.05, delta=0.05, min_accepted=50
        )

    def fit(
        self,
        train_binary_observables: np.ndarray,
        train_precision_targets: np.ndarray,
        train_binary_errors: np.ndarray,
        calibration_binary_observables: np.ndarray,
        calibration_binary_errors: np.ndarray,
    ) -> "QFuseController":
        train_x = np.asarray(train_binary_observables, dtype=float)
        cal_x = np.asarray(calibration_binary_observables, dtype=float)
        train_errors = np.asarray(train_binary_errors, dtype=int).reshape(-1)
        cal_errors = np.asarray(calibration_binary_errors, dtype=int).reshape(-1)
        if len(train_x) != len(train_errors):
            raise ValueError("training observables and errors have inconsistent lengths")
        if len(cal_x) != len(cal_errors):
            raise ValueError("calibration observables and errors have inconsistent lengths")

        self.sensitivity_model.fit(train_x, train_precision_targets)
        train_precision = self.sensitivity_model.predict(train_x)
        train_features = build_deployable_features(train_x, train_precision)
        self.failure_model.fit(train_features, train_errors)

        cal_precision = self.sensitivity_model.predict(cal_x)
        cal_features = build_deployable_features(cal_x, cal_precision)
        cal_scores = self.failure_model.predict_proba(cal_features)[:, 1]
        self.risk_controller.fit(cal_scores, cal_errors)
        return self

    def predict_precision_sensitivity(self, binary_observables: np.ndarray) -> np.ndarray:
        x = np.asarray(binary_observables, dtype=float)
        was_vector = x.ndim == 1
        if was_vector:
            x = x[None, :]
        prediction = self.sensitivity_model.predict(x)
        return prediction[0] if was_vector else prediction

    def predict_failure_probability(self, binary_observables: np.ndarray) -> np.ndarray | float:
        x = np.asarray(binary_observables, dtype=float)
        was_vector = x.ndim == 1
        if was_vector:
            x = x[None, :]
        precision = self.sensitivity_model.predict(x)
        features = build_deployable_features(x, precision)
        score = self.failure_model.predict_proba(features)[:, 1]
        return float(score[0]) if was_vector else score

    def accept(self, binary_observables: np.ndarray) -> np.ndarray | bool:
        score = self.predict_failure_probability(binary_observables)
        if np.ndim(score) == 0:
            return bool(self.risk_controller.accept(np.array([score]))[0])
        return self.risk_controller.accept(np.asarray(score, dtype=float))

    @property
    def threshold(self) -> float:
        if self.risk_controller.result_ is None:
            raise RuntimeError("risk controller is not fitted")
        return float(self.risk_controller.result_.threshold)

    def to_dict(self) -> dict[str, Any]:
        if self.risk_controller.result_ is None:
            raise RuntimeError("risk controller is not fitted")
        return {
            "format_version": 1,
            "binary_observable_order": [
                "sensor_uncertainty",
                "fusion_conflict",
                "binary_entropy",
                "binary_margin",
                "mean_quality",
            ],
            "deployable_feature_order": [
                "sensor_uncertainty",
                "fusion_conflict",
                "predicted_quantization_disagreement",
                "binary_entropy",
                "binary_margin",
                "predicted_precision_flip_probability",
                "mean_quality",
            ],
            "sensitivity_model": self.sensitivity_model.to_dict(),
            "failure_model": self.failure_model.to_dict(),
            "risk_calibration": asdict(self.risk_controller.result_),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QFuseController":
        sensitivity = PrecisionSensitivityModel.from_dict(payload["sensitivity_model"])
        failure = LogisticFailureModel.from_dict(payload["failure_model"])
        risk_payload = payload["risk_calibration"]
        risk = SelectiveRiskController(
            target_risk=float(risk_payload["target_risk"]),
            delta=float(risk_payload["delta"]),
            min_accepted=1,
        )
        risk.result_ = RiskCalibrationResult(**risk_payload)
        return cls(sensitivity, failure, risk)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "QFuseController":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

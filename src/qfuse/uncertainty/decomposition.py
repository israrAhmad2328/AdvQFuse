from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qfuse.math_utils import entropy, js_divergence, margin, sigmoid


@dataclass(slots=True)
class UncertaintyFeatures:
    sensor_uncertainty: float
    fusion_conflict: float
    quantization_disagreement: float
    binary_entropy: float
    binary_margin: float
    precision_label_flip: float
    mean_quality: float

    def as_array(self) -> np.ndarray:
        """Full paired-precision feature vector used for analysis/teacher targets."""
        return np.array(
            [
                self.sensor_uncertainty,
                self.fusion_conflict,
                self.quantization_disagreement,
                self.binary_entropy,
                self.binary_margin,
                self.precision_label_flip,
                self.mean_quality,
            ],
            dtype=float,
        )

    def binary_observable_array(self) -> np.ndarray:
        """Features available after the binary pass, before ternary escalation."""
        return np.array(
            [
                self.sensor_uncertainty,
                self.fusion_conflict,
                self.binary_entropy,
                self.binary_margin,
                self.mean_quality,
            ],
            dtype=float,
        )

    def paired_precision_target(self) -> np.ndarray:
        """Teacher targets obtained from paired binary/ternary calibration runs."""
        return np.array(
            [self.quantization_disagreement, self.precision_label_flip], dtype=float
        )


def build_deployable_features(
    binary_observables: np.ndarray, predicted_precision: np.ndarray
) -> np.ndarray:
    """Combine binary-side observables with distilled precision sensitivity.

    Parameters
    ----------
    binary_observables:
        Array with columns [sensor uncertainty, fusion conflict, binary entropy,
        binary margin, mean quality].
    predicted_precision:
        Array with columns [predicted binary/ternary JS disagreement,
        predicted probability of a precision label flip].

    Returns
    -------
    np.ndarray
        Seven-column feature matrix in the same semantic order as
        :meth:`UncertaintyFeatures.as_array`, but without using a ternary pass.
    """
    x = np.asarray(binary_observables, dtype=float)
    p = np.asarray(predicted_precision, dtype=float)
    was_vector = x.ndim == 1
    if was_vector:
        x = x[None, :]
    if p.ndim == 1:
        p = p[None, :]
    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError("binary_observables must have shape (n, 5)")
    if p.ndim != 2 or p.shape != (len(x), 2):
        raise ValueError("predicted_precision must have shape (n, 2)")
    out = np.column_stack(
        [x[:, 0], x[:, 1], p[:, 0], x[:, 2], x[:, 3], p[:, 1], x[:, 4]]
    )
    return out[0] if was_vector else out


def decompose_uncertainty(
    binary_modalities: list[np.ndarray],
    ternary_modalities: list[np.ndarray] | None,
    qualities: list[float],
    fused_uncertainty: float,
) -> UncertaintyFeatures:
    if not binary_modalities:
        raise ValueError("binary_modalities cannot be empty")
    if len(qualities) != len(binary_modalities):
        raise ValueError("qualities must match the number of binary modalities")
    pairwise = []
    for i in range(len(binary_modalities)):
        for j in range(i + 1, len(binary_modalities)):
            pairwise.append(js_divergence(binary_modalities[i], binary_modalities[j]))
    conflict = float(np.mean(pairwise)) if pairwise else 0.0
    binary_mean = np.mean(np.stack(binary_modalities, axis=0), axis=0)
    if ternary_modalities is not None:
        if len(ternary_modalities) != len(binary_modalities):
            raise ValueError("ternary_modalities must match binary_modalities")
        ternary_mean = np.mean(np.stack(ternary_modalities, axis=0), axis=0)
        q_disagreement = js_divergence(binary_mean, ternary_mean)
        flip = float(np.argmax(binary_mean) != np.argmax(ternary_mean))
    else:
        q_disagreement = 0.0
        flip = 0.0
    return UncertaintyFeatures(
        sensor_uncertainty=float(np.clip(fused_uncertainty, 0.0, 1.0)),
        fusion_conflict=conflict,
        quantization_disagreement=q_disagreement,
        binary_entropy=entropy(binary_mean) / np.log(binary_mean.size),
        binary_margin=margin(binary_mean),
        precision_label_flip=flip,
        mean_quality=float(np.mean(qualities)),
    )


class LogisticFailureModel:
    """Small dependency-free calibrator for a binary event probability."""

    def __init__(self, l2: float = 1e-3) -> None:
        self.l2 = float(l2)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def _prepare(self, x: np.ndarray, fit: bool) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2-D matrix")
        if fit:
            self.mean_ = x.mean(axis=0)
            self.scale_ = x.std(axis=0)
            self.scale_[self.scale_ < 1e-8] = 1.0
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("model is not fitted")
        z = (x - self.mean_) / self.scale_
        return np.column_stack([np.ones(len(z)), z])

    def fit(
        self, x: np.ndarray, y: np.ndarray, lr: float = 0.08, steps: int = 2500
    ) -> "LogisticFailureModel":
        xz = self._prepare(x, fit=True)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) != len(xz):
            raise ValueError("x and y have inconsistent lengths")
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("y must contain binary 0/1 targets")
        w = np.zeros(xz.shape[1], dtype=float)
        for step in range(int(steps)):
            p = sigmoid(xz @ w)
            grad = (xz.T @ (p - y)) / len(y)
            grad[1:] += self.l2 * w[1:]
            eta = lr / np.sqrt(1.0 + step / 250.0)
            w -= eta * grad
        self.coef_ = w
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        xz = self._prepare(x, fit=False)
        p1 = np.asarray(sigmoid(xz @ self.coef_), dtype=float)
        return np.column_stack([1.0 - p1, p1])

    def to_dict(self) -> dict[str, Any]:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None:
            raise RuntimeError("model is not fitted")
        return {
            "l2": self.l2,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "coef": self.coef_.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LogisticFailureModel":
        model = cls(l2=float(payload.get("l2", 1e-3)))
        model.mean_ = np.asarray(payload["mean"], dtype=float)
        model.scale_ = np.asarray(payload["scale"], dtype=float)
        model.coef_ = np.asarray(payload["coef"], dtype=float)
        return model


class PrecisionSensitivityModel:
    """Distill paired-precision disagreement into binary-pass observables.

    The continuous head predicts Jensen-Shannon disagreement with a ridge model.
    The classification head predicts whether binary and ternary labels will flip.
    Neither head requires ternary inference after fitting.
    """

    def __init__(self, l2: float = 1e-2, flip_l2: float = 5e-3) -> None:
        self.l2 = float(l2)
        self.flip_model = LogisticFailureModel(l2=flip_l2)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.disagreement_coef_: np.ndarray | None = None

    def _prepare(self, x: np.ndarray, fit: bool) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2-D matrix")
        if fit:
            self.mean_ = x.mean(axis=0)
            self.scale_ = x.std(axis=0)
            self.scale_[self.scale_ < 1e-8] = 1.0
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("model is not fitted")
        z = (x - self.mean_) / self.scale_
        return np.column_stack([np.ones(len(z)), z])

    def fit(self, x: np.ndarray, targets: np.ndarray) -> "PrecisionSensitivityModel":
        x = np.asarray(x, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if targets.ndim != 2 or targets.shape != (len(x), 2):
            raise ValueError("targets must have shape (n, 2)")
        xz = self._prepare(x, fit=True)
        regularizer = self.l2 * np.eye(xz.shape[1])
        regularizer[0, 0] = 0.0
        self.disagreement_coef_ = np.linalg.solve(
            xz.T @ xz + regularizer, xz.T @ targets[:, 0]
        )
        self.flip_model.fit(x, targets[:, 1])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.disagreement_coef_ is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray(x, dtype=float)
        xz = self._prepare(x, fit=False)
        disagreement = np.clip(xz @ self.disagreement_coef_, 0.0, np.log(2.0))
        flip_probability = self.flip_model.predict_proba(x)[:, 1]
        return np.column_stack([disagreement, flip_probability])

    def to_dict(self) -> dict[str, Any]:
        if self.mean_ is None or self.scale_ is None or self.disagreement_coef_ is None:
            raise RuntimeError("model is not fitted")
        return {
            "l2": self.l2,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "disagreement_coef": self.disagreement_coef_.tolist(),
            "flip_model": self.flip_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PrecisionSensitivityModel":
        model = cls(
            l2=float(payload.get("l2", 1e-2)),
            flip_l2=float(payload.get("flip_model", {}).get("l2", 5e-3)),
        )
        model.mean_ = np.asarray(payload["mean"], dtype=float)
        model.scale_ = np.asarray(payload["scale"], dtype=float)
        model.disagreement_coef_ = np.asarray(
            payload["disagreement_coef"], dtype=float
        )
        model.flip_model = LogisticFailureModel.from_dict(payload["flip_model"])
        return model

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from qfuse.math_utils import entropy, js_divergence, normalize, softmax


@dataclass(slots=True)
class PrecisionPoint:
    """One aligned model evaluation along an ordered precision path.

    `precision` is a monotone numeric coordinate (for example 2, 4, 8, 16).
    At least one of `probabilities` or `logits` must be supplied.
    Hidden and semantic vectors are optional because some serving stacks expose
    outputs only.
    """

    precision: float
    probabilities: np.ndarray | None = None
    logits: np.ndarray | None = None
    hidden: np.ndarray | None = None
    semantic: np.ndarray | None = None
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    energy_j: float = 0.0
    name: str = ""

    def probability_vector(self) -> np.ndarray:
        if self.probabilities is not None:
            p = normalize(np.asarray(self.probabilities, dtype=float).reshape(-1))
        elif self.logits is not None:
            p = softmax(np.asarray(self.logits, dtype=float).reshape(-1))
        else:
            raise ValueError("PrecisionPoint requires probabilities or logits")
        if not np.all(np.isfinite(p)):
            raise ValueError("probabilities contain non-finite values")
        return p

    def logit_vector(self) -> np.ndarray:
        if self.logits is not None:
            return np.asarray(self.logits, dtype=float).reshape(-1)
        # Log probabilities preserve argmax and are sufficient for path checks.
        p = self.probability_vector()
        return np.log(np.clip(p, 1e-12, 1.0))


@dataclass(slots=True)
class PrecisionPathFeatures:
    n_points: int
    adjacent_js_mean: float
    adjacent_js_max: float
    js_total_variation: float
    adjacent_l1_mean: float
    adjacent_l1_max: float
    l1_total_variation: float
    path_curvature_l1: float
    rank_flip_count: int
    reference_disagreement_count: int
    entropy_mean: float
    entropy_std: float
    entropy_range: float
    margin_min: float
    margin_reference: float
    margin_erosion: float
    hidden_cosine_drift_mean: float
    hidden_cosine_drift_max: float
    semantic_cosine_drift_mean: float
    semantic_cosine_drift_max: float
    precision_span: float
    latency_span_ms: float
    memory_span_mb: float
    energy_span_j: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class PathInvarianceCertificate:
    certified: bool
    reference_class: int
    reference_margin: float
    max_logit_deviation: float
    slack: float
    n_points: int


def _top_margin(p: np.ndarray) -> float:
    s = np.sort(np.asarray(p, dtype=float).reshape(-1))[::-1]
    if len(s) < 2:
        return float("inf")
    return float(s[0] - s[1])


def _cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("vectors must have equal shape")
    denom = max(float(np.linalg.norm(x) * np.linalg.norm(y)), eps)
    return float(1.0 - np.clip(float(np.dot(x, y)) / denom, -1.0, 1.0))


def _optional_adjacent_drift(points: Sequence[PrecisionPoint], attr: str) -> tuple[float, float]:
    values: list[float] = []
    for left, right in zip(points[:-1], points[1:], strict=True):
        a = getattr(left, attr)
        b = getattr(right, attr)
        if a is not None and b is not None:
            values.append(_cosine_distance(a, b))
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.max(values))


def validate_precision_path(points: Iterable[PrecisionPoint]) -> list[PrecisionPoint]:
    ordered = sorted(list(points), key=lambda point: float(point.precision))
    if len(ordered) < 2:
        raise ValueError("a precision path requires at least two points")
    precisions = np.asarray([point.precision for point in ordered], dtype=float)
    if np.any(~np.isfinite(precisions)):
        raise ValueError("precision coordinates must be finite")
    if np.any(np.diff(precisions) <= 0):
        raise ValueError("precision coordinates must be unique and strictly increasing")
    dimensions = {point.probability_vector().shape[0] for point in ordered}
    if len(dimensions) != 1:
        raise ValueError("all path points must use the same answer space")
    return ordered


def extract_precision_path_features(points: Iterable[PrecisionPoint]) -> PrecisionPathFeatures:
    ordered = validate_precision_path(points)
    probs = np.stack([point.probability_vector() for point in ordered])
    adjacent_js = np.asarray(
        [js_divergence(probs[i], probs[i + 1]) for i in range(len(probs) - 1)], dtype=float
    )
    adjacent_l1 = np.sum(np.abs(np.diff(probs, axis=0)), axis=1)
    curvature = (
        float(np.sum(np.abs(probs[2:] - 2.0 * probs[1:-1] + probs[:-2])))
        if len(probs) >= 3
        else 0.0
    )
    ranks = np.argmax(probs, axis=1)
    reference_class = int(ranks[-1])
    margins = np.asarray([_top_margin(p) for p in probs], dtype=float)
    entropies = np.asarray([entropy(p) for p in probs], dtype=float)
    hidden_mean, hidden_max = _optional_adjacent_drift(ordered, "hidden")
    semantic_mean, semantic_max = _optional_adjacent_drift(ordered, "semantic")

    latencies = np.asarray([point.latency_ms for point in ordered], dtype=float)
    memories = np.asarray([point.memory_mb for point in ordered], dtype=float)
    energies = np.asarray([point.energy_j for point in ordered], dtype=float)
    precisions = np.asarray([point.precision for point in ordered], dtype=float)

    return PrecisionPathFeatures(
        n_points=len(ordered),
        adjacent_js_mean=float(adjacent_js.mean()),
        adjacent_js_max=float(adjacent_js.max()),
        js_total_variation=float(adjacent_js.sum()),
        adjacent_l1_mean=float(adjacent_l1.mean()),
        adjacent_l1_max=float(adjacent_l1.max()),
        l1_total_variation=float(adjacent_l1.sum()),
        path_curvature_l1=curvature,
        rank_flip_count=int(np.sum(ranks[1:] != ranks[:-1])),
        reference_disagreement_count=int(np.sum(ranks[:-1] != reference_class)),
        entropy_mean=float(entropies.mean()),
        entropy_std=float(entropies.std()),
        entropy_range=float(entropies.max() - entropies.min()),
        margin_min=float(margins.min()),
        margin_reference=float(margins[-1]),
        margin_erosion=float(max(0.0, margins[-1] - margins.min())),
        hidden_cosine_drift_mean=hidden_mean,
        hidden_cosine_drift_max=hidden_max,
        semantic_cosine_drift_mean=semantic_mean,
        semantic_cosine_drift_max=semantic_max,
        precision_span=float(precisions.max() - precisions.min()),
        latency_span_ms=float(latencies.max() - latencies.min()),
        memory_span_mb=float(memories.max() - memories.min()),
        energy_span_j=float(energies.max() - energies.min()),
    )


def certify_observed_path_invariance(
    points: Iterable[PrecisionPoint], reference_index: int = -1
) -> PathInvarianceCertificate:
    """Certify argmax invariance along the observed precision path.

    This is *not* an input-space adversarial certificate. It applies only to the
    supplied aligned precision points. The sufficient condition is
    max_k ||z_k-z_ref||_inf < gamma_ref / 2.
    """

    ordered = validate_precision_path(points)
    logits = np.stack([point.logit_vector() for point in ordered])
    ref = logits[reference_index]
    order = np.argsort(ref)[::-1]
    reference_class = int(order[0])
    margin = float(ref[order[0]] - ref[order[1]]) if ref.size >= 2 else float("inf")
    deviations = np.max(np.abs(logits - ref[None, :]), axis=1)
    max_deviation = float(np.max(deviations))
    slack = float(margin / 2.0 - max_deviation)
    certified = bool(max_deviation < margin / 2.0)
    return PathInvarianceCertificate(
        certified=certified,
        reference_class=reference_class,
        reference_margin=margin,
        max_logit_deviation=max_deviation,
        slack=slack,
        n_points=len(ordered),
    )

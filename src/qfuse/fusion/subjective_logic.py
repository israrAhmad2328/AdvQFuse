from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qfuse.math_utils import js_divergence, normalize


@dataclass(slots=True)
class Opinion:
    belief: np.ndarray
    uncertainty: float
    base_rate: np.ndarray

    def expected_probability(self) -> np.ndarray:
        return normalize(self.belief + self.uncertainty * self.base_rate)


def probability_to_opinion(
    probabilities: np.ndarray,
    reliability: float,
    evidence_strength: float = 18.0,
) -> Opinion:
    p = normalize(np.asarray(probabilities, dtype=float))
    k = p.size
    reliability = float(np.clip(reliability, 0.0, 1.0))
    evidence = evidence_strength * reliability * p
    strength = float(evidence.sum() + k)
    belief = evidence / strength
    uncertainty = k / strength
    base = np.full(k, 1.0 / k)
    return Opinion(belief=belief, uncertainty=uncertainty, base_rate=base)


def discount(opinion: Opinion, reliability: float) -> Opinion:
    r = float(np.clip(reliability, 0.0, 1.0))
    belief = r * opinion.belief
    uncertainty = 1.0 - r + r * opinion.uncertainty
    return Opinion(belief=belief, uncertainty=float(uncertainty), base_rate=opinion.base_rate)


def cumulative_fuse(a: Opinion, b: Opinion, eps: float = 1e-12) -> Opinion:
    if a.belief.shape != b.belief.shape:
        raise ValueError("opinions must have the same number of classes")
    denom = a.uncertainty + b.uncertainty - a.uncertainty * b.uncertainty
    denom = max(float(denom), eps)
    belief = (a.belief * b.uncertainty + b.belief * a.uncertainty) / denom
    uncertainty = (a.uncertainty * b.uncertainty) / denom
    base = normalize(0.5 * (a.base_rate + b.base_rate))
    belief_mass = float(belief.sum())
    if belief_mass + uncertainty > 1.0 + 1e-6:
        belief = belief / max(belief_mass, eps) * max(1.0 - uncertainty, 0.0)
    return Opinion(belief=belief, uncertainty=float(uncertainty), base_rate=base)


def conflict_aware_reliabilities(
    probability_list: list[np.ndarray],
    quality_list: list[float],
    conflict_temperature: float = 3.0,
) -> np.ndarray:
    if len(probability_list) != len(quality_list):
        raise ValueError("probability_list and quality_list must have equal length")
    n = len(probability_list)
    if n == 0:
        raise ValueError("at least one modality is required")
    if n == 1:
        return np.array([float(np.clip(quality_list[0], 0.0, 1.0))])
    conflicts = np.zeros(n, dtype=float)
    for i in range(n):
        vals = [js_divergence(probability_list[i], probability_list[j]) for j in range(n) if j != i]
        conflicts[i] = float(np.mean(vals))
    quality = np.clip(np.asarray(quality_list, dtype=float), 0.0, 1.0)
    reliability = quality * np.exp(-float(conflict_temperature) * conflicts)
    return np.clip(reliability, 0.0, 1.0)


def fuse_modalities(
    probability_list: list[np.ndarray],
    quality_list: list[float],
    evidence_strength: float = 18.0,
    conflict_temperature: float = 3.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    reliabilities = conflict_aware_reliabilities(
        probability_list, quality_list, conflict_temperature=conflict_temperature
    )
    opinions = [
        probability_to_opinion(p, r, evidence_strength=evidence_strength)
        for p, r in zip(probability_list, reliabilities, strict=True)
    ]
    fused = opinions[0]
    for op in opinions[1:]:
        fused = cumulative_fuse(fused, op)
    return fused.expected_probability(), fused.uncertainty, reliabilities

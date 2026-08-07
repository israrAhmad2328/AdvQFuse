from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qfuse.fusion import fuse_modalities
from qfuse.math_utils import softmax
from qfuse.uncertainty import decompose_uncertainty


@dataclass(slots=True)
class SyntheticSample:
    label: int
    qualities: np.ndarray
    binary_modalities: list[np.ndarray]
    ternary_modalities: list[np.ndarray]
    binary_fused: np.ndarray
    ternary_fused: np.ndarray
    fused_uncertainty: float
    binary_observables: np.ndarray
    paired_precision_targets: np.ndarray


def _draw_modality_distribution(
    label: int,
    classes: int,
    quality: float,
    precision_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    logits = rng.normal(0.0, 0.55 + precision_noise, size=classes)
    signal = 1.0 + 4.5 * quality
    logits[label] += signal
    if rng.random() < (1.0 - quality) * (0.20 + precision_noise):
        wrong = int(rng.integers(0, classes - 1))
        if wrong >= label:
            wrong += 1
        logits[wrong] += 2.0 + 2.0 * (1.0 - quality)
    return softmax(logits)


def generate_synthetic_dataset(
    n: int = 3000,
    classes: int = 6,
    modalities: int = 3,
    seed: int = 42,
    evidence_strength: float = 18.0,
    conflict_temperature: float = 3.0,
) -> list[SyntheticSample]:
    rng = np.random.default_rng(seed)
    samples: list[SyntheticSample] = []
    for _ in range(n):
        label = int(rng.integers(0, classes))
        base_quality = float(rng.beta(4.0, 2.0))
        qualities = np.clip(base_quality + rng.normal(0.0, 0.18, size=modalities), 0.05, 1.0)
        if rng.random() < 0.16:
            qualities[int(rng.integers(0, modalities))] *= 0.15
        binary = [
            _draw_modality_distribution(label, classes, float(q), precision_noise=0.72, rng=rng)
            for q in qualities
        ]
        ternary = [
            _draw_modality_distribution(label, classes, float(min(1.0, q + 0.08)), precision_noise=0.36, rng=rng)
            for q in qualities
        ]
        p_bin, u_bin, _ = fuse_modalities(
            binary,
            qualities.tolist(),
            evidence_strength=evidence_strength,
            conflict_temperature=conflict_temperature,
        )
        p_ter, _, _ = fuse_modalities(
            ternary,
            qualities.tolist(),
            evidence_strength=evidence_strength,
            conflict_temperature=conflict_temperature,
        )
        paired_features = decompose_uncertainty(
            binary_modalities=binary,
            ternary_modalities=ternary,
            qualities=qualities.tolist(),
            fused_uncertainty=u_bin,
        )
        binary_features = decompose_uncertainty(
            binary_modalities=binary,
            ternary_modalities=None,
            qualities=qualities.tolist(),
            fused_uncertainty=u_bin,
        )
        samples.append(
            SyntheticSample(
                label=label,
                qualities=qualities,
                binary_modalities=binary,
                ternary_modalities=ternary,
                binary_fused=p_bin,
                ternary_fused=p_ter,
                fused_uncertainty=u_bin,
                binary_observables=binary_features.binary_observable_array(),
                paired_precision_targets=paired_features.paired_precision_target(),
            )
        )
    return samples

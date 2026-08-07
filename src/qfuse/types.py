from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Action(str, Enum):
    ACCEPT_BINARY = "accept_binary"
    REPERCEIVE = "reperceive"
    ESCALATE_TERNARY = "escalate_ternary"
    ABSTAIN = "abstain"


@dataclass(slots=True)
class ModelPrediction:
    probabilities: np.ndarray
    quality: float = 1.0
    latency_ms: float | None = None
    energy_j: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        p = np.asarray(self.probabilities, dtype=float)
        if p.ndim != 1 or p.size < 2:
            raise ValueError("probabilities must be a 1-D array with at least two classes")
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("probabilities must be finite and non-negative")
        total = float(p.sum())
        if total <= 0:
            raise ValueError("probabilities must have positive mass")
        self.probabilities = p / total
        self.quality = float(np.clip(self.quality, 0.0, 1.0))

    @property
    def label(self) -> int:
        return int(np.argmax(self.probabilities))

    @property
    def confidence(self) -> float:
        return float(np.max(self.probabilities))


@dataclass(slots=True)
class PolicyDecision:
    action: Action
    failure_probability: float
    estimated_cost: float
    reason: str

from __future__ import annotations

from dataclasses import dataclass

from qfuse.types import Action, PolicyDecision


@dataclass(slots=True)
class PolicyCosts:
    binary: float = 1.0
    reperception: float = 0.8
    ternary: float = 2.2
    abstain: float = 0.0


class ProgressivePrecisionPolicy:
    def __init__(
        self,
        acceptance_threshold: float,
        costs: PolicyCosts | None = None,
        reperception_quality_threshold: float = 0.45,
    ) -> None:
        self.acceptance_threshold = float(acceptance_threshold)
        self.costs = costs or PolicyCosts()
        self.reperception_quality_threshold = float(reperception_quality_threshold)

    def decide(
        self,
        failure_probability: float,
        mean_quality: float,
        ternary_available: bool = True,
        reperception_available: bool = True,
        remaining_budget: float = float("inf"),
    ) -> PolicyDecision:
        p = float(min(max(failure_probability, 0.0), 1.0))
        if p <= self.acceptance_threshold:
            return PolicyDecision(
                action=Action.ACCEPT_BINARY,
                failure_probability=p,
                estimated_cost=self.costs.binary,
                reason="calibrated failure score is within the accepted-risk region",
            )
        if (
            reperception_available
            and mean_quality < self.reperception_quality_threshold
            and self.costs.reperception <= remaining_budget
        ):
            return PolicyDecision(
                action=Action.REPERCEIVE,
                failure_probability=p,
                estimated_cost=self.costs.reperception,
                reason="sensor quality is low; acquire a crop, higher-resolution view, or repeated frame",
            )
        if ternary_available and self.costs.ternary <= remaining_budget:
            return PolicyDecision(
                action=Action.ESCALATE_TERNARY,
                failure_probability=p,
                estimated_cost=self.costs.ternary,
                reason="binary prediction is unsafe and higher-precision inference is affordable",
            )
        return PolicyDecision(
            action=Action.ABSTAIN,
            failure_probability=p,
            estimated_cost=self.costs.abstain,
            reason="risk is high and no safe escalation action is available",
        )

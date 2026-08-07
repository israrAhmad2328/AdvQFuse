from __future__ import annotations

import argparse
import json

import numpy as np

from qfuse import QFuseController
from qfuse.clients import BonsaiEndpoint, OpenAICompatibleBonsaiClient
from qfuse.fusion import fuse_modalities
from qfuse.policy import ProgressivePrecisionPolicy
from qfuse.uncertainty import decompose_uncertainty


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a deployable binary-first QFuse decision and optionally execute ternary escalation."
    )
    parser.add_argument("--controller", required=True)
    parser.add_argument(
        "--binary-url", default="http://127.0.0.1:8081/v1/chat/completions"
    )
    parser.add_argument("--ternary-url", default=None)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--qualities", nargs="+", type=float, required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--disable-reperception", action="store_true")
    parser.add_argument("--reperception-quality-threshold", type=float, default=0.45)
    args = parser.parse_args()
    if len(args.images) != len(args.qualities):
        raise SystemExit("--images and --qualities must have the same length")

    controller = QFuseController.load(args.controller)
    binary_client = OpenAICompatibleBonsaiClient(BonsaiEndpoint(args.binary_url))
    binary_predictions = [
        binary_client.classify(args.prompt, [image], args.classes, quality=quality)
        for image, quality in zip(args.images, args.qualities, strict=True)
    ]
    p_bin, u_bin, reliabilities = fuse_modalities(
        [p.probabilities for p in binary_predictions], args.qualities
    )
    binary_features = decompose_uncertainty(
        [p.probabilities for p in binary_predictions], None, args.qualities, u_bin
    ).binary_observable_array()
    sensitivity = controller.predict_precision_sensitivity(binary_features)
    failure_probability = float(controller.predict_failure_probability(binary_features))

    policy = ProgressivePrecisionPolicy(
        acceptance_threshold=controller.threshold,
        reperception_quality_threshold=args.reperception_quality_threshold,
    )
    decision = policy.decide(
        failure_probability=failure_probability,
        mean_quality=float(np.mean(args.qualities)),
        ternary_available=args.ternary_url is not None,
        reperception_available=not args.disable_reperception,
    )

    output = {
        "action": decision.action.value,
        "reason": decision.reason,
        "binary_label": args.classes[int(np.argmax(p_bin))],
        "binary_probabilities": dict(
            zip(args.classes, [float(v) for v in p_bin], strict=True)
        ),
        "failure_probability": failure_probability,
        "calibrated_acceptance_threshold": controller.threshold,
        "predicted_quantization_disagreement": float(sensitivity[0]),
        "predicted_precision_flip_probability": float(sensitivity[1]),
        "modality_reliabilities": reliabilities.tolist(),
        "binary_latency_ms": float(
            sum(p.latency_ms or 0.0 for p in binary_predictions)
        ),
    }

    if decision.action.value == "escalate_ternary" and args.ternary_url:
        ternary_client = OpenAICompatibleBonsaiClient(BonsaiEndpoint(args.ternary_url))
        ternary_predictions = [
            ternary_client.classify(args.prompt, [image], args.classes, quality=quality)
            for image, quality in zip(args.images, args.qualities, strict=True)
        ]
        p_ter, _, _ = fuse_modalities(
            [p.probabilities for p in ternary_predictions], args.qualities
        )
        output["ternary_label"] = args.classes[int(np.argmax(p_ter))]
        output["ternary_probabilities"] = dict(
            zip(args.classes, [float(v) for v in p_ter], strict=True)
        )
        output["ternary_latency_ms"] = float(
            sum(p.latency_ms or 0.0 for p in ternary_predictions)
        )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

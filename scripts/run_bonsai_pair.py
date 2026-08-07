from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from qfuse.clients import BonsaiEndpoint, OpenAICompatibleBonsaiClient
from qfuse.fusion import fuse_modalities
from qfuse.uncertainty import decompose_uncertainty


def _prediction_record(prediction: Any) -> dict[str, Any]:
    return {
        "probabilities": prediction.probabilities.tolist(),
        "quality": float(prediction.quality),
        "latency_ms": prediction.latency_ms,
        "energy_j": prediction.energy_j,
        "metadata": prediction.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query binary and ternary Bonsai servers on one multimodal sample."
    )
    parser.add_argument(
        "--binary-url", default="http://127.0.0.1:8081/v1/chat/completions"
    )
    parser.add_argument(
        "--ternary-url", default="http://127.0.0.1:8082/v1/chat/completions"
    )
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--qualities", nargs="+", type=float, required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--sample-id", default="sample")
    parser.add_argument("--dataset", default="unspecified")
    parser.add_argument("--split", choices=["train", "cal", "test"], default="train")
    parser.add_argument(
        "--true-label",
        default=None,
        help="Ground-truth class name. Required before fitting a controller.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Append the paired record to this JSONL file as well as printing it.",
    )
    args = parser.parse_args()
    if len(args.images) != len(args.qualities):
        raise SystemExit("--images and --qualities must have the same length")
    if args.true_label is not None and args.true_label not in args.classes:
        raise SystemExit("--true-label must be one of --classes")

    binary_client = OpenAICompatibleBonsaiClient(BonsaiEndpoint(args.binary_url))
    ternary_client = OpenAICompatibleBonsaiClient(BonsaiEndpoint(args.ternary_url))
    binary_predictions = []
    ternary_predictions = []
    for image, quality in zip(args.images, args.qualities, strict=True):
        binary_predictions.append(
            binary_client.classify(args.prompt, [image], args.classes, quality=quality)
        )
        ternary_predictions.append(
            ternary_client.classify(args.prompt, [image], args.classes, quality=quality)
        )

    p_bin, u_bin, reliabilities = fuse_modalities(
        [p.probabilities for p in binary_predictions], args.qualities
    )
    p_ter, _, _ = fuse_modalities(
        [p.probabilities for p in ternary_predictions], args.qualities
    )
    binary_features = decompose_uncertainty(
        [p.probabilities for p in binary_predictions],
        None,
        args.qualities,
        u_bin,
    )
    paired_features = decompose_uncertainty(
        [p.probabilities for p in binary_predictions],
        [p.probabilities for p in ternary_predictions],
        args.qualities,
        u_bin,
    )

    record: dict[str, Any] = {
        "sample_id": args.sample_id,
        "dataset": args.dataset,
        "split": args.split,
        "prompt": args.prompt,
        "classes": args.classes,
        "true_label": args.true_label,
        "true_label_index": (
            args.classes.index(args.true_label) if args.true_label is not None else None
        ),
        "images": [str(Path(path)) for path in args.images],
        "qualities": args.qualities,
        "binary_modalities": [_prediction_record(p) for p in binary_predictions],
        "ternary_modalities": [_prediction_record(p) for p in ternary_predictions],
        "binary_fused": p_bin.tolist(),
        "ternary_fused": p_ter.tolist(),
        "modality_reliabilities": reliabilities.tolist(),
        "binary_observables": binary_features.binary_observable_array().tolist(),
        "paired_precision_targets": paired_features.paired_precision_target().tolist(),
        "paired_uncertainty_analysis": {
            "sensor_uncertainty": paired_features.sensor_uncertainty,
            "fusion_conflict": paired_features.fusion_conflict,
            "quantization_disagreement": paired_features.quantization_disagreement,
            "binary_entropy": paired_features.binary_entropy,
            "binary_margin": paired_features.binary_margin,
            "precision_label_flip": paired_features.precision_label_flip,
            "mean_quality": paired_features.mean_quality,
        },
    }
    text = json.dumps(record, ensure_ascii=False)
    if args.output_jsonl:
        output = Path(args.output_jsonl)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

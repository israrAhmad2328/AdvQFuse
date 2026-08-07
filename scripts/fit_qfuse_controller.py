from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from qfuse import QFuseController
from qfuse.calibration import SelectiveRiskController
from qfuse.metrics import accuracy, binary_auroc, selective_risk


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
    if not records:
        raise ValueError("paired JSONL is empty")
    return records


def _arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    missing = [r.get("sample_id", "unknown") for r in records if r.get("true_label_index") is None]
    if missing:
        raise ValueError(f"records missing true labels: {missing[:5]}")
    x = np.asarray([r["binary_observables"] for r in records], dtype=float)
    targets = np.asarray([r["paired_precision_targets"] for r in records], dtype=float)
    y = np.asarray([r["true_label_index"] for r in records], dtype=int)
    p_bin = np.asarray([r["binary_fused"] for r in records], dtype=float)
    p_ter = np.asarray([r["ternary_fused"] for r in records], dtype=float)
    errors = (p_bin.argmax(axis=1) != y).astype(int)
    return x, targets, y, p_bin, p_ter, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate a deployable QFuse controller from paired JSONL records."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--model-output", default="artifacts/qfuse_controller.json")
    parser.add_argument("--metrics-output", default="results/qfuse_real_metrics.json")
    parser.add_argument("--target-risk", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--min-accepted", type=int, default=50)
    args = parser.parse_args()

    records = _read_jsonl(Path(args.input_jsonl))
    class_orders = {tuple(r["classes"]) for r in records}
    if len(class_orders) != 1:
        raise ValueError("all records must use the same ordered class list")
    by_split = {name: [r for r in records if r.get("split") == name] for name in ("train", "cal", "test")}
    if any(not by_split[name] for name in by_split):
        raise ValueError("input must contain non-empty train, cal, and test splits")

    train_x, train_targets, _, _, _, train_errors = _arrays(by_split["train"])
    cal_x, _, _, _, _, cal_errors = _arrays(by_split["cal"])
    test_x, test_targets, test_y, test_bin, test_ter, test_errors = _arrays(by_split["test"])

    controller = QFuseController(
        risk_controller=SelectiveRiskController(
            target_risk=args.target_risk,
            delta=args.delta,
            min_accepted=args.min_accepted,
        )
    ).fit(train_x, train_targets, train_errors, cal_x, cal_errors)

    score = np.asarray(controller.predict_failure_probability(test_x), dtype=float)
    accepted = np.asarray(controller.accept(test_x), dtype=bool)
    binary_pred = test_bin.argmax(axis=1)
    ternary_pred = test_ter.argmax(axis=1)
    progressive_pred = binary_pred.copy()
    progressive_pred[~accepted] = ternary_pred[~accepted]
    predicted_precision = controller.predict_precision_sensitivity(test_x)

    metrics = {
        "status": "real_paired_run_evaluation",
        "input": str(Path(args.input_jsonl)),
        "classes": list(next(iter(class_orders))),
        "split_sizes": {name: len(by_split[name]) for name in by_split},
        "binary_accuracy": accuracy(test_y, binary_pred),
        "ternary_accuracy": accuracy(test_y, ternary_pred),
        "progressive_accuracy": accuracy(test_y, progressive_pred),
        "failure_prediction_auroc": binary_auroc(test_errors, score),
        "precision_flip_prediction_auroc": binary_auroc(
            test_targets[:, 1], predicted_precision[:, 1]
        ),
        "precision_disagreement_mae": float(
            np.mean(np.abs(test_targets[:, 0] - predicted_precision[:, 0]))
        ),
        "accepted_binary_coverage": float(accepted.mean()),
        "accepted_binary_selective_risk": selective_risk(
            test_y, binary_pred, accepted
        ),
        "escalation_rate": float((~accepted).mean()),
        "risk_calibration": asdict(controller.risk_controller.result_),
    }
    model_output = Path(args.model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    controller.save(model_output)
    metrics_output = Path(args.metrics_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved controller to {model_output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from qfuse.benchmark import generate_synthetic_dataset
from qfuse.calibration import SelectiveRiskController
from qfuse.metrics import (
    accuracy,
    binary_auroc,
    expected_calibration_error,
    selective_risk,
)
from qfuse.uncertainty import (
    LogisticFailureModel,
    PrecisionSensitivityModel,
    build_deployable_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic QFuse sanity check.")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument(
        "--target-risk",
        type=float,
        default=0.04,
        help="Target selective risk for accepted binary decisions (default: 0.04).",
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = generate_synthetic_dataset(n=args.n, seed=args.seed)
    binary_x = np.stack([s.binary_observables for s in samples])
    paired_targets = np.stack([s.paired_precision_targets for s in samples])
    y = np.array([s.label for s in samples])
    p_bin = np.stack([s.binary_fused for s in samples])
    p_ter = np.stack([s.ternary_fused for s in samples])
    qualities = np.array([float(np.mean(s.qualities)) for s in samples])

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(samples))
    n_train = int(0.50 * len(idx))
    n_cal = int(0.25 * len(idx))
    train, cal, test = idx[:n_train], idx[n_train : n_train + n_cal], idx[n_train + n_cal :]

    # Paired binary/ternary runs are used only on the fitting split to distill
    # quantization sensitivity into features observable after the binary pass.
    sensitivity = PrecisionSensitivityModel(l2=1e-2, flip_l2=5e-3).fit(
        binary_x[train], paired_targets[train]
    )
    predicted_precision = sensitivity.predict(binary_x)
    deployable_x = build_deployable_features(binary_x, predicted_precision)

    binary_pred = p_bin.argmax(axis=1)
    ternary_pred = p_ter.argmax(axis=1)
    binary_error = (binary_pred != y).astype(int)
    failure_model = LogisticFailureModel(l2=5e-3).fit(
        deployable_x[train], binary_error[train]
    )
    failure_score = failure_model.predict_proba(deployable_x)[:, 1]

    controller = SelectiveRiskController(
        target_risk=args.target_risk, delta=0.05, min_accepted=75
    )
    calibration = controller.fit(failure_score[cal], binary_error[cal])
    accepted_test = controller.accept(failure_score[test])

    progressive_pred = binary_pred[test].copy()
    progressive_pred[~accepted_test] = ternary_pred[test][~accepted_test]

    rows = pd.DataFrame(
        {
            "sample_id": test,
            "label": y[test],
            "binary_pred": binary_pred[test],
            "ternary_pred": ternary_pred[test],
            "failure_score": failure_score[test],
            "predicted_precision_disagreement": predicted_precision[test, 0],
            "predicted_precision_flip_probability": predicted_precision[test, 1],
            "true_precision_disagreement": paired_targets[test, 0],
            "true_precision_flip": paired_targets[test, 1],
            "accepted_binary": accepted_test,
            "mean_quality": qualities[test],
            "progressive_pred": progressive_pred,
        }
    )
    rows.to_csv(out / "synthetic_predictions.csv", index=False)

    metrics = {
        "status": "synthetic_sanity_check_not_paper_evidence",
        "deployment_note": (
            "Ternary outputs are used only to fit the precision-sensitivity teacher; "
            "test-time failure scores use binary-pass observables and distilled predictions."
        ),
        "n_total": len(samples),
        "n_test": len(test),
        "binary_accuracy": accuracy(y[test], binary_pred[test]),
        "ternary_accuracy": accuracy(y[test], ternary_pred[test]),
        "progressive_accuracy": accuracy(y[test], progressive_pred),
        "binary_ece": expected_calibration_error(y[test], p_bin[test]),
        "ternary_ece": expected_calibration_error(y[test], p_ter[test]),
        "failure_prediction_auroc": binary_auroc(binary_error[test], failure_score[test]),
        "precision_flip_prediction_auroc": binary_auroc(
            paired_targets[test, 1], predicted_precision[test, 1]
        ),
        "precision_disagreement_mae": float(
            np.mean(np.abs(predicted_precision[test, 0] - paired_targets[test, 0]))
        ),
        "accepted_binary_coverage": float(accepted_test.mean()),
        "accepted_binary_selective_risk": selective_risk(
            y[test], binary_pred[test], accepted_test
        ),
        "escalation_rate": float((~accepted_test).mean()),
        "normalized_average_cost": float(np.mean(np.where(accepted_test, 1.0, 3.2))),
        "calibration": asdict(calibration),
    }
    (out / "synthetic_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

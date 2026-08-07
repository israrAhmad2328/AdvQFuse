from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qfuse.evaluation import aggregate_results, generate_all_figures


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labelled synthetic AdvQFuse-RS result tables and figures.")
    parser.add_argument("--n-per-cell", type=int, default=45)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--results-dir", default="results/advanced_demo")
    parser.add_argument("--figures-dir", default="figures/advanced_demo")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    datasets = {
        "EarthVQA": 0.79,
        "FloodNet": 0.76,
        "RSVQA-HR": 0.74,
        "UAV-OBB-QA": 0.81,
        "SEN12MS-QA": 0.72,
    }
    question_types = ["count", "existence", "spatial", "relational", "scene", "orientation"]
    attacks = {
        "clean": 0.00,
        "pgd_transfer": 0.12,
        "patch_transfer": 0.16,
        "spsa_blackbox": 0.14,
        "prompt_semantic": 0.10,
        "cross_modal_conflict": 0.18,
    }
    models = {
        "Bonsai-1bit": {"gain": -0.035, "latency": 850, "energy": 13.0},
        "Ternary-1.7bit": {"gain": 0.025, "latency": 1280, "energy": 19.0},
        "AdvQFuse-RS": {"gain": 0.075, "latency": 1040, "energy": 15.2},
    }
    attack_sources = ["Qwen2.5-VL-3B", "LLaVA-1.6-7B", "Bonsai-1bit-SPSA"]
    failure_types = ["counting", "small_object", "spatial_relation", "sensor_conflict", "answer_anchor"]
    rows = []
    sample_counter = 0

    for dataset, base in datasets.items():
        for attack, penalty in attacks.items():
            severities = [0] if attack == "clean" else [1, 2, 3, 4, 5]
            for severity in severities:
                for local_idx in range(args.n_per_cell):
                    sample_id = f"{dataset}_{attack}_{severity}_{local_idx:04d}"
                    qtype = question_types[(local_idx + severity) % len(question_types)]
                    sensor = np.clip(rng.beta(2 + severity, 7), 0, 1)
                    conflict = np.clip(rng.beta(1.6 + severity * 0.7, 6), 0, 1)
                    patch_area = [0.01, 0.025, 0.05, 0.08][local_idx % 4]
                    common = rng.normal(0, 0.035)
                    for model, spec in models.items():
                        quant = np.clip(
                            0.07
                            + (0.13 if model == "Bonsai-1bit" else 0.05)
                            + severity * (0.035 if model == "Bonsai-1bit" else 0.018)
                            + rng.normal(0, 0.035),
                            0,
                            1,
                        )
                        effective_penalty = penalty * severity / 3.0
                        if model == "AdvQFuse-RS":
                            effective_penalty *= 0.53
                        elif model == "Ternary-1.7bit":
                            effective_penalty *= 0.78
                        if attack == "patch_transfer":
                            effective_penalty += patch_area * (1.4 if model == "Bonsai-1bit" else 0.8)
                        difficulty = 0.13 * sensor + 0.18 * conflict + 0.22 * quant
                        p = np.clip(
                            base + spec["gain"] - effective_penalty - difficulty + common + rng.normal(0, 0.012),
                            0.03,
                            0.98,
                        )
                        correct = int(rng.random() < p)
                        model_bias = 0.10 if model == "Bonsai-1bit" else (0.04 if model == "Ternary-1.7bit" else -0.02)
                        failure_score = np.clip(
                            1.0 - p + model_bias + rng.normal(0, 0.045),
                            0,
                            1,
                        )
                        # Confidence is intentionally mildly overconfident under attack,
                        # reproducing a realistic calibration failure without random step functions.
                        attack_overconfidence = 0.08 * severity / 5.0 if attack != "clean" else 0.0
                        confidence = np.clip(1.0 - failure_score + attack_overconfidence + rng.normal(0, 0.035), 0.03, 0.99)
                        if model == "AdvQFuse-RS":
                            if failure_score < 0.28:
                                action = "accept_binary"
                                latency = spec["latency"] * 0.78
                                energy = spec["energy"] * 0.72
                            elif sensor > 0.55:
                                action = "reobserve"
                                latency = spec["latency"] * 1.12
                                energy = spec["energy"] * 0.95
                            elif failure_score < 0.68:
                                action = "escalate_ternary"
                                latency = spec["latency"] * 1.58
                                energy = spec["energy"] * 1.36
                            else:
                                action = "abstain"
                                latency = spec["latency"] * 0.82
                                energy = spec["energy"] * 0.77
                            accepted = int(action != "abstain")
                        else:
                            action = "always_predict"
                            accepted = 1
                            latency = spec["latency"]
                            energy = spec["energy"]
                        latency *= rng.lognormal(0, 0.06)
                        energy *= rng.lognormal(0, 0.05)
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "dataset": dataset,
                                "question_type": qtype,
                                "attack_family": attack,
                                "attack_source": None if attack == "clean" else attack_sources[(local_idx + severity) % len(attack_sources)],
                                "severity": severity,
                                "patch_area": patch_area if attack == "patch_transfer" else 0.0,
                                "model": model,
                                "correct": correct,
                                "confidence": confidence,
                                "failure_score": failure_score,
                                "accepted": accepted,
                                "policy_action": action,
                                "latency_ms": latency,
                                "energy_j": energy,
                                "sensor_uncertainty": sensor,
                                "semantic_conflict": conflict,
                                "quant_disagreement": quant,
                                "attack_success": int(attack != "clean" and not correct),
                                "failure_type": failure_types[(local_idx + severity) % len(failure_types)],
                                "status": "synthetic_demo_not_paper_evidence",
                            }
                        )
                    sample_counter += 1

    df = pd.DataFrame(rows)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "advrs_synthetic_predictions.csv", index=False)
    aggregate_results(df).to_csv(results_dir / "advrs_synthetic_aggregate.csv", index=False)
    generate_all_figures(df, args.figures_dir)
    print(f"Created {len(df):,} model predictions and advanced figures in {args.figures_dir}")
    print("These are synthetic visualization checks, not scientific results.")


if __name__ == "__main__":
    main()

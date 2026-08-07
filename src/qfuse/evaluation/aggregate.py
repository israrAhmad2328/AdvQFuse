from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "model",
        "attack_family",
        "severity",
        "correct",
        "latency_ms",
        "energy_j",
        "accepted",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing result columns: {sorted(missing)}")
    grouped = (
        df.groupby(["dataset", "model", "attack_family", "severity"], dropna=False)
        .agg(
            accuracy=("correct", "mean"),
            n=("correct", "size"),
            latency_ms=("latency_ms", "mean"),
            energy_j=("energy_j", "mean"),
            coverage=("accepted", "mean"),
            mean_failure_score=("failure_score", "mean"),
            mean_quant_disagreement=("quant_disagreement", "mean"),
        )
        .reset_index()
    )
    return grouped


def bootstrap_accuracy_difference(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    group_columns: tuple[str, ...] = ("dataset", "attack_family"),
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired bootstrap difference in accuracy where sample IDs overlap."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(list(group_columns)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        a = group[group.model == model_a][["sample_id", "correct"]].rename(
            columns={"correct": "a"}
        )
        b = group[group.model == model_b][["sample_id", "correct"]].rename(
            columns={"correct": "b"}
        )
        paired = a.merge(b, on="sample_id")
        if len(paired) < 10:
            continue
        values = paired[["a", "b"]].to_numpy(dtype=float)
        diffs = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            idx = rng.integers(0, len(values), len(values))
            draw = values[idx]
            diffs[i] = float(np.mean(draw[:, 0] - draw[:, 1]))
        row = {name: value for name, value in zip(group_columns, keys)}
        row.update(
            {
                "difference": float(np.mean(values[:, 0] - values[:, 1])),
                "ci_low": float(np.quantile(diffs, 0.025)),
                "ci_high": float(np.quantile(diffs, 0.975)),
                "n": len(values),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)

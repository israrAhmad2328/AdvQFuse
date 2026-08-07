from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .aggregate import aggregate_results, bootstrap_accuracy_difference


WATERMARK = "SYNTHETIC DEMO — REPLACE WITH LOGGED REAL EXPERIMENTS"


def _save(fig, path: Path) -> None:
    fig.text(0.5, 0.008, WATERMARK, ha="center", va="bottom", fontsize=8, alpha=0.55)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _risk_coverage_curve(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = group.sort_values("failure_score")
    errors = 1.0 - ordered["correct"].to_numpy(dtype=float)
    n = len(errors)
    coverage = np.arange(1, n + 1, dtype=float) / max(n, 1)
    risk = np.cumsum(errors) / np.arange(1, n + 1)
    return coverage, risk


def plot_attack_severity(df: pd.DataFrame, out: Path) -> None:
    agg = aggregate_results(df)
    subset = agg[agg.attack_family.isin(["patch_transfer", "pgd_transfer", "cross_modal_conflict"])]
    models = list(dict.fromkeys(subset.model.tolist()))
    attacks = list(dict.fromkeys(subset.attack_family.tolist()))
    fig, axes = plt.subplots(1, len(attacks), figsize=(5.2 * len(attacks), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, attack in zip(axes, attacks):
        part = subset[subset.attack_family == attack]
        for model in models:
            curve = part[part.model == model].groupby("severity")["accuracy"].mean()
            ax.plot(curve.index, curve.values, marker="o", label=model)
        ax.set_title(attack.replace("_", " ").title())
        ax.set_xlabel("Attack severity")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(loc="best")
    fig.suptitle("Robustness degradation across adversarial severity")
    _save(fig, out / "fig01_attack_severity_curves.png")


def plot_risk_coverage(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for model, group in df[df.attack_family != "clean"].groupby("model"):
        coverage, risk = _risk_coverage_curve(group)
        ax.plot(coverage, risk, label=model)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title("Risk–coverage behavior under all adversarial conditions")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, out / "fig02_risk_coverage.png")


def plot_reliability(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    bins = np.linspace(0, 1, 11)
    for model, group in df.groupby("model"):
        conf = group["confidence"].to_numpy(float)
        corr = group["correct"].to_numpy(float)
        ids = np.clip(np.digitize(conf, bins) - 1, 0, len(bins) - 2)
        xs, ys = [], []
        for i in range(len(bins) - 1):
            mask = ids == i
            if mask.any():
                xs.append(float(conf[mask].mean()))
                ys.append(float(corr[mask].mean()))
        ax.plot(xs, ys, marker="o", label=model)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, out / "fig03_reliability_diagram.png")


def plot_transfer_matrix(df: pd.DataFrame, out: Path) -> None:
    part = df[df.attack_source.notna() & (df.attack_family != "clean")]
    matrix = part.pivot_table(
        index="attack_source", columns="model", values="attack_success", aggfunc="mean"
    ).fillna(0.0)
    fig, ax = plt.subplots(figsize=(8.0, 5.1))
    im = ax.imshow(matrix.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title("Cross-model adversarial transfer matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Attack success rate")
    _save(fig, out / "fig04_attack_transfer_matrix.png")


def plot_pareto(df: pd.DataFrame, out: Path) -> None:
    agg = (
        df.groupby("model")
        .agg(accuracy=("correct", "mean"), latency=("latency_ms", "mean"), energy=("energy_j", "mean"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.0, 5.3))
    sizes = 180 + 900 * (agg.energy - agg.energy.min()) / max(agg.energy.max() - agg.energy.min(), 1e-9)
    ax.scatter(agg.latency, agg.accuracy, s=sizes, alpha=0.75)
    for _, row in agg.iterrows():
        ax.annotate(row.model, (row.latency, row.accuracy), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Mean latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy–latency–energy Pareto view (marker area = energy)")
    ax.grid(True, alpha=0.25)
    _save(fig, out / "fig05_pareto_accuracy_latency_energy.png")


def plot_uncertainty_decomposition(df: pd.DataFrame, out: Path) -> None:
    columns = ["sensor_uncertainty", "semantic_conflict", "quant_disagreement"]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), sharey=True)
    for ax, col in zip(axes, columns):
        correct = df[df.correct == 1][col].to_numpy(float)
        wrong = df[df.correct == 0][col].to_numpy(float)
        ax.violinplot([correct, wrong], positions=[1, 2], showmeans=True, showextrema=False)
        ax.set_xticks([1, 2], ["Correct", "Failure"])
        ax.set_title(col.replace("_", " ").title())
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Uncertainty value")
    fig.suptitle("Failure separation by uncertainty source")
    _save(fig, out / "fig06_uncertainty_decomposition.png")


def plot_dataset_radar(df: pd.DataFrame, out: Path) -> None:
    clean = df[df.attack_family == "clean"]
    robust = df[df.attack_family != "clean"]
    datasets = sorted(df.dataset.unique())
    models = list(dict.fromkeys(df.model.tolist()))
    metrics: dict[str, list[float]] = {}
    for model in models:
        vals = []
        for dataset in datasets:
            c = clean[(clean.model == model) & (clean.dataset == dataset)].correct.mean()
            r = robust[(robust.model == model) & (robust.dataset == dataset)].correct.mean()
            vals.append(float(np.nanmean([c, r])))
        metrics[model] = vals
    angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False)
    angles = np.r_[angles, angles[0]]
    fig = plt.figure(figsize=(7.2, 6.5))
    ax = fig.add_subplot(111, polar=True)
    for model, vals in metrics.items():
        closed = np.r_[vals, vals[0]]
        ax.plot(angles, closed, marker="o", label=model)
        ax.fill(angles, closed, alpha=0.08)
    ax.set_xticks(angles[:-1], datasets)
    ax.set_ylim(0, 1)
    ax.set_title("Cross-dataset clean-plus-robust performance profile", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    _save(fig, out / "fig07_cross_dataset_radar.png")


def plot_policy_flow(df: pd.DataFrame, out: Path) -> None:
    q = df[df.model == "AdvQFuse-RS"].copy()
    counts = q.policy_action.value_counts().reindex(
        ["accept_binary", "reobserve", "escalate_ternary", "abstain"], fill_value=0
    )
    outcomes = q.groupby(["policy_action", "correct"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    left_x, right_x = 0.15, 0.85
    y_positions = np.linspace(0.85, 0.15, len(counts))
    total = max(counts.sum(), 1)
    for y, (action, count) in zip(y_positions, counts.items()):
        ax.text(left_x, y, f"{action}\n{count:,}", ha="center", va="center", bbox={"boxstyle": "round", "alpha": 0.12})
        good = int(outcomes.loc[action, 1]) if action in outcomes.index and 1 in outcomes.columns else 0
        bad = int(outcomes.loc[action, 0]) if action in outcomes.index and 0 in outcomes.columns else 0
        width = 1.0 + 18.0 * count / total
        good_width = width * good / max(count, 1)
        bad_width = width * bad / max(count, 1)
        if good_width > 0:
            ax.plot([left_x + 0.08, right_x - 0.08], [y, 0.65], linewidth=good_width, alpha=0.35)
        if bad_width > 0:
            ax.plot([left_x + 0.08, right_x - 0.08], [y, 0.35], linewidth=bad_width, alpha=0.35, linestyle="--")
    ax.text(right_x, 0.65, "Correct", ha="center", va="center", bbox={"boxstyle": "round", "alpha": 0.12})
    ax.text(right_x, 0.35, "Failure", ha="center", va="center", bbox={"boxstyle": "round", "alpha": 0.12})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Progressive policy flow and outcome routing")
    _save(fig, out / "fig08_policy_flow.png")


def plot_robustness_surface(df: pd.DataFrame, out: Path) -> None:
    part = df[(df.attack_family == "patch_transfer") & (df.model == "AdvQFuse-RS")]
    surface = part.pivot_table(index="patch_area", columns="severity", values="correct", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    im = ax.imshow(surface.to_numpy(), aspect="auto", origin="lower")
    ax.set_xticks(range(len(surface.columns)), surface.columns)
    ax.set_yticks(range(len(surface.index)), [f"{100*x:.1f}%" for x in surface.index])
    ax.set_xlabel("Attack severity")
    ax.set_ylabel("Patch area")
    ax.set_title("Robustness surface over patch area and severity")
    for i in range(surface.shape[0]):
        for j in range(surface.shape[1]):
            ax.text(j, i, f"{surface.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Accuracy")
    _save(fig, out / "fig09_patch_robustness_surface.png")


def plot_forest(df: pd.DataFrame, out: Path) -> None:
    forest = bootstrap_accuracy_difference(df, "AdvQFuse-RS", "Bonsai-1bit", n_bootstrap=800)
    forest = forest.sort_values("difference")
    labels = [f"{r.dataset} / {r.attack_family}" for r in forest.itertuples()]
    y = np.arange(len(forest))
    fig, ax = plt.subplots(figsize=(9.0, max(5.0, 0.34 * len(forest))))
    xerr = np.vstack([forest.difference - forest.ci_low, forest.ci_high - forest.difference])
    ax.errorbar(forest.difference, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Accuracy difference: AdvQFuse-RS minus Bonsai-1bit")
    ax.set_title("Paired bootstrap effect sizes with 95% confidence intervals")
    ax.grid(True, axis="x", alpha=0.25)
    _save(fig, out / "fig10_bootstrap_forest.png")


def plot_failure_taxonomy(df: pd.DataFrame, out: Path) -> None:
    failures = df[df.correct == 0]
    table = failures.pivot_table(index="failure_type", columns="model", values="sample_id", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    bottom = np.zeros(len(table.index))
    for model in table.columns:
        vals = table[model].to_numpy()
        ax.bar(table.index, vals, bottom=bottom, label=model)
        bottom += vals
    ax.set_ylabel("Failure count")
    ax.set_title("Failure taxonomy by model")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    _save(fig, out / "fig11_failure_taxonomy.png")


def plot_feature_embedding(df: pd.DataFrame, out: Path) -> None:
    features = df[["sensor_uncertainty", "semantic_conflict", "quant_disagreement", "failure_score"]].to_numpy(float)
    features = (features - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1e-8)
    _, _, vt = np.linalg.svd(features, full_matrices=False)
    embedding = features @ vt[:2].T
    fig, ax = plt.subplots(figsize=(7.3, 5.6))
    for label, group_idx in [("Correct", df.correct.to_numpy() == 1), ("Failure", df.correct.to_numpy() == 0)]:
        ax.scatter(embedding[group_idx, 0], embedding[group_idx, 1], s=12, alpha=0.45, label=label)
    ax.set_xlabel("Uncertainty principal component 1")
    ax.set_ylabel("Uncertainty principal component 2")
    ax.set_title("Uncertainty-space embedding of successes and failures")
    ax.legend()
    ax.grid(True, alpha=0.2)
    _save(fig, out / "fig12_uncertainty_embedding.png")


def generate_all_figures(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    functions = [
        plot_attack_severity,
        plot_risk_coverage,
        plot_reliability,
        plot_transfer_matrix,
        plot_pareto,
        plot_uncertainty_decomposition,
        plot_dataset_radar,
        plot_policy_flow,
        plot_robustness_surface,
        plot_forest,
        plot_failure_taxonomy,
        plot_feature_embedding,
    ]
    for fn in functions:
        fn(df, out)
    return sorted(out.glob("*.png"))

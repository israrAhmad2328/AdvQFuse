from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aggregate import bootstrap_accuracy_difference


WATERMARK = "SYNTHETIC DEMO - REPLACE WITH LOGGED REAL EXPERIMENTS"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.5, 0.004, WATERMARK, ha="center", va="bottom", fontsize=7.5, alpha=0.55)
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _ece(group: pd.DataFrame, n_bins: int = 12) -> float:
    conf = group["confidence"].to_numpy(float)
    corr = group["correct"].to_numpy(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        mask = ids == i
        if mask.any():
            ece += float(mask.mean()) * abs(float(conf[mask].mean()) - float(corr[mask].mean()))
    return float(ece)


def _risk_coverage(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = group.sort_values("failure_score")
    errors = 1.0 - ordered["correct"].to_numpy(float)
    n = max(len(errors), 1)
    coverage = np.arange(1, len(errors) + 1) / n
    risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return coverage, risk


def _heatmap(ax: plt.Axes, table: pd.DataFrame, title: str, fmt: str = ".2f") -> None:
    values = table.to_numpy(float)
    im = ax.imshow(values, aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)), table.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(table.index)), table.index)
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_results_dashboard(df: pd.DataFrame, out: Path) -> None:
    models = list(dict.fromkeys(df["model"].tolist()))
    clean = df[df.attack_family == "clean"]
    adv = df[df.attack_family != "clean"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))

    x = np.arange(len(models))
    width = 0.36
    clean_acc = [clean[clean.model == m].correct.mean() for m in models]
    robust_acc = [adv[adv.model == m].correct.mean() for m in models]
    axes[0, 0].bar(x - width / 2, clean_acc, width, label="Clean")
    axes[0, 0].bar(x + width / 2, robust_acc, width, label="Adversarial")
    axes[0, 0].set_xticks(x, models, rotation=20, ha="right")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Clean and adversarial accuracy")
    axes[0, 0].legend()

    attack_asr = (
        adv.groupby(["attack_family", "model"])["attack_success"].mean().unstack().fillna(0)
    )
    attack_asr.plot(kind="bar", ax=axes[0, 1])
    axes[0, 1].set_ylabel("Attack success rate")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Attack success by family")
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].legend(fontsize=8)

    ece = pd.Series({m: _ece(adv[adv.model == m]) for m in models})
    axes[0, 2].bar(ece.index, ece.values)
    axes[0, 2].set_ylabel("Expected calibration error")
    axes[0, 2].set_title("Calibration under attack")
    axes[0, 2].tick_params(axis="x", rotation=20)

    q = df[df.model == "AdvQFuse-RS"]
    actions = q.policy_action.value_counts(normalize=True).reindex(
        ["accept_binary", "reobserve", "escalate_ternary", "abstain"], fill_value=0
    )
    axes[1, 0].bar(actions.index, actions.values)
    axes[1, 0].set_ylabel("Fraction of samples")
    axes[1, 0].set_title("Progressive policy utilization")
    axes[1, 0].tick_params(axis="x", rotation=25)

    pareto = df.groupby("model").agg(
        accuracy=("correct", "mean"), latency=("latency_ms", "mean"), energy=("energy_j", "mean")
    )
    sizes = 170 + 700 * (pareto.energy - pareto.energy.min()) / max(
        pareto.energy.max() - pareto.energy.min(), 1e-9
    )
    axes[1, 1].scatter(pareto.latency, pareto.accuracy, s=sizes, alpha=0.75)
    for name, row in pareto.iterrows():
        axes[1, 1].annotate(name, (row.latency, row.accuracy), xytext=(4, 4), textcoords="offset points")
    axes[1, 1].set_xlabel("Latency (ms)")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_title("Accuracy-latency-energy trade-off")

    unc = df.groupby("correct")[["sensor_uncertainty", "semantic_conflict", "quant_disagreement"]].mean().T
    unc.columns = ["Failure", "Correct"] if list(unc.columns) == [0, 1] else [str(c) for c in unc.columns]
    unc.plot(kind="bar", ax=axes[1, 2])
    axes[1, 2].set_ylabel("Mean uncertainty")
    axes[1, 2].set_title("Uncertainty decomposition by outcome")
    axes[1, 2].tick_params(axis="x", rotation=25)

    for ax in axes.flat:
        ax.grid(True, alpha=0.18)
    fig.suptitle("AdvQFuse-RS result dashboard", fontsize=16)
    _save(fig, out / "fig13_results_dashboard.png")


def plot_dataset_attack_summary(df: pd.DataFrame, out: Path) -> None:
    q = df[df.model == "AdvQFuse-RS"].copy()
    clean = q[q.attack_family == "clean"]
    adv = q[q.attack_family != "clean"]
    clean_acc = clean.pivot_table(index="dataset", columns="question_type", values="correct", aggfunc="mean")
    robust_acc = adv.pivot_table(index="dataset", columns="question_type", values="correct", aggfunc="mean")
    drop = clean_acc.reindex_like(robust_acc) - robust_acc
    coverage = adv.pivot_table(index="dataset", columns="attack_family", values="accepted", aggfunc="mean")
    risk = adv.assign(error=1 - adv.correct).pivot_table(
        index="dataset", columns="attack_family", values="error", aggfunc="mean"
    )
    uncertainty = adv.pivot_table(
        index="dataset", columns="attack_family", values="failure_score", aggfunc="mean"
    )
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    _heatmap(axes[0, 0], clean_acc.fillna(0), "Clean accuracy by dataset and question")
    _heatmap(axes[0, 1], robust_acc.fillna(0), "Robust accuracy by dataset and question")
    _heatmap(axes[0, 2], drop.fillna(0), "Robustness drop")
    _heatmap(axes[1, 0], coverage.fillna(0), "Coverage by dataset and attack")
    _heatmap(axes[1, 1], risk.fillna(0), "Empirical risk by dataset and attack")
    _heatmap(axes[1, 2], uncertainty.fillna(0), "Mean predicted failure score")
    fig.suptitle("Dataset-question-attack diagnostic matrix", fontsize=16)
    _save(fig, out / "fig14_dataset_attack_summary.png")


def plot_question_type_profiles(df: pd.DataFrame, out: Path) -> None:
    qtypes = sorted(df.question_type.unique())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    for ax, qtype in zip(axes.flat, qtypes[:6]):
        part = df[df.question_type == qtype]
        agg = part.groupby(["severity", "model"])["correct"].mean().unstack()
        for model in agg.columns:
            ax.plot(agg.index, agg[model], marker="o", label=model)
        ax.set_title(qtype.replace("_", " ").title())
        ax.set_xlabel("Attack severity")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
    axes[0, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylabel("Accuracy")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Question-type robustness profiles", fontsize=16)
    _save(fig, out / "fig15_question_type_profiles.png")


def plot_calibration_small_multiples(df: pd.DataFrame, out: Path) -> None:
    attacks = list(dict.fromkeys(df.attack_family.tolist()))
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 11)
    for ax, attack in zip(axes.flat, attacks[:6]):
        part = df[df.attack_family == attack]
        for model, group in part.groupby("model"):
            conf = group.confidence.to_numpy(float)
            corr = group.correct.to_numpy(float)
            ids = np.clip(np.digitize(conf, bins) - 1, 0, len(bins) - 2)
            xs, ys = [], []
            for i in range(len(bins) - 1):
                mask = ids == i
                if mask.any():
                    xs.append(conf[mask].mean())
                    ys.append(corr[mask].mean())
            ax.plot(xs, ys, marker="o", label=model)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_title(attack.replace("_", " ").title())
        ax.grid(True, alpha=0.2)
    axes[1, 0].set_xlabel("Confidence")
    axes[1, 1].set_xlabel("Confidence")
    axes[1, 2].set_xlabel("Confidence")
    axes[0, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylabel("Accuracy")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Calibration behavior across attack families", fontsize=16)
    _save(fig, out / "fig16_calibration_by_attack.png")


def plot_policy_analysis(df: pd.DataFrame, out: Path) -> None:
    q = df[df.model == "AdvQFuse-RS"].copy()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    action_dataset = q.pivot_table(index="dataset", columns="policy_action", values="sample_id", aggfunc="count", fill_value=0)
    action_dataset = action_dataset.div(action_dataset.sum(axis=1), axis=0)
    action_dataset.plot(kind="bar", stacked=True, ax=axes[0, 0])
    axes[0, 0].set_ylabel("Policy fraction")
    axes[0, 0].set_title("Action mix by dataset")
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].legend(fontsize=7)

    action_acc = q.groupby("policy_action")["correct"].mean().sort_values()
    axes[0, 1].barh(action_acc.index, action_acc.values)
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_xlabel("Accuracy")
    axes[0, 1].set_title("Outcome quality per action")

    action_cost = q.groupby("policy_action")[["latency_ms", "energy_j"]].mean()
    x = np.arange(len(action_cost))
    ax2 = axes[0, 2].twinx()
    axes[0, 2].bar(x - 0.18, action_cost.latency_ms, 0.36, label="Latency")
    ax2.bar(x + 0.18, action_cost.energy_j, 0.36, label="Energy")
    axes[0, 2].set_xticks(x, action_cost.index, rotation=25, ha="right")
    axes[0, 2].set_ylabel("Latency (ms)")
    ax2.set_ylabel("Energy (J)")
    axes[0, 2].set_title("Cost of progressive actions")

    for dataset, group in q.groupby("dataset"):
        cov, risk = _risk_coverage(group)
        axes[1, 0].plot(cov, risk, label=dataset)
    axes[1, 0].set_xlabel("Coverage")
    axes[1, 0].set_ylabel("Selective risk")
    axes[1, 0].set_title("Dataset-specific risk-coverage")
    axes[1, 0].legend(fontsize=7)

    policy_attack = q.pivot_table(index="attack_family", columns="policy_action", values="sample_id", aggfunc="count", fill_value=0)
    policy_attack = policy_attack.div(policy_attack.sum(axis=1), axis=0)
    policy_attack.plot(kind="bar", stacked=True, ax=axes[1, 1])
    axes[1, 1].set_ylabel("Policy fraction")
    axes[1, 1].set_title("Action mix by attack")
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[1, 1].legend(fontsize=7)

    q["utility"] = q.correct - 0.00012 * q.latency_ms - 0.006 * q.energy_j - 0.25 * (q.policy_action == "abstain")
    utility = q.groupby(["dataset", "policy_action"])["utility"].mean().unstack()
    utility.plot(kind="bar", ax=axes[1, 2])
    axes[1, 2].set_ylabel("Illustrative utility")
    axes[1, 2].set_title("Accuracy-cost utility by dataset")
    axes[1, 2].tick_params(axis="x", rotation=25)
    axes[1, 2].legend(fontsize=7)

    for ax in axes.flat:
        ax.grid(True, alpha=0.16)
    fig.suptitle("Progressive-precision policy analysis", fontsize=16)
    _save(fig, out / "fig17_policy_analysis.png")


def plot_uncertainty_diagnostics(df: pd.DataFrame, out: Path) -> None:
    q = df[df.model == "AdvQFuse-RS"].sample(min(7000, len(df[df.model == "AdvQFuse-RS"])), random_state=0)
    cols = ["sensor_uncertainty", "semantic_conflict", "quant_disagreement"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes[0], cols):
        ax.hexbin(q[col], q.failure_score, gridsize=35, mincnt=1)
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Failure score")
        ax.set_title("Density relationship")
        ax.grid(True, alpha=0.12)

    corr = q[cols + ["failure_score", "correct", "confidence"]].corr()
    _heatmap(axes[1, 0], corr, "Uncertainty correlation matrix")

    q["uncertainty_quartile"] = pd.qcut(q.failure_score, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    quart = q.groupby("uncertainty_quartile", observed=False).agg(
        accuracy=("correct", "mean"), coverage=("accepted", "mean"), confidence=("confidence", "mean")
    )
    quart.plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Behavior across failure-score quartiles")
    axes[1, 1].tick_params(axis="x", rotation=0)

    bins = pd.cut(q.failure_score, bins=np.linspace(0, 1, 11), include_lowest=True)
    calib = q.groupby(bins, observed=False).agg(
        predicted=("failure_score", "mean"), observed=("correct", lambda x: 1 - x.mean()), n=("correct", "size")
    ).dropna()
    axes[1, 2].plot(calib.predicted, calib.observed, marker="o")
    axes[1, 2].plot([0, 1], [0, 1], linestyle="--")
    for row in calib.itertuples():
        axes[1, 2].annotate(str(int(row.n)), (row.predicted, row.observed), xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[1, 2].set_xlabel("Predicted failure probability")
    axes[1, 2].set_ylabel("Observed failure rate")
    axes[1, 2].set_title("Failure-probability calibration")
    axes[1, 2].grid(True, alpha=0.2)

    fig.suptitle("Uncertainty and failure-prediction diagnostics", fontsize=16)
    _save(fig, out / "fig18_uncertainty_diagnostics.png")


def plot_attack_family_matrix(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), sharey=True)
    attacks = [a for a in df.attack_family.unique() if a != "clean"]
    for ax, attack in zip(axes.flat, attacks[:6]):
        part = df[df.attack_family == attack]
        summary = part.groupby(["severity", "model"]).agg(
            accuracy=("correct", "mean"),
            confidence=("confidence", "mean"),
            uncertainty=("failure_score", "mean"),
        ).reset_index()
        for model, group in summary.groupby("model"):
            ax.plot(group.severity, group.accuracy, marker="o", label=f"{model} acc")
            if model == "AdvQFuse-RS":
                ax.plot(group.severity, group.uncertainty, marker="x", linestyle="--", label="QFuse uncertainty")
        ax.set_title(attack.replace("_", " ").title())
        ax.set_xlabel("Severity")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    axes[0, -1].legend(fontsize=7)
    fig.suptitle("Attack-specific robustness and uncertainty trajectories", fontsize=16)
    _save(fig, out / "fig19_attack_family_matrix.png")


def plot_statistical_summary(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    forest_dataset = bootstrap_accuracy_difference(
        df, "AdvQFuse-RS", "Bonsai-1bit", group_columns=("dataset",), n_bootstrap=600, seed=1
    ).sort_values("difference")
    y = np.arange(len(forest_dataset))
    xerr = np.vstack([
        forest_dataset.difference - forest_dataset.ci_low,
        forest_dataset.ci_high - forest_dataset.difference,
    ])
    axes[0, 0].errorbar(forest_dataset.difference, y, xerr=xerr, fmt="o", capsize=3)
    axes[0, 0].axvline(0, linestyle="--")
    axes[0, 0].set_yticks(y, forest_dataset.dataset)
    axes[0, 0].set_xlabel("Accuracy difference")
    axes[0, 0].set_title("Effect size by dataset")

    forest_attack = bootstrap_accuracy_difference(
        df[df.attack_family != "clean"],
        "AdvQFuse-RS",
        "Bonsai-1bit",
        group_columns=("attack_family",),
        n_bootstrap=600,
        seed=2,
    ).sort_values("difference")
    y2 = np.arange(len(forest_attack))
    xerr2 = np.vstack([
        forest_attack.difference - forest_attack.ci_low,
        forest_attack.ci_high - forest_attack.difference,
    ])
    axes[0, 1].errorbar(forest_attack.difference, y2, xerr=xerr2, fmt="o", capsize=3)
    axes[0, 1].axvline(0, linestyle="--")
    axes[0, 1].set_yticks(y2, [s.replace("_", " ") for s in forest_attack.attack_family])
    axes[0, 1].set_xlabel("Accuracy difference")
    axes[0, 1].set_title("Effect size by attack")

    paired = df.pivot_table(index="sample_id", columns="model", values="correct", aggfunc="first").dropna()
    win = pd.DataFrame(index=paired.columns, columns=paired.columns, dtype=float)
    for a in paired.columns:
        for b in paired.columns:
            win.loc[a, b] = float((paired[a] > paired[b]).mean())
    _heatmap(axes[1, 0], win, "Pairwise exclusive win rate")

    rng = np.random.default_rng(3)
    q = paired["AdvQFuse-RS"].to_numpy(float)
    b = paired["Bonsai-1bit"].to_numpy(float)
    diffs = []
    for _ in range(1200):
        idx = rng.integers(0, len(q), len(q))
        diffs.append(float(np.mean(q[idx] - b[idx])))
    axes[1, 1].hist(diffs, bins=35, density=True, alpha=0.75)
    axes[1, 1].axvline(0, linestyle="--")
    axes[1, 1].axvline(np.mean(diffs), linestyle="-")
    axes[1, 1].set_xlabel("Bootstrap accuracy difference")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Paired bootstrap distribution")

    for ax in axes.flat:
        ax.grid(True, alpha=0.16)
    fig.suptitle("Statistical effect-size and consistency analysis", fontsize=16)
    _save(fig, out / "fig20_statistical_summary.png")


def plot_failure_overlap(df: pd.DataFrame, out: Path) -> None:
    pivot = df.pivot_table(index="sample_id", columns="model", values="correct", aggfunc="first").dropna()
    models = list(pivot.columns)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    overlap = pd.DataFrame(index=models, columns=models, dtype=float)
    for a in models:
        for b in models:
            fa = pivot[a] == 0
            fb = pivot[b] == 0
            overlap.loc[a, b] = float((fa & fb).sum() / max((fa | fb).sum(), 1))
    _heatmap(axes[0, 0], overlap, "Failure-set Jaccard overlap")

    q = df[df.model == "AdvQFuse-RS"]
    failure_type = q[q.correct == 0].pivot_table(
        index="failure_type", columns="attack_family", values="sample_id", aggfunc="count", fill_value=0
    )
    failure_type.plot(kind="bar", stacked=True, ax=axes[0, 1])
    axes[0, 1].set_ylabel("Failure count")
    axes[0, 1].set_title("Failure taxonomy by attack")
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].legend(fontsize=6)

    recover = df.pivot_table(index="sample_id", columns="model", values="correct", aggfunc="first").dropna()
    categories = {
        "All correct": ((recover == 1).all(axis=1)).sum(),
        "QFuse only correct": ((recover["AdvQFuse-RS"] == 1) & (recover.drop(columns="AdvQFuse-RS") == 0).all(axis=1)).sum(),
        "Ternary recovers binary": ((recover["Ternary-1.7bit"] == 1) & (recover["Bonsai-1bit"] == 0)).sum(),
        "All fail": ((recover == 0).all(axis=1)).sum(),
    }
    axes[0, 2].bar(categories.keys(), categories.values())
    axes[0, 2].set_ylabel("Samples")
    axes[0, 2].set_title("Recovery and shared-failure categories")
    axes[0, 2].tick_params(axis="x", rotation=25)

    dataset_failure = q.assign(error=1 - q.correct).pivot_table(
        index="dataset", columns="failure_type", values="error", aggfunc="sum", fill_value=0
    )
    dataset_failure.plot(kind="bar", stacked=True, ax=axes[1, 0])
    axes[1, 0].set_ylabel("Failure count")
    axes[1, 0].set_title("Dataset-specific failure composition")
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].legend(fontsize=6)

    disagreement = df.pivot_table(index="sample_id", columns="model", values="correct", aggfunc="first").dropna()
    pattern = disagreement.astype(str).agg("-".join, axis=1).value_counts().head(8)
    axes[1, 1].barh(pattern.index, pattern.values)
    axes[1, 1].set_xlabel("Samples")
    axes[1, 1].set_title("Most common model outcome patterns")

    attack_failure = q[q.correct == 0].pivot_table(
        index="attack_family", columns="severity", values="sample_id", aggfunc="count", fill_value=0
    )
    attack_failure.plot(kind="bar", ax=axes[1, 2])
    axes[1, 2].set_ylabel("Failure count")
    axes[1, 2].set_title("Failure escalation across severity")
    axes[1, 2].tick_params(axis="x", rotation=25)
    axes[1, 2].legend(fontsize=7, title="Severity")

    for ax in axes.flat:
        ax.grid(True, alpha=0.14)
    fig.suptitle("Failure overlap, recovery, and taxonomy", fontsize=16)
    _save(fig, out / "fig21_failure_overlap.png")


def plot_operating_point_sweep(df: pd.DataFrame, out: Path) -> None:
    q = df[(df.model == "AdvQFuse-RS") & (df.attack_family != "clean")].copy()
    thresholds = np.linspace(0.03, 0.95, 45)
    rows = []
    for threshold in thresholds:
        accepted = q.failure_score <= threshold
        coverage = accepted.mean()
        risk = (1 - q.loc[accepted, "correct"].mean()) if accepted.any() else np.nan
        accuracy = q.loc[accepted, "correct"].mean() if accepted.any() else np.nan
        latency = q.loc[accepted, "latency_ms"].mean() if accepted.any() else np.nan
        energy = q.loc[accepted, "energy_j"].mean() if accepted.any() else np.nan
        utility = (accuracy if np.isfinite(accuracy) else 0) - 0.30 * (1 - coverage) - 0.25 * (risk if np.isfinite(risk) else 1)
        rows.append((threshold, coverage, risk, accuracy, latency, energy, utility))
    sweep = pd.DataFrame(rows, columns=["threshold", "coverage", "risk", "accuracy", "latency", "energy", "utility"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    pairs = [
        ("coverage", "Coverage"),
        ("risk", "Selective risk"),
        ("accuracy", "Accepted-set accuracy"),
        ("latency", "Mean latency (ms)"),
        ("energy", "Mean energy (J)"),
        ("utility", "Illustrative utility"),
    ]
    for ax, (col, label) in zip(axes.flat, pairs):
        ax.plot(sweep.threshold, sweep[col], linewidth=2)
        ax.set_xlabel("Acceptance threshold")
        ax.set_ylabel(label)
        ax.set_title(label + " vs threshold")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Selective operating-point sweep", fontsize=16)
    _save(fig, out / "fig22_operating_point_sweep.png")
    sweep.to_csv(out.parent.parent / "results" / "extended_demo" / "operating_point_sweep.csv", index=False)


def _synthetic_ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    q = df[(df.model == "AdvQFuse-RS") & (df.attack_family != "clean")].copy()
    full = {
        "accuracy": q.correct.mean(),
        "coverage": q.accepted.mean(),
        "risk": 1 - q[q.accepted == 1].correct.mean(),
        "latency": q.latency_ms.mean(),
        "energy": q.energy_j.mean(),
    }
    variants = {
        "Full": (0.000, 0.000, 0.000, 1.00, 1.00),
        "w/o quant signal": (-0.035, 0.028, 0.025, 0.97, 0.98),
        "w/o sensor signal": (-0.026, 0.020, 0.018, 0.98, 0.98),
        "w/o conflict signal": (-0.031, 0.024, 0.022, 0.98, 0.99),
        "fixed threshold": (-0.018, 0.055, -0.015, 0.94, 0.95),
        "always ternary": (0.012, -0.010, 0.000, 1.34, 1.29),
        "always binary": (-0.071, 0.065, 0.080, 0.79, 0.73),
    }
    rows = []
    for name, (da, dr, dc, lm, em) in variants.items():
        rows.append({
            "variant": name,
            "accuracy": np.clip(full["accuracy"] + da, 0, 1),
            "risk": np.clip(full["risk"] + dr, 0, 1),
            "coverage": np.clip(full["coverage"] + dc, 0, 1),
            "latency_ms": full["latency"] * lm,
            "energy_j": full["energy"] * em,
        })
    result = pd.DataFrame(rows)
    result["utility"] = result.accuracy - 0.25 * result.risk - 0.00012 * result.latency_ms - 0.005 * result.energy_j
    return result


def plot_ablation_template(df: pd.DataFrame, out: Path) -> None:
    abl = _synthetic_ablation_table(df)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    metrics = [
        ("accuracy", "Accuracy"),
        ("risk", "Selective risk"),
        ("coverage", "Coverage"),
        ("latency_ms", "Latency (ms)"),
        ("energy_j", "Energy (J)"),
        ("utility", "Illustrative utility"),
    ]
    for ax, (col, title) in zip(axes.flat, metrics):
        ordered = abl.sort_values(col, ascending=(col in {"risk", "latency_ms", "energy_j"}))
        ax.barh(ordered.variant, ordered[col])
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.2)
    fig.suptitle("Ablation visualization template (synthetic placeholders)", fontsize=16)
    _save(fig, out / "fig23_ablation_template.png")
    abl.to_csv(out.parent.parent / "results" / "extended_demo" / "synthetic_ablation_template.csv", index=False)


def plot_distribution_suite(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    models = list(dict.fromkeys(df.model.tolist()))
    columns = [
        ("latency_ms", "Latency distribution"),
        ("energy_j", "Energy distribution"),
        ("confidence", "Confidence distribution"),
        ("failure_score", "Failure-score distribution"),
        ("quant_disagreement", "Quantization disagreement"),
        ("semantic_conflict", "Cross-modal conflict"),
    ]
    for ax, (col, title) in zip(axes.flat, columns):
        arrays = [df[df.model == model][col].dropna().to_numpy(float) for model in models]
        ax.violinplot(arrays, positions=np.arange(1, len(models) + 1), showmeans=True, showextrema=False)
        ax.set_xticks(np.arange(1, len(models) + 1), models, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.18)
    fig.suptitle("Distributional result analysis", fontsize=16)
    _save(fig, out / "fig24_distribution_suite.png")


def generate_extended_figures(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    functions = [
        plot_results_dashboard,
        plot_dataset_attack_summary,
        plot_question_type_profiles,
        plot_calibration_small_multiples,
        plot_policy_analysis,
        plot_uncertainty_diagnostics,
        plot_attack_family_matrix,
        plot_statistical_summary,
        plot_failure_overlap,
        plot_operating_point_sweep,
        plot_ablation_template,
        plot_distribution_suite,
    ]
    for fn in functions:
        fn(df, out)
    return sorted(out.glob("*.png"))

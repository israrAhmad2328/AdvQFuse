from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


WATERMARK = "SYNTHETIC LAYOUT — NOT PAPER EVIDENCE"


def _finish(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(fig._suptitle.get_text() if fig._suptitle else "", y=0.995)
    fig.text(0.5, 0.004, WATERMARK, ha="center", va="bottom", fontsize=9, alpha=0.65)
    fig.tight_layout(rect=(0, 0.025, 1, 0.965))
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure37_precision_path_geometry(output: Path, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    precisions = np.array([2, 3, 4, 8, 16])
    cases = ["Stable", "Natural difficulty", "Corruption", "Adversarial"]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    fig.suptitle("Fig. 37 — Precision-path geometry and failure mechanisms")

    for i, label in enumerate(cases):
        base = np.linspace(0.15 + i * 0.09, 0.72, len(precisions))
        noise = rng.normal(0, 0.012 + i * 0.008, len(precisions))
        path = np.clip(base + noise + (0.08 * np.sin(precisions) if i >= 2 else 0), 0.01, 0.98)
        axes[0, 0].plot(precisions, path, marker="o", label=label)
    axes[0, 0].set(title="(a) Correct-answer probability", xlabel="Precision coordinate", ylabel="Probability")
    axes[0, 0].legend(fontsize=8)

    drift = np.abs(rng.normal([0.015, 0.04, 0.09, 0.16], [0.006, 0.01, 0.02, 0.025], (80, 4)))
    axes[0, 1].boxplot([drift[:, i] for i in range(4)], tick_labels=cases, showfliers=False)
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].set(title="(b) Adjacent JS drift", ylabel="Jensen–Shannon divergence")

    margins = np.linspace(0.02, 0.75, 180)
    path_var = np.clip(0.28 * np.exp(-3.3 * margins) + rng.normal(0, 0.018, len(margins)), 0, None)
    axes[0, 2].scatter(margins, path_var, s=13, alpha=0.6)
    axes[0, 2].set(title="(c) Margin–instability relation", xlabel="High-precision top-two margin", ylabel="Path variation")

    classes = np.arange(8)
    stable = np.exp(-0.75 * (classes - 2.0) ** 2)
    unstable = np.exp(-0.55 * (classes - 5.0) ** 2)
    for k, precision in enumerate(precisions):
        mix = (k / (len(precisions) - 1)) * stable + (1 - k / (len(precisions) - 1)) * unstable
        axes[1, 0].plot(classes, mix / mix.sum(), marker=".", label=f"q={precision}")
    axes[1, 0].set(title="(d) Rank migration example", xlabel="Answer class", ylabel="Probability")
    axes[1, 0].legend(fontsize=7, ncol=2)

    sample_types = ["Clean", "Corrupt", "Attack", "Conflict"]
    feature_names = ["JS", "Curvature", "Flips", "Margin loss", "Semantic", "Modality"]
    heat = rng.uniform(0.05, 0.35, (4, 6)) + np.arange(4)[:, None] * 0.14
    im = axes[1, 1].imshow(heat, aspect="auto")
    axes[1, 1].set(title="(e) Path-feature signatures", xticks=np.arange(6), xticklabels=feature_names, yticks=np.arange(4), yticklabels=sample_types)
    axes[1, 1].tick_params(axis="x", rotation=35)
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    cert_margin = rng.uniform(0.05, 1.0, 220)
    deviation = np.clip(rng.gamma(1.4, 0.09, 220), 0, 0.7)
    certified = deviation < cert_margin / 2
    axes[1, 2].scatter(cert_margin[~certified], deviation[~certified], s=14, alpha=0.5, label="Not certified")
    axes[1, 2].scatter(cert_margin[certified], deviation[certified], s=14, alpha=0.6, label="Certified")
    x = np.linspace(0, 1, 100)
    axes[1, 2].plot(x, x / 2, linestyle="--", label="γ/2 boundary")
    axes[1, 2].set(title="(f) Observed-path invariance", xlabel="Reference margin γ", ylabel="Maximum logit deviation")
    axes[1, 2].legend(fontsize=8)
    _finish(fig, output)


def figure38_heldout_generalization(output: Path, seed: int = 23) -> None:
    rng = np.random.default_rng(seed)
    families = ["Qwen", "LLaVA", "InternVL", "MiniCPM", "Bonsai"]
    datasets = ["VQAv2", "GQA", "TextVQA", "POPE", "ImageNet-X", "Remote sensing"]
    attacks = ["Corrupt.", "PGD", "Transfer", "Patch", "Text", "Conflict"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.3))
    fig.suptitle("Fig. 38 — Held-out model, dataset, attack, and precision generalization")

    mats = [
        (rng.normal(0.82, 0.035, (5, 5)), families, families, "(a) Train-family → test-family AUROC"),
        (rng.normal(0.80, 0.045, (6, 6)), datasets, datasets, "(b) Train-dataset → test-dataset AUROC"),
        (rng.normal(0.78, 0.05, (6, 6)), attacks, attacks, "(c) Train-attack → test-attack AUROC"),
    ]
    for ax, (mat, xlabels, ylabels, title) in zip(axes[0], mats, strict=True):
        np.fill_diagonal(mat, np.clip(np.diag(mat) + 0.08, 0, 0.98))
        im = ax.imshow(np.clip(mat, 0.5, 0.98), vmin=0.5, vmax=1.0, aspect="auto")
        ax.set(title=title, xticks=np.arange(len(xlabels)), xticklabels=xlabels, yticks=np.arange(len(ylabels)), yticklabels=ylabels)
        ax.tick_params(axis="x", rotation=40)
        fig.colorbar(im, ax=ax, fraction=0.046)

    methods = ["Entropy", "TTA consistency", "Counterfactuals", "Path only", "Full PPSE"]
    x = np.arange(len(families))
    width = 0.15
    for j, method in enumerate(methods):
        values = 0.63 + 0.035 * j + rng.normal(0, 0.012, len(families))
        axes[1, 0].bar(x + (j - 2) * width, values, width=width, label=method)
    axes[1, 0].set(title="(d) Leave-one-family-out comparison", xticks=x, xticklabels=families, ylabel="Error AUROC", ylim=(0.55, 0.9))
    axes[1, 0].legend(fontsize=7, ncol=2)

    precisions = ["16→8", "8→4", "4→3", "3→2", "2→ternary", "ternary→binary"]
    observed = np.clip(rng.normal([0.84, 0.83, 0.81, 0.78, 0.75, 0.72], 0.014), 0, 1)
    lower = observed - rng.uniform(0.015, 0.03, len(observed))
    upper = observed + rng.uniform(0.015, 0.03, len(observed))
    axes[1, 1].errorbar(np.arange(len(observed)), observed, yerr=[observed - lower, upper - observed], marker="o", capsize=3)
    axes[1, 1].set(title="(e) Held-out precision-edge interpolation", xticks=np.arange(len(precisions)), xticklabels=precisions, ylabel="Error AUROC", ylim=(0.65, 0.9))
    axes[1, 1].tick_params(axis="x", rotation=35)

    generic_fraction = np.arange(10, 101, 10)
    generic_auc = 0.71 + 0.12 * (1 - np.exp(-generic_fraction / 35))
    rs_auc = 0.69 + 0.11 * (1 - np.exp(-generic_fraction / 45)) + rng.normal(0, 0.006, len(generic_fraction))
    axes[1, 2].plot(generic_fraction, generic_auc, marker="o", label="Generic held-out")
    axes[1, 2].plot(generic_fraction, rs_auc, marker="s", label="Remote-sensing transfer")
    axes[1, 2].set(title="(f) Domain breadth scaling", xlabel="Generic-domain training fraction (%)", ylabel="Error AUROC")
    axes[1, 2].legend(fontsize=8)
    _finish(fig, output)


def figure39_risk_cost_frontier(output: Path, seed: int = 47) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    fig.suptitle("Fig. 39 — Risk-controlled adaptive inference at matched compute")
    coverages = np.linspace(0.15, 1.0, 18)
    methods = ["Entropy", "Consistency", "Path only", "AdvQFuse"]
    for j, method in enumerate(methods):
        risk = np.clip(0.008 + (coverages ** (2.0 - 0.15 * j)) * (0.26 - 0.035 * j) + rng.normal(0, 0.004, len(coverages)), 0, 1)
        axes[0, 0].plot(coverages, risk, marker=".", label=method)
    axes[0, 0].axhline(0.05, linestyle="--")
    axes[0, 0].set(title="(a) Risk–coverage", xlabel="Coverage", ylabel="Selective risk")
    axes[0, 0].legend(fontsize=8)

    budgets = np.linspace(1.0, 4.2, 16)
    for j, method in enumerate(methods):
        robust = 0.50 + (0.20 + 0.025 * j) * (1 - np.exp(-(budgets - 0.8) / (1.5 + 0.2 * j)))
        axes[0, 1].plot(budgets, robust, marker=".", label=method)
    axes[0, 1].set(title="(b) Robust score at matched cost", xlabel="Normalized expected compute", ylabel="Robust task score")

    actions = ["Accept", "Reobserve", "Escalate", "Ensemble", "Abstain"]
    shifts = ["Clean", "Corrupt.", "Attack", "Text", "Conflict"]
    action_mix = rng.dirichlet(np.array([5, 2, 3, 1.5, 1.2]), size=len(shifts))
    bottom = np.zeros(len(shifts))
    for j, action in enumerate(actions):
        axes[0, 2].bar(shifts, action_mix[:, j], bottom=bottom, label=action)
        bottom += action_mix[:, j]
    axes[0, 2].set(title="(c) Policy action composition", ylabel="Fraction")
    axes[0, 2].tick_params(axis="x", rotation=25)
    axes[0, 2].legend(fontsize=7, ncol=2)

    alpha = np.array([0.01, 0.02, 0.05, 0.10])
    groups = ["Aggregate", "Model", "Dataset", "Attack", "Precision"]
    x = np.arange(len(alpha))
    for i, group in enumerate(groups):
        violations = np.clip(rng.normal(0.018 + i * 0.004, 0.005, len(alpha)) * (0.06 / alpha), 0, 0.12)
        axes[1, 0].plot(x, violations, marker="o", label=group)
    axes[1, 0].set(title="(d) Empirical bound-violation rate", xticks=x, xticklabels=alpha, xlabel="Target risk α", ylabel="Violation rate")
    axes[1, 0].legend(fontsize=7)

    lat = rng.uniform(20, 160, 60)
    energy = 0.018 * lat + rng.normal(0, 0.25, 60)
    score = 0.58 + 0.16 * (1 - np.exp(-lat / 75)) + rng.normal(0, 0.008, 60)
    scatter = axes[1, 1].scatter(lat, energy, s=30 + 260 * (score - score.min()), c=score)
    axes[1, 1].set(title="(e) Latency–energy–accuracy frontier", xlabel="Latency (ms)", ylabel="Energy proxy (J)")
    fig.colorbar(scatter, ax=axes[1, 1], fraction=0.046, label="Task score")

    groups2 = ["Qwen", "LLaVA", "InternVL", "MiniCPM", "Generic", "Text-rich", "Remote", "Attack"]
    target = 0.05
    empirical = np.clip(rng.normal(0.036, 0.009, len(groups2)), 0.01, 0.07)
    upper = empirical + rng.uniform(0.006, 0.016, len(groups2))
    axes[1, 2].errorbar(np.arange(len(groups2)), empirical, yerr=upper - empirical, fmt="o", capsize=4)
    axes[1, 2].axhline(target, linestyle="--", label="Target α=0.05")
    axes[1, 2].set(title="(f) Group-wise selective risk", xticks=np.arange(len(groups2)), xticklabels=groups2, ylabel="Risk / upper bound")
    axes[1, 2].tick_params(axis="x", rotation=35)
    axes[1, 2].legend(fontsize=8)
    _finish(fig, output)


def _synthetic_scene(rng: np.random.Generator, size: int = 96) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    image = 0.25 + 0.25 * np.sin(x / 13) + 0.18 * np.cos(y / 17)
    image += rng.normal(0, 0.04, (size, size))
    for _ in range(6):
        cx, cy = rng.integers(10, size - 10, 2)
        rr = rng.integers(4, 10)
        image[(x - cx) ** 2 + (y - cy) ** 2 < rr**2] += rng.uniform(0.2, 0.45)
    return np.clip(image, 0, 1)


def figure40_qualitative_interventions(output: Path, seed: int = 89) -> None:
    rng = np.random.default_rng(seed)
    rows = ["Stable → accept", "Cloud → reobserve", "Attack → escalate", "Conflict → abstain"]
    cols = ["Input", "Perturbation map", "Precision path", "Evidence audit", "Final action"]
    fig, axes = plt.subplots(4, 5, figsize=(16, 11))
    fig.suptitle("Fig. 40 — Qualitative precision-intervention casebook")
    for r, row in enumerate(rows):
        scene = _synthetic_scene(rng)
        if r == 1:
            scene[20:63, 35:83] = np.clip(scene[20:63, 35:83] + 0.35, 0, 1)
        if r == 2:
            checker = np.indices((25, 25)).sum(axis=0) % 2
            scene[12:37, 58:83] = checker
        axes[r, 0].imshow(scene, cmap="gray")
        axes[r, 0].set_ylabel(row, fontsize=10)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

        perturb = np.abs(np.gradient(scene)[0]) + np.abs(np.gradient(scene)[1])
        axes[r, 1].imshow(perturb)
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])

        precisions = np.array([2, 3, 4, 8, 16])
        if r == 0:
            p = np.array([0.83, 0.84, 0.85, 0.86, 0.87])
        elif r == 1:
            p = np.array([0.42, 0.51, 0.57, 0.66, 0.69])
        elif r == 2:
            p = np.array([0.78, 0.27, 0.55, 0.41, 0.74])
        else:
            p = np.array([0.62, 0.59, 0.57, 0.60, 0.61])
        axes[r, 2].plot(precisions, p, marker="o")
        axes[r, 2].set_ylim(0, 1)
        axes[r, 2].set_xlabel("Precision") if r == 3 else None
        axes[r, 2].set_ylabel("P(correct)")

        evidence = [
            [0.12, 0.10, 0.08, 0.06],
            [0.55, 0.23, 0.34, 0.29],
            [0.71, 0.62, 0.48, 0.66],
            [0.18, 0.74, 0.31, 0.79],
        ][r]
        axes[r, 3].bar(["Path", "Sensor", "Text", "Conflict"], evidence)
        axes[r, 3].set_ylim(0, 1)
        axes[r, 3].tick_params(axis="x", rotation=35)

        actions = ["ACCEPT", "REOBSERVE", "ESCALATE", "ABSTAIN"]
        reasons = [
            "Low path variation\nRisk UCB < α",
            "Low visual quality\nView recovers evidence",
            "Rank flips under attack\nHigher precision helps",
            "Modal contradiction\nNo certified action",
        ]
        axes[r, 4].axis("off")
        axes[r, 4].add_patch(Rectangle((0.08, 0.38), 0.84, 0.30, fill=False, linewidth=2))
        axes[r, 4].text(0.5, 0.56, actions[r], ha="center", va="center", fontsize=14, weight="bold")
        axes[r, 4].text(0.5, 0.26, reasons[r], ha="center", va="center", fontsize=9)
    for c, col in enumerate(cols):
        axes[0, c].set_title(col)
    _finish(fig, output)


def figure41_scope_and_evidence(output: Path, seed: int = 131) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    fig.suptitle("Fig. 41 — Scope breadth, evidence hierarchy, and anti-narrowness audit")
    dimensions = ["Models", "Precisions", "Tasks", "Datasets", "Shifts", "Held-out tests"]
    versions = ["HNST profile", "AdvQFuse-RS", "TPAMI redesign"]
    values = np.array([[1, 1, 1, 3, 1, 0], [1, 2, 1, 5, 8, 1], [5, 5, 5, 10, 11, 5]], dtype=float)
    values = values / values.max(axis=0, keepdims=True)
    x = np.arange(len(dimensions)); width = 0.24
    for i, version in enumerate(versions):
        axes[0, 0].bar(x + (i - 1) * width, values[i], width, label=version)
    axes[0, 0].set(title="(a) Breadth profile", xticks=x, xticklabels=dimensions, ylabel="Normalized breadth", ylim=(0, 1.05))
    axes[0, 0].tick_params(axis="x", rotation=35)
    axes[0, 0].legend(fontsize=8)

    tasks = ["VQA", "OCR", "Halluc.", "Classif.", "Remote"]
    models = ["Qwen", "LLaVA", "InternVL", "MiniCPM", "Bonsai"]
    coverage = np.ones((5, 5)); coverage[-1, :3] = 0.55; coverage[-1, 3] = 0.2
    im = axes[0, 1].imshow(coverage, vmin=0, vmax=1)
    axes[0, 1].set(title="(b) Model × task coverage", xticks=np.arange(5), xticklabels=tasks, yticks=np.arange(5), yticklabels=models)
    axes[0, 1].tick_params(axis="x", rotation=30)
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)

    evidence = ["Single demo", "Within-domain", "Cross-dataset", "Held-out family", "Mechanistic + risk"]
    strength = [0.12, 0.34, 0.53, 0.78, 0.95]
    axes[0, 2].barh(evidence, strength)
    axes[0, 2].set(title="(c) Evidence hierarchy", xlabel="Editorial strength", xlim=(0, 1))

    claims = ["Reliability signal", "Cross-family", "Risk control", "Cost benefit", "Adversarial", "Sub-2-bit"]
    evidence_types = ["Theory", "Main table", "Held-out", "Ablation", "Qualitative", "Failure cases"]
    traceability = rng.uniform(0.45, 0.75, (len(claims), len(evidence_types)))
    for i in range(len(claims)):
        traceability[i, i % len(evidence_types)] = 0.95
    im2 = axes[1, 0].imshow(traceability, vmin=0, vmax=1, aspect="auto")
    axes[1, 0].set(title="(d) Claim–evidence traceability", xticks=np.arange(len(evidence_types)), xticklabels=evidence_types, yticks=np.arange(len(claims)), yticklabels=claims)
    axes[1, 0].tick_params(axis="x", rotation=35)
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    sample_share = [42, 18, 12, 11, 10, 7]
    domains = ["General VQA", "Text-rich", "Recognition shift", "Hallucination", "Remote sensing", "Other"]
    axes[1, 1].pie(sample_share, labels=domains, autopct="%1.0f%%", textprops={"fontsize": 8})
    axes[1, 1].set_title("(e) Recommended principal sample mix")

    gates = ["≥4 families", "≥8 datasets", "LOMO gain", "Risk targets", "Matched compute", "No synthetic evidence"]
    pass_probability = [0.95, 0.92, 0.68, 0.73, 0.70, 1.0]
    axes[1, 2].barh(gates, pass_probability)
    axes[1, 2].axvline(0.8, linestyle="--")
    axes[1, 2].set(title="(f) Pre-submission firewall", xlabel="Required completion / evidence", xlim=(0, 1.05))
    _finish(fig, output)


def figure42_statistical_evidence(output: Path, seed: int = 181) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    fig.suptitle("Fig. 42 — Statistical evidence, uncertainty, and failure analysis")
    comparisons = ["vs entropy", "vs TTA", "vs paraphrase", "vs path-only", "vs CF-only", "vs ensemble"]
    effects = rng.normal([0.09, 0.07, 0.075, 0.035, 0.03, 0.025], 0.006)
    half = rng.uniform(0.009, 0.018, len(effects))
    axes[0, 0].errorbar(effects, np.arange(len(effects)), xerr=half, fmt="o", capsize=4)
    axes[0, 0].axvline(0, linestyle="--")
    axes[0, 0].set(title="(a) Paired bootstrap AUROC effects", xlabel="AdvQFuse minus baseline", yticks=np.arange(len(comparisons)), yticklabels=comparisons)

    seeds = np.arange(1, 6)
    methods = ["Entropy", "Path only", "Full"]
    for j, method in enumerate(methods):
        vals = rng.normal(0.70 + 0.055 * j, 0.012, len(seeds))
        axes[0, 1].plot(seeds, vals, marker="o", label=method)
    axes[0, 1].set(title="(b) Seed sensitivity", xlabel="Seed index", ylabel="Held-out AUROC")
    axes[0, 1].legend(fontsize=8)

    sizes = np.array([250, 500, 1000, 2000, 5000, 10000])
    risk_gap = 0.18 / np.sqrt(sizes / 250) + rng.normal(0, 0.002, len(sizes))
    axes[0, 2].plot(sizes, risk_gap, marker="o")
    axes[0, 2].set_xscale("log")
    axes[0, 2].set(title="(c) Calibration sample complexity", xlabel="Calibration samples", ylabel="UCB minus empirical risk")

    categories = ["Confident stable error", "Path-unstable", "Visual corruption", "Text attack", "Modal conflict", "OOD"]
    counts = np.array([18, 31, 22, 14, 10, 5])
    axes[1, 0].barh(categories, counts)
    axes[1, 0].set(title="(d) Residual failure taxonomy", xlabel="Share of residual errors (%)")

    nominal = np.linspace(0.01, 0.15, 10)
    empirical = nominal * rng.normal(0.82, 0.06, len(nominal))
    upper = empirical + rng.uniform(0.008, 0.018, len(nominal))
    axes[1, 1].fill_between(nominal, empirical, upper, alpha=0.25, label="Empirical→UCB")
    axes[1, 1].plot(nominal, empirical, marker="o", label="Empirical risk")
    axes[1, 1].plot(nominal, nominal, linestyle="--", label="Target")
    axes[1, 1].set(title="(e) Risk-control validity", xlabel="Target risk", ylabel="Observed risk")
    axes[1, 1].legend(fontsize=8)

    features = ["JS total", "Curvature", "Rank flips", "Margin erosion", "Semantic drift", "Modality conflict", "Entropy"]
    importance = np.sort(rng.uniform(0.03, 0.24, len(features)))[::-1]
    axes[1, 2].barh(features[::-1], importance[::-1])
    axes[1, 2].set(title="(f) Permutation importance with CIs", xlabel="Drop in held-out AUROC")
    _finish(fig, output)


def generate_tpami_suite(output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    specs = [
        ("fig37_precision_path_geometry", figure37_precision_path_geometry),
        ("fig38_heldout_generalization", figure38_heldout_generalization),
        ("fig39_risk_cost_frontier", figure39_risk_cost_frontier),
        ("fig40_qualitative_interventions", figure40_qualitative_interventions),
        ("fig41_scope_and_evidence", figure41_scope_and_evidence),
        ("fig42_statistical_evidence", figure42_statistical_evidence),
        ("fig43_method_architecture", figure43_method_architecture),
    ]
    paths: list[Path] = []
    for name, fn in specs:
        target = out / name
        fn(target)
        paths.extend([target.with_suffix(".png"), target.with_suffix(".pdf")])
    return paths


def figure43_method_architecture(output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.5), gridspec_kw={"height_ratios": [1.15, 0.85]})
    fig.suptitle("Fig. 43 — AdvQFuse field-level architecture and progressive deployment policy")

    ax = axes[0]
    ax.axis("off")
    boxes = [
        (0.02, 0.58, 0.13, 0.24, "Image + text\ninput"),
        (0.19, 0.58, 0.17, 0.24, "Aligned precision path\nq1 < ... < qK"),
        (0.40, 0.67, 0.16, 0.16, "Output geometry\nJS, curvature, flips"),
        (0.40, 0.43, 0.16, 0.16, "Cross-modal probes\nmask, paraphrase, conflict"),
        (0.60, 0.55, 0.16, 0.27, "Precision-Path\nSet Encoder\n(PPSE)"),
        (0.80, 0.64, 0.17, 0.18, "Action-conditioned\npost-action risk"),
        (0.80, 0.37, 0.17, 0.18, "Finite-sample\ngroup-risk calibration"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    arrows = [
        ((0.15, 0.70), (0.19, 0.70)),
        ((0.36, 0.70), (0.40, 0.75)),
        ((0.36, 0.66), (0.40, 0.51)),
        ((0.56, 0.75), (0.60, 0.69)),
        ((0.56, 0.51), (0.60, 0.63)),
        ((0.76, 0.69), (0.80, 0.73)),
        ((0.76, 0.62), (0.80, 0.46)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "linewidth": 1.8})
    ax.text(0.275, 0.48, "Controlled parameter intervention", ha="center", va="center", fontsize=9)
    ax.text(0.885, 0.28, "Certified thresholds are learned only on calibration data", ha="center", fontsize=9)

    ax2 = axes[1]
    ax2.axis("off")
    stages = [
        (0.025, "Low-cost pass", "Output + cheap path/probes"),
        (0.225, "Accept", "Risk UCB <= alpha"),
        (0.395, "Reobserve", "New crop/view/preprocess"),
        (0.565, "Escalate", "Higher precision"),
        (0.735, "Ensemble", "Selected path points"),
        (0.865, "Abstain", "No certified action"),
    ]
    for i, (x, title, subtitle) in enumerate(stages):
        w = 0.13 if i == 0 else (0.11 if i < 5 else 0.105)
        ax2.add_patch(Rectangle((x, 0.38), w, 0.32, fill=False, linewidth=2))
        ax2.text(x + w / 2, 0.58, title, ha="center", va="center", fontsize=10, weight="bold")
        ax2.text(x + w / 2, 0.46, subtitle, ha="center", va="center", fontsize=8, wrap=True)
    arrow_segments = [(0.155, 0.225), (0.335, 0.395), (0.505, 0.565), (0.675, 0.735), (0.845, 0.865)]
    for start_x, end_x in arrow_segments:
        ax2.annotate("", xy=(end_x, 0.54), xytext=(start_x, 0.54), arrowprops={"arrowstyle": "->", "linewidth": 1.6})
    ax2.text(0.5, 0.18, "The policy pays for every acquired path point and probe; comparisons must use matched expected compute.", ha="center", fontsize=10)
    _finish(fig, output)

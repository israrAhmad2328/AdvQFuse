from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path("figures/v4_extended")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260725)


def finish(fig: plt.Figure, filename: str) -> None:
    fig.text(
        0.5,
        0.012,
        "SYNTHETIC LAYOUT DEMO - NOT MEASURED RESULTS - REPLACE WITH LOCKED TEST OUTPUTS",
        ha="center",
        va="bottom",
        fontsize=8,
        alpha=0.75,
        weight="bold",
    )
    fig.savefig(OUT / f"{filename}.png", dpi=190, bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def scene(seed: int, size: int = 128, flood: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.48, 0.14, (size, size, 3))
    yy, xx = np.mgrid[:size, :size]
    base += 0.15 * np.sin(xx / 13)[..., None] + 0.1 * np.cos(yy / 17)[..., None]
    for _ in range(10):
        x, y = rng.integers(4, size - 24, 2)
        w, h = rng.integers(6, 25, 2)
        base[y : y + h, x : x + w] += rng.uniform(-0.35, 0.35, 3)
    if flood:
        mask = (yy > 0.55 * size + 8 * np.sin(xx / 11))
        base[mask, 2] += 0.28
        base[mask, 0] -= 0.12
    return np.clip(base, 0, 1)


def heat(size: int = 128, centers: list[tuple[float, float, float]] | None = None) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    z = np.zeros((size, size))
    centers = centers or [(0.3, 0.4, 0.12), (0.7, 0.65, 0.18)]
    for cx, cy, sigma in centers:
        z += np.exp(-((xx / size - cx) ** 2 + (yy / size - cy) ** 2) / (2 * sigma**2))
    return z / max(z.max(), 1e-8)


def fig44() -> None:
    fig, axes = plt.subplots(4, 5, figsize=(16, 11), constrained_layout=True)
    cases = ["Stable accept", "Attack detected", "Reobserve", "Safe abstain"]
    precisions = ["BF16", "INT8", "INT4", "INT2"]
    for r, case in enumerate(cases):
        img = scene(10 + r, flood=r == 2)
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{case}\nInput")
        axes[r, 0].axis("off")
        attacked = img.copy()
        if r > 0:
            attacked[25:62, 75:112] = np.clip(attacked[25:62, 75:112] + 0.45, 0, 1)
        axes[r, 1].imshow(attacked)
        axes[r, 1].imshow(heat(128, [(0.72, 0.35, 0.13)]), alpha=0.35)
        axes[r, 1].set_title("Perturbation / evidence")
        axes[r, 1].axis("off")
        p = np.clip(np.array([0.90, 0.87, 0.82, 0.79]) - r * np.array([0.03, 0.05, 0.11, 0.16]), 0.1, 0.98)
        axes[r, 2].plot(precisions, p, marker="o")
        axes[r, 2].fill_between(precisions, p - 0.04, p + 0.04, alpha=0.2)
        axes[r, 2].set_ylim(0, 1)
        axes[r, 2].set_title("Precision path")
        axes[r, 2].set_ylabel("Correct-answer probability")
        u = np.clip(np.array([0.12, 0.18, 0.10, 0.08]) + r * np.array([0.13, 0.12, 0.18, 0.17]), 0, 1)
        axes[r, 3].bar(["sensor", "conflict", "path", "attack"], u)
        axes[r, 3].tick_params(axis="x", rotation=35)
        axes[r, 3].set_ylim(0, 1)
        axes[r, 3].set_title("Uncertainty audit")
        axes[r, 4].axis("off")
        action = ["ACCEPT INT4", "ESCALATE", "REOBSERVE", "ABSTAIN"][r]
        answer = ["Yes", "3", "Flooded road", "No answer"][r]
        txt = (
            f"Action: {action}\n\n"
            f"Final answer: {answer}\n\n"
            f"Risk UCB: {0.018 + 0.012*r:.3f}\n"
            f"Cost units: {1.0 + 0.8*r:.1f}\n"
            f"Reason: path curvature +\nmodality evidence"
        )
        axes[r, 4].text(0.05, 0.92, txt, va="top", fontsize=11, bbox=dict(boxstyle="round", alpha=0.15))
        axes[r, 4].set_title("Decision card")
    fig.suptitle("Figure 44. Per-sample precision-path qualitative atlas", fontsize=18, weight="bold")
    finish(fig, "fig44_precision_path_case_atlas")


def fig45() -> None:
    attacks = ["Patch", "PGD transfer", "Prompt injection", "Modality conflict"]
    stages = ["Clean", "Attacked", "Localized", "Intervened", "Final audit"]
    fig, axes = plt.subplots(len(attacks), len(stages), figsize=(16, 11), constrained_layout=True)
    for r, attack in enumerate(attacks):
        base = scene(30 + r, flood=r == 3)
        for c, stage in enumerate(stages):
            ax = axes[r, c]
            show = base.copy()
            if c in {1, 2}:
                show[20 + 8 * r : 55 + 8 * r, 62:102] = np.clip(show[20 + 8 * r : 55 + 8 * r, 62:102] + 0.5, 0, 1)
            if c == 3:
                show = np.clip(0.75 * base + 0.25 * show, 0, 1)
            ax.imshow(show)
            if c == 2:
                ax.imshow(heat(128, [(0.65, 0.35 + 0.06 * r, 0.12)]), alpha=0.45)
            if c == 4:
                ax.text(0.03, 0.97, ["ACCEPT", "ESCALATE", "REOBSERVE", "ABSTAIN"][r], transform=ax.transAxes,
                        va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            if r == 0:
                ax.set_title(stage)
            if c == 0:
                ax.set_ylabel(attack, fontsize=11, weight="bold")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Figure 45. Multi-attack qualitative recovery matrix", fontsize=18, weight="bold")
    finish(fig, "fig45_multi_attack_recovery_matrix")


def fig46() -> None:
    datasets = ["VQAv2", "GQA", "TextVQA", "EarthVQA", "FloodNet", "RSVQA-HR", "UAV-OBB-QA", "SEN12MS"]
    fig, axes = plt.subplots(4, 4, figsize=(15, 13), constrained_layout=True)
    for i, dataset in enumerate(datasets):
        r = i // 2
        c0 = (i % 2) * 2
        ax_img, ax_card = axes[r, c0], axes[r, c0 + 1]
        img = scene(60 + i, flood=dataset == "FloodNet")
        ax_img.imshow(img)
        if dataset in {"UAV-OBB-QA", "EarthVQA"}:
            for j in range(5):
                x, y = 12 + j * 19, 25 + (j % 2) * 30
                ax_img.add_patch(Rectangle((x, y), 14, 7, fill=False, lw=1.5))
        ax_img.set_title(dataset)
        ax_img.axis("off")
        path = np.clip(0.85 - 0.12 * RNG.random(4) - (i % 3) * 0.05, 0.2, 0.98)
        action = ["accept", "escalate", "reobserve", "abstain"][i % 4]
        ax_card.axis("off")
        ax_card.text(
            0.02, 0.95,
            f"Question: synthetic case {i+1}\n"
            f"GT: class-{i%5}\n"
            f"Path: {', '.join(f'{x:.2f}' for x in path)}\n"
            f"Action: {action}\n"
            f"Correct after action: {'yes' if i not in {2,6} else 'no'}\n"
            f"Residual risk: {0.01 + 0.008*i:.3f}",
            va="top", fontsize=10, bbox=dict(boxstyle="round", alpha=0.12)
        )
        ax_card.set_title("Audit trace")
    fig.suptitle("Figure 46. Cross-domain and cross-dataset qualitative atlas", fontsize=18, weight="bold")
    finish(fig, "fig46_cross_dataset_qualitative_atlas")


def fig47() -> None:
    groups = ["General VQA", "OCR", "Relational", "Remote sensing", "Clean", "Adversarial"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for i, (ax, group) in enumerate(zip(axes.flat, groups)):
        conf = np.linspace(0.05, 0.95, 10)
        gap = (i - 2.5) * 0.008
        acc = np.clip(conf + gap + RNG.normal(0, 0.035, len(conf)), 0, 1)
        ax.plot([0, 1], [0, 1], linestyle="--", label="ideal")
        ax.plot(conf, acc, marker="o", label="AdvQFuse")
        ax.fill_between(conf, np.clip(acc - 0.04, 0, 1), np.clip(acc + 0.04, 0, 1), alpha=0.2)
        ax.set_title(f"{group}\nECE={abs(gap)+0.025:.3f}, n={600+i*120}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Empirical accuracy")
        ax.legend(fontsize=8)
    fig.suptitle("Figure 47. Subgroup calibration small-multiple dashboard", fontsize=18, weight="bold")
    finish(fig, "fig47_subgroup_calibration_dashboard")


def fig48() -> None:
    budgets = np.arange(1, 7)
    actions = np.array([
        [0.82, 0.10, 0.04, 0.04],
        [0.70, 0.17, 0.08, 0.05],
        [0.61, 0.22, 0.11, 0.06],
        [0.53, 0.27, 0.13, 0.07],
        [0.48, 0.29, 0.15, 0.08],
        [0.44, 0.31, 0.16, 0.09],
    ])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ax = axes[0, 0]
    bottom = np.zeros(len(budgets))
    for j, label in enumerate(["accept", "escalate", "reobserve", "abstain"]):
        ax.bar(budgets, actions[:, j], bottom=bottom, label=label)
        bottom += actions[:, j]
    ax.set_title("Action allocation by budget"); ax.set_xlabel("Compute budget"); ax.set_ylabel("Fraction"); ax.legend(fontsize=8)
    axes[0, 1].plot(budgets, 0.79 + 0.035 * np.log1p(budgets), marker="o", label="accuracy")
    axes[0, 1].plot(budgets, 0.08 / budgets + 0.01, marker="s", label="selective risk")
    axes[0, 1].set_title("Accuracy-risk response"); axes[0, 1].legend()
    axes[0, 2].plot(budgets, 45 + 34 * budgets, marker="o", label="latency (ms)")
    axes[0, 2].plot(budgets, 0.4 + 0.28 * budgets, marker="s", label="energy (J)")
    axes[0, 2].set_title("Measured-cost template"); axes[0, 2].legend()
    utility = (0.79 + 0.035 * np.log1p(budgets)) - 0.0015 * (45 + 34 * budgets) - 0.6 * (0.08 / budgets + 0.01)
    axes[1, 0].plot(budgets, utility, marker="D")
    axes[1, 0].axvline(budgets[np.argmax(utility)], linestyle="--")
    axes[1, 0].set_title("Utility and selected operating point"); axes[1, 0].set_xlabel("Compute budget")
    group_risk = RNG.uniform(0.015, 0.08, (5, len(budgets))) / np.sqrt(budgets)
    im = axes[1, 1].imshow(group_risk, aspect="auto")
    axes[1, 1].set_yticks(range(5), ["OCR", "count", "attack", "RS", "conflict"])
    axes[1, 1].set_xticks(range(len(budgets)), budgets)
    axes[1, 1].set_title("Worst-group risk"); fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    axes[1, 2].axis("off")
    x = [0.08, 0.37, 0.67]
    labels = ["Low precision", "Controller", "Action"]
    for xx, label in zip(x, labels):
        box = FancyBboxPatch((xx, 0.45), 0.22, 0.16, boxstyle="round,pad=0.02", transform=axes[1, 2].transAxes, alpha=0.2)
        axes[1, 2].add_patch(box); axes[1, 2].text(xx + 0.11, 0.53, label, ha="center", va="center", transform=axes[1, 2].transAxes)
    for a, b in zip(x[:-1], x[1:]):
        axes[1, 2].add_patch(FancyArrowPatch((a + 0.22, 0.53), (b, 0.53), transform=axes[1, 2].transAxes, arrowstyle="->", mutation_scale=16))
    axes[1, 2].text(0.5, 0.28, "Every probe is charged to the budget", ha="center", transform=axes[1, 2].transAxes)
    axes[1, 2].set_title("Matched-compute accounting")
    fig.suptitle("Figure 48. Compute-budget and action-routing analysis", fontsize=18, weight="bold")
    finish(fig, "fig48_budget_action_routing")


def fig49() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    base = scene(88)
    patch = base.copy(); patch[32:72, 78:118] = 1 - patch[32:72, 78:118]
    axes[0, 0].imshow(base); axes[0, 0].set_title("Clean")
    axes[0, 1].imshow(patch); axes[0, 1].set_title("Adversarial patch")
    axes[0, 2].imshow(patch); axes[0, 2].imshow(heat(128, [(0.75, 0.42, 0.12)]), alpha=0.5); axes[0, 2].set_title("Localized evidence")
    recovered = base * 0.9 + patch * 0.1
    axes[0, 3].imshow(recovered); axes[0, 3].set_title("Counterfactual reobservation")
    for ax in axes[0]: ax.axis("off")
    steps = np.arange(0, 101, 10)
    axes[1, 0].plot(steps, 0.15 + 0.75 * (1 - np.exp(-steps / 28)), marker="o")
    axes[1, 0].set_title("Attack optimization"); axes[1, 0].set_xlabel("Queries"); axes[1, 0].set_ylabel("Attack success proxy")
    axes[1, 1].plot(steps, 0.12 + 0.62 * (1 - np.exp(-steps / 35)), label="path disagreement")
    axes[1, 1].plot(steps, 0.08 + 0.48 * (1 - np.exp(-steps / 25)), label="semantic conflict")
    axes[1, 1].set_title("Reliability signals"); axes[1, 1].legend(fontsize=8)
    axes[1, 2].bar(["BF16", "INT8", "INT4", "INT2"], [0.78, 0.73, 0.42, 0.21])
    axes[1, 2].set_ylim(0, 1); axes[1, 2].set_title("Correct-answer path")
    axes[1, 3].axis("off")
    axes[1, 3].text(0.03, 0.94, "Detected: yes\nAction: reobserve + escalate\nFinal answer: correct\nRisk UCB: 0.027\nQueries charged: 100\nResidual failure: patch outside probe", va="top", fontsize=12, bbox=dict(boxstyle="round", alpha=0.15))
    axes[1, 3].set_title("Attack audit")
    fig.suptitle("Figure 49. Adversarial patch localization and intervention sequence", fontsize=18, weight="bold")
    finish(fig, "fig49_patch_localization_sequence")


def fig50() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    optical = scene(100)
    sar = np.mean(optical, axis=2) + 0.18 * RNG.normal(size=(128, 128))
    sar = np.clip(sar, 0, 1)
    conflict = np.abs(np.mean(optical, axis=2) - sar)
    fused = 0.55 * np.mean(optical, axis=2) + 0.45 * sar
    for ax, data, title in zip(axes[0], [optical, sar, conflict, fused], ["Sentinel-2 optical", "Sentinel-1 SAR", "Cross-sensor conflict", "Reliability-weighted fusion"]):
        ax.imshow(data); ax.set_title(title); ax.axis("off")
    axes[1, 0].bar(["optical", "SAR", "agreement", "path"], [0.61, 0.79, 0.42, 0.35])
    axes[1, 0].set_ylim(0, 1); axes[1, 0].set_title("Evidence reliability")
    axes[1, 1].plot(["BF16", "INT8", "INT4", "INT2"], [0.84, 0.82, 0.68, 0.49], marker="o", label="forest")
    axes[1, 1].plot(["BF16", "INT8", "INT4", "INT2"], [0.10, 0.13, 0.25, 0.39], marker="s", label="urban")
    axes[1, 1].set_title("Answer trajectory"); axes[1, 1].legend()
    axes[1, 2].imshow(conflict > np.quantile(conflict, 0.78)); axes[1, 2].set_title("Conflict mask"); axes[1, 2].axis("off")
    axes[1, 3].axis("off")
    axes[1, 3].text(0.04, 0.94, "Question: dominant land cover?\nOptical answer: urban\nSAR answer: forest\nFused answer: forest\nAction: request SAR-preserving path\nGround truth: forest\nDecision: correct", va="top", fontsize=12, bbox=dict(boxstyle="round", alpha=0.15))
    axes[1, 3].set_title("Multi-sensor decision trace")
    fig.suptitle("Figure 50. Optical-SAR conflict and fusion casebook", fontsize=18, weight="bold")
    finish(fig, "fig50_optical_sar_conflict_case")


def fig51() -> None:
    prompts = ["original", "paraphrase A", "paraphrase B", "negated distractor", "counterfactual"]
    precisions = ["BF16", "INT8", "INT4", "INT2"]
    scores = np.clip(0.82 + RNG.normal(0, 0.12, (len(prompts), len(precisions))) - np.arange(4)[None, :] * 0.07, 0.02, 0.98)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    im = axes[0, 0].imshow(scores, vmin=0, vmax=1, aspect="auto")
    axes[0, 0].set_xticks(range(4), precisions); axes[0, 0].set_yticks(range(5), prompts); axes[0, 0].set_title("Correct-answer probability")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)
    flips = (scores < 0.5).astype(int)
    axes[0, 1].imshow(flips, vmin=0, vmax=1, aspect="auto")
    axes[0, 1].set_xticks(range(4), precisions); axes[0, 1].set_yticks(range(5), prompts); axes[0, 1].set_title("Answer flip lattice")
    entropy = -(scores * np.log(scores + 1e-9) + (1 - scores) * np.log(1 - scores + 1e-9))
    axes[0, 2].plot(precisions, entropy.T, alpha=0.65)
    axes[0, 2].set_title("Prompt-conditioned entropy")
    axes[1, 0].bar(prompts, np.abs(np.diff(scores, axis=1)).sum(axis=1)); axes[1, 0].tick_params(axis="x", rotation=30); axes[1, 0].set_title("Path total variation")
    curvature = np.abs(scores[:, 2:] - 2 * scores[:, 1:-1] + scores[:, :-2]).sum(axis=1)
    axes[1, 1].bar(prompts, curvature); axes[1, 1].tick_params(axis="x", rotation=30); axes[1, 1].set_title("Path curvature")
    axes[1, 2].axis("off")
    axes[1, 2].text(0.04, 0.95, "Audit rule\n\n- Stable across precision\n- Unstable under negation\n- High curvature at INT4\n- Prompt conflict detected\n\nAction: abstain or ask clarification", va="top", fontsize=12, bbox=dict(boxstyle="round", alpha=0.15))
    axes[1, 2].set_title("Semantic path diagnosis")
    fig.suptitle("Figure 51. Prompt-by-precision answer-trajectory lattice", fontsize=18, weight="bold")
    finish(fig, "fig51_answer_trajectory_lattice")


def fig52() -> None:
    groups = ["VQAv2", "GQA", "TextVQA", "EarthVQA", "FloodNet", "Patch", "Prompt", "Conflict"]
    risk = np.array([0.031, 0.036, 0.042, 0.047, 0.053, 0.058, 0.049, 0.061])
    ucb = risk + np.array([0.011, 0.010, 0.013, 0.014, 0.016, 0.018, 0.015, 0.019])
    coverage = np.array([0.83, 0.81, 0.76, 0.72, 0.68, 0.63, 0.71, 0.59])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    x = np.arange(len(groups))
    axes[0, 0].bar(x, risk, yerr=ucb-risk, capsize=3); axes[0, 0].axhline(0.08, linestyle="--"); axes[0, 0].set_xticks(x, groups, rotation=35); axes[0, 0].set_title("Empirical risk with one-sided UCB")
    axes[0, 1].bar(x, coverage); axes[0, 1].set_xticks(x, groups, rotation=35); axes[0, 1].set_ylim(0, 1); axes[0, 1].set_title("Coverage by supported group")
    violation = RNG.binomial(1, np.clip((ucb - 0.06) * 3, 0.02, 0.4), (5, len(groups)))
    axes[0, 2].imshow(violation, aspect="auto", vmin=0, vmax=1); axes[0, 2].set_yticks(range(5), [f"seed {i}" for i in range(1,6)]); axes[0, 2].set_xticks(range(len(groups)), groups, rotation=35); axes[0, 2].set_title("Bound-violation audit")
    action_mix = RNG.dirichlet([6, 2, 1.5, 0.8], len(groups))
    bottom = np.zeros(len(groups))
    for j, name in enumerate(["accept", "escalate", "reobserve", "abstain"]):
        axes[1, 0].bar(x, action_mix[:, j], bottom=bottom, label=name); bottom += action_mix[:, j]
    axes[1, 0].set_xticks(x, groups, rotation=35); axes[1, 0].set_title("Group action composition"); axes[1, 0].legend(fontsize=8)
    axes[1, 1].scatter(coverage, risk, s=90)
    for i, g in enumerate(groups): axes[1, 1].annotate(g, (coverage[i], risk[i]), fontsize=8)
    axes[1, 1].set_xlabel("Coverage"); axes[1, 1].set_ylabel("Risk"); axes[1, 1].set_title("Risk-coverage group frontier")
    axes[1, 2].axis("off")
    worst = groups[int(np.argmax(ucb))]
    axes[1, 2].text(0.04, 0.95, f"Target risk: 0.08\nWorst UCB group: {worst}\nAll UCBs below target: yes\nCalibration samples: 4,800\nSeeds: 5\nUnsupported tiny groups: excluded and reported\nDecision: calibration passes synthetic check", va="top", fontsize=12, bbox=dict(boxstyle="round", alpha=0.15))
    axes[1, 2].set_title("Risk-control report card")
    fig.suptitle("Figure 52. Group-wise selective-risk control audit", fontsize=18, weight="bold")
    finish(fig, "fig52_group_risk_control_audit")


def fig53() -> None:
    families = ["Qwen-VL", "LLaVA-OV", "InternVL", "MiniCPM-V", "Bonsai"]
    precisions = ["BF16", "INT8", "INT4", "INT2", "ternary", "binary"]
    acc = np.clip(0.88 - np.arange(len(precisions))[None, :] * 0.045 + RNG.normal(0, 0.025, (len(families), len(precisions))), 0.45, 0.95)
    acc[-1, :4] = np.nan
    auroc = np.clip(0.69 + RNG.normal(0, 0.04, acc.shape) + np.arange(len(precisions))[None, :] * 0.018, 0.55, 0.92)
    auroc[-1, :4] = np.nan
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    im0 = axes[0, 0].imshow(acc, vmin=0.45, vmax=0.95, aspect="auto"); axes[0, 0].set_xticks(range(6), precisions, rotation=30); axes[0, 0].set_yticks(range(5), families); axes[0, 0].set_title("Task accuracy lattice"); fig.colorbar(im0, ax=axes[0,0], fraction=0.046)
    im1 = axes[0, 1].imshow(auroc, vmin=0.55, vmax=0.92, aspect="auto"); axes[0, 1].set_xticks(range(6), precisions, rotation=30); axes[0, 1].set_yticks(range(5), families); axes[0, 1].set_title("Failure-prediction AUROC"); fig.colorbar(im1, ax=axes[0,1], fraction=0.046)
    axes[0, 2].imshow(np.isnan(acc), vmin=0, vmax=1, aspect="auto"); axes[0, 2].set_xticks(range(6), precisions, rotation=30); axes[0, 2].set_yticks(range(5), families); axes[0, 2].set_title("Available precision points")
    heldout = 0.02 + RNG.normal(0.015, 0.012, (4,5))
    im2 = axes[1, 0].imshow(heldout, aspect="auto"); axes[1, 0].set_xticks(range(5), families, rotation=30); axes[1, 0].set_yticks(range(4), ["model", "dataset", "attack", "precision"]); axes[1, 0].set_title("Held-out AUROC gain"); fig.colorbar(im2, ax=axes[1,0], fraction=0.046)
    axes[1, 1].boxplot([RNG.normal(0.035 + i*0.004, 0.018, 80) for i in range(5)], tick_labels=families, showfliers=False); axes[1, 1].tick_params(axis="x", rotation=30); axes[1, 1].axhline(0, linestyle="--"); axes[1, 1].set_title("Paired gain distribution")
    axes[1, 2].axis("off"); axes[1, 2].text(0.03,0.95,"Required claim check\n\n✓ Four independent families\n✓ Precision ladder\n✓ Missing points explicit\n✓ Leave-one-family-out\n✓ Bonsai external only\n✓ Main effect survives removal\n\nNo interpolation across absent checkpoints",va="top",fontsize=12,bbox=dict(boxstyle="round",alpha=0.15)); axes[1,2].set_title("Breadth audit")
    fig.suptitle("Figure 53. Model-family and precision-lattice evidence", fontsize=18, weight="bold")
    finish(fig, "fig53_model_precision_lattice")


def fig54() -> None:
    variants = ["Full input", "Image only", "Text only", "Masked image", "Contradictory text", "Reobserved"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
    base = scene(130, flood=True)
    for i, variant in enumerate(variants):
        r = i // 2
        c = (i % 2) * 2
        img = base.copy()
        if variant == "Masked image": img[35:90, 40:105] *= 0.15
        if variant == "Image only": pass
        if variant == "Text only": img[:] = 0.92
        if variant == "Reobserved": img = np.clip(base * 1.12, 0, 1)
        axes[r, c].imshow(img); axes[r, c].set_title(variant); axes[r, c].axis("off")
        axes[r, c+1].axis("off")
        prob = [0.84, 0.71, 0.46, 0.39, 0.22, 0.90][i]
        axes[r, c+1].text(0.03,0.95,f"Answer: {'flooded road' if prob>0.5 else 'non-flooded road'}\nCorrect prob.: {prob:.2f}\nConflict: {abs(0.72-prob):.2f}\nPath TV: {0.08+0.06*i:.2f}\nAction: {['accept','escalate','abstain','reobserve','abstain','accept'][i]}",va="top",fontsize=11,bbox=dict(boxstyle="round",alpha=0.15)); axes[r,c+1].set_title("Counterfactual audit")
    fig.suptitle("Figure 54. Cross-modal counterfactual and modality-ablation gallery", fontsize=18, weight="bold")
    finish(fig, "fig54_modality_ablation_gallery")


def fig55() -> None:
    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    ax0 = fig.add_subplot(gs[0, :2]); ax1 = fig.add_subplot(gs[0, 2:]); ax2 = fig.add_subplot(gs[1:, 0]); ax3 = fig.add_subplot(gs[1:, 1]); ax4 = fig.add_subplot(gs[1:, 2]); ax5 = fig.add_subplot(gs[1:, 3])
    cats = ["OCR", "count", "small object", "prompt", "sensor conflict", "calibration"]
    counts = [41, 57, 68, 32, 49, 27]
    ax0.barh(cats, counts); ax0.set_title("Residual failure taxonomy"); ax0.set_xlabel("Locked-test failures")
    causes = np.array([[24, 8, 4], [14, 27, 7], [9, 11, 31], [18, 5, 8], [7, 29, 13], [12, 6, 9]])
    im = ax1.imshow(causes, aspect="auto"); ax1.set_yticks(range(len(cats)), cats); ax1.set_xticks(range(3), ["perception", "fusion", "controller"]); ax1.set_title("Root-cause attribution"); fig.colorbar(im, ax=ax1, fraction=0.046)
    examples = [
        ("Undetected failure", "TextVQA", "confident wrong OCR", "add OCR probe"),
        ("Recovered", "FloodNet", "water ambiguity", "reobserve"),
        ("Safe abstention", "SEN12MS", "sensor conflict", "request modality"),
        ("Cost failure", "GQA", "unneeded escalation", "improve utility model"),
    ]
    for ax, (status, ds, cause, mitigation), seed in zip([ax2,ax3,ax4,ax5], examples, [150,151,152,153]):
        ax.imshow(scene(seed, flood=ds=="FloodNet")); ax.axis("off")
        ax.set_title(f"{status}: {ds}")
        ax.text(0.02,0.02,f"Cause: {cause}\nMitigation: {mitigation}\nSelection: pre-registered rule",transform=ax.transAxes,fontsize=10,bbox=dict(boxstyle="round",facecolor="white",alpha=0.82))
    fig.suptitle("Figure 55. Reviewer-ready residual-failure and mitigation audit", fontsize=18, weight="bold")
    finish(fig, "fig55_residual_failure_audit")


def main() -> None:
    for fn in [fig44, fig45, fig46, fig47, fig48, fig49, fig50, fig51, fig52, fig53, fig54, fig55]:
        fn()
    print(f"Generated 12 v4 layouts in {OUT.resolve()}")


if __name__ == "__main__":
    main()

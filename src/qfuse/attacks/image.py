from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PatchAttackResult:
    patch: np.ndarray
    best_score: float
    queries: int
    history: tuple[float, ...]


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Gradient attacks require PyTorch. Install with `pip install -e .[attacks]`."
        ) from exc
    return torch


def fgsm_linf(model, image, target, loss_fn, epsilon: float):
    """Untargeted FGSM for a differentiable surrogate model.

    `image` must be a normalized BCHW tensor. The returned tensor is clipped to
    [0, 1]. This function is intended for transfer attacks; Bonsai GGUF itself is
    treated as a black-box target.
    """
    torch = _require_torch()
    x = image.detach().clone().requires_grad_(True)
    logits = model(x)
    loss = loss_fn(logits, target)
    grad = torch.autograd.grad(loss, x)[0]
    return torch.clamp(x + epsilon * grad.sign(), 0.0, 1.0).detach()


def pgd_linf(
    model,
    image,
    target,
    loss_fn,
    epsilon: float,
    step_size: float,
    steps: int,
    random_start: bool = True,
):
    """Untargeted L-infinity PGD for a differentiable surrogate model."""
    torch = _require_torch()
    x0 = image.detach()
    if random_start:
        x = torch.clamp(x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon), 0, 1)
    else:
        x = x0.clone()
    for _ in range(int(steps)):
        x.requires_grad_(True)
        loss = loss_fn(model(x), target)
        grad = torch.autograd.grad(loss, x)[0]
        x = x.detach() + step_size * grad.sign()
        x = torch.max(torch.min(x, x0 + epsilon), x0 - epsilon)
        x = torch.clamp(x, 0.0, 1.0)
    return x.detach()


def random_patch(
    height: int,
    width: int,
    channels: int = 3,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(height, width, channels)).astype(np.float32)


def apply_patch(
    image: Image.Image,
    patch: np.ndarray,
    x: int,
    y: int,
    opacity: float = 1.0,
) -> Image.Image:
    """Apply an RGB patch to a PIL image with clipping-safe placement."""
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    p = np.asarray(patch, dtype=np.float32)
    if p.ndim != 3 or p.shape[2] != 3:
        raise ValueError("patch must have shape H x W x 3")
    p = np.clip(p, 0.0, 1.0)
    h, w = base.shape[:2]
    ph, pw = p.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, x0 + pw), min(h, y0 + ph)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("patch does not overlap the image")
    alpha = float(np.clip(opacity, 0.0, 1.0))
    crop = p[: y1 - y0, : x1 - x0]
    base[y0:y1, x0:x1] = (1 - alpha) * base[y0:y1, x0:x1] + alpha * crop
    return Image.fromarray(np.round(base * 255).astype(np.uint8))


def spsa_patch_attack(
    initial_patch: np.ndarray,
    score_fn: Callable[[np.ndarray], float],
    iterations: int = 100,
    samples_per_step: int = 8,
    step_size: float = 0.02,
    perturbation: float = 0.01,
    seed: int = 0,
    maximize: bool = True,
) -> PatchAttackResult:
    """Query-based SPSA optimization of a patch against a black-box score.

    The caller defines `score_fn`. For an untargeted attack, a suitable score is
    the negative probability of the correct answer or a semantic mismatch score.
    It can call a local Bonsai server, but caching is strongly recommended.
    """
    rng = np.random.default_rng(seed)
    patch = np.clip(np.asarray(initial_patch, dtype=np.float32), 0.0, 1.0)
    best = patch.copy()
    best_score = float(score_fn(best))
    queries = 1
    history = [best_score]
    direction_sign = 1.0 if maximize else -1.0

    for t in range(int(iterations)):
        grad = np.zeros_like(patch, dtype=np.float32)
        c_t = perturbation / ((t + 1) ** 0.101)
        a_t = step_size / ((t + 10) ** 0.602)
        for _ in range(int(samples_per_step)):
            delta = rng.choice((-1.0, 1.0), size=patch.shape).astype(np.float32)
            plus = np.clip(patch + c_t * delta, 0.0, 1.0)
            minus = np.clip(patch - c_t * delta, 0.0, 1.0)
            s_plus = float(score_fn(plus))
            s_minus = float(score_fn(minus))
            queries += 2
            grad += ((s_plus - s_minus) / (2.0 * c_t)) * delta
        grad /= max(1, int(samples_per_step))
        patch = np.clip(patch + direction_sign * a_t * np.sign(grad), 0.0, 1.0)
        current = float(score_fn(patch))
        queries += 1
        better = current > best_score if maximize else current < best_score
        if better:
            best_score = current
            best = patch.copy()
        history.append(best_score)

    return PatchAttackResult(best, best_score, queries, tuple(history))

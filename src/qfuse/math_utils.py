from __future__ import annotations

import numpy as np


def normalize(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, None)
    return p / p.sum(axis=-1, keepdims=True)


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    q = normalize(np.asarray(p, dtype=float), eps=eps)
    return float(-np.sum(q * np.log(q + eps)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = normalize(np.asarray(p, dtype=float), eps=eps)
    q = normalize(np.asarray(q, dtype=float), eps=eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def margin(p: np.ndarray) -> float:
    p = np.sort(normalize(np.asarray(p, dtype=float)))[::-1]
    return float(p[0] - p[1])


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=float) / max(float(temperature), 1e-8)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=float)
    out = np.empty_like(x_arr)
    pos = x_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    if np.ndim(x) == 0:
        return float(out)
    return out

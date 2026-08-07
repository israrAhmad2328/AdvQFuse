from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def apply_corruption(
    image: Image.Image,
    kind: str,
    severity: int,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    if severity not in {1, 2, 3, 4, 5}:
        raise ValueError("severity must be an integer from 1 to 5")
    rng = rng or np.random.default_rng()
    image = image.convert("RGB")
    if kind == "gaussian_noise":
        arr = np.asarray(image).astype(np.float32) / 255.0
        sigma = 0.025 * severity
        arr = np.clip(arr + rng.normal(0.0, sigma, arr.shape), 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=0.6 * severity))
    if kind == "low_light":
        return ImageEnhance.Brightness(image).enhance(max(0.15, 1.0 - 0.15 * severity))
    if kind == "contrast":
        return ImageEnhance.Contrast(image).enhance(max(0.15, 1.0 - 0.15 * severity))
    if kind == "occlusion":
        arr = np.asarray(image).copy()
        h, w = arr.shape[:2]
        frac = 0.08 + 0.05 * severity
        oh, ow = max(1, int(h * frac)), max(1, int(w * frac))
        y = int(rng.integers(0, max(h - oh + 1, 1)))
        x = int(rng.integers(0, max(w - ow + 1, 1)))
        arr[y : y + oh, x : x + ow] = 0
        return Image.fromarray(arr)
    if kind == "grayscale":
        return ImageOps.grayscale(image).convert("RGB")
    raise ValueError(f"unsupported corruption: {kind}")


def corrupt_file(input_path: str | Path, output_path: str | Path, kind: str, severity: int, seed: int = 0) -> None:
    image = Image.open(input_path)
    out = apply_corruption(image, kind=kind, severity=severity, rng=np.random.default_rng(seed))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def apply_remote_sensing_corruption(
    image: Image.Image,
    kind: str,
    severity: int,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Remote-sensing-specific corruptions with five severity levels."""
    if severity not in {1, 2, 3, 4, 5}:
        raise ValueError("severity must be an integer from 1 to 5")
    rng = rng or np.random.default_rng()
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0

    if kind == "speckle":
        sigma = 0.06 * severity
        noisy = arr + arr * rng.normal(0.0, sigma, arr.shape)
        return Image.fromarray(np.round(np.clip(noisy, 0, 1) * 255).astype(np.uint8))
    if kind == "haze":
        alpha = min(0.12 * severity, 0.65)
        hazy = (1 - alpha) * arr + alpha
        return Image.fromarray(np.round(np.clip(hazy, 0, 1) * 255).astype(np.uint8))
    if kind == "cloud":
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cloud = np.zeros((h, w), dtype=np.float32)
        blobs = 2 + severity * 2
        for _ in range(blobs):
            cx = rng.uniform(0, w)
            cy = rng.uniform(0, h)
            sx = rng.uniform(0.08, 0.25) * w
            sy = rng.uniform(0.08, 0.25) * h
            cloud += np.exp(-(((xx - cx) / max(sx, 1)) ** 2 + ((yy - cy) / max(sy, 1)) ** 2) / 2)
        cloud = np.clip(cloud * (0.18 + 0.10 * severity), 0, 0.92)[..., None]
        cloudy = arr * (1 - cloud) + cloud
        return Image.fromarray(np.round(np.clip(cloudy, 0, 1) * 255).astype(np.uint8))
    if kind == "jpeg":
        from io import BytesIO

        quality = max(8, 95 - severity * 16)
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if kind == "resolution_loss":
        factor = 1 + severity
        small = rgb.resize((max(1, rgb.width // factor), max(1, rgb.height // factor)))
        return small.resize(rgb.size)
    return apply_corruption(rgb, kind, severity, rng)

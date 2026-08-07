from __future__ import annotations

import numpy as np
from PIL import Image

from qfuse.attacks.cross_modal import compose_sensor_panel


def percentile_rgb(array: np.ndarray, bands: tuple[int, int, int] = (3, 2, 1)) -> Image.Image:
    """Create an 8-bit RGB composite from a CHW or HWC multispectral array."""
    x = np.asarray(array, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("array must be three-dimensional")
    if x.shape[0] < x.shape[-1]:
        x = np.moveaxis(x, 0, -1)
    rgb = x[..., list(bands)]
    lo = np.nanpercentile(rgb, 2, axis=(0, 1), keepdims=True)
    hi = np.nanpercentile(rgb, 98, axis=(0, 1), keepdims=True)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return Image.fromarray(np.round(rgb * 255).astype(np.uint8))


def sar_pseudocolor(vv: np.ndarray, vh: np.ndarray) -> Image.Image:
    vv = np.asarray(vv, dtype=np.float32)
    vh = np.asarray(vh, dtype=np.float32)
    ratio = vv - vh
    stack = np.stack([vv, vh, ratio], axis=-1)
    lo = np.nanpercentile(stack, 2, axis=(0, 1), keepdims=True)
    hi = np.nanpercentile(stack, 98, axis=(0, 1), keepdims=True)
    stack = np.clip((stack - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return Image.fromarray(np.round(stack * 255).astype(np.uint8))


def make_sen12ms_panel(optical: Image.Image, sar: Image.Image) -> Image.Image:
    return compose_sensor_panel([optical, sar], ["Sentinel-2 optical", "Sentinel-1 SAR"])


# Clear alias used by the dataset-ready package.
build_optical_sar_panel = make_sen12ms_panel

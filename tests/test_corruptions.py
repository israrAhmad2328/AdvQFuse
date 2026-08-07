import numpy as np
from PIL import Image

from qfuse.data import apply_corruption


def test_corruptions_preserve_image_size() -> None:
    image = Image.fromarray(np.full((64, 80, 3), 128, dtype=np.uint8))
    for kind in ["gaussian_noise", "blur", "low_light", "contrast", "occlusion", "grayscale"]:
        out = apply_corruption(image, kind, severity=3, rng=np.random.default_rng(0))
        assert out.size == image.size

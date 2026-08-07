import numpy as np
from PIL import Image

from qfuse.attacks import apply_patch, generate_question_attacks, spsa_patch_attack
from qfuse.data import apply_remote_sensing_corruption


def test_patch_and_text_attacks():
    image = Image.new("RGB", (32, 32), "white")
    patch = np.zeros((8, 8, 3), dtype=np.float32)
    attacked = apply_patch(image, patch, 4, 4)
    assert attacked.size == image.size
    variants = generate_question_attacks("Is the vehicle left of the road?", "yes")
    assert "spatial_swap" in variants
    assert "right" in variants["spatial_swap"].lower()


def test_spsa_runs():
    initial = np.zeros((3, 3, 3), dtype=np.float32)
    result = spsa_patch_attack(
        initial,
        score_fn=lambda p: float(p.mean()),
        iterations=3,
        samples_per_step=2,
        seed=1,
    )
    assert result.queries > 1
    assert result.patch.shape == initial.shape


def test_remote_sensing_corruptions():
    image = Image.new("RGB", (48, 48), "gray")
    for kind in ["speckle", "haze", "cloud", "jpeg", "resolution_loss"]:
        out = apply_remote_sensing_corruption(image, kind, 2, np.random.default_rng(0))
        assert out.size == image.size

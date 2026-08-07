from __future__ import annotations

import numpy as np

from qfuse.evaluation.qualitative_visualizations import (
    apply_attack,
    make_scene,
    recover_observation,
    to_sar,
    uncertainty_map,
)


def test_scene_generation_is_deterministic_and_well_formed():
    a = make_scene("uav", 13, size=192)
    b = make_scene("uav", 13, size=192)
    assert a.image.size == (192, 192)
    assert np.array_equal(np.asarray(a.image), np.asarray(b.image))
    assert len(a.object_boxes) > 0
    assert a.water_mask.shape == (192, 192)
    assert a.road_mask.shape == (192, 192)


def test_attacks_and_recovery_preserve_image_size():
    scene = make_scene("flood", 19, size=192)
    for attack in ["patch", "cloud", "haze", "misregistration", "instruction", "speckle"]:
        attacked, mask = apply_attack(scene, attack, severity=3, seed=21)
        recovered = recover_observation(attacked, attack)
        assert attacked.size == scene.image.size
        assert recovered.size == scene.image.size
        assert mask.shape == (192, 192)
        assert float(mask.max()) > 0


def test_sar_and_uncertainty_outputs():
    scene = make_scene("coastal", 23, size=192)
    sar = to_sar(scene, 29)
    _, attack_mask = apply_attack(scene, "misregistration", severity=4, seed=31)
    u = uncertainty_map(scene, attack_mask, seed=37)
    assert sar.size == scene.image.size
    assert u.shape == (192, 192)
    assert 0.0 <= float(u.min()) <= float(u.max()) <= 1.0

import numpy as np

from qfuse.fusion import fuse_modalities


def test_fusion_normalized_and_confident_when_modalities_agree() -> None:
    p, u, rel = fuse_modalities(
        [np.array([0.9, 0.1]), np.array([0.8, 0.2])],
        [0.9, 0.8],
    )
    assert np.isclose(p.sum(), 1.0)
    assert p[0] > p[1]
    assert 0 <= u <= 1
    assert np.all((rel >= 0) & (rel <= 1))


def test_conflict_downweights_modalities() -> None:
    _, _, rel_agree = fuse_modalities(
        [np.array([0.9, 0.1]), np.array([0.85, 0.15])], [1.0, 1.0]
    )
    _, _, rel_conflict = fuse_modalities(
        [np.array([0.9, 0.1]), np.array([0.1, 0.9])], [1.0, 1.0]
    )
    assert rel_conflict.mean() < rel_agree.mean()

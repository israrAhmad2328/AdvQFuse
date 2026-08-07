import numpy as np

from qfuse.precision_path import (
    PrecisionPoint,
    certify_observed_path_invariance,
    extract_precision_path_features,
)


def test_stable_path_has_no_rank_flips_and_certificate():
    points = [
        PrecisionPoint(2, logits=np.array([3.00, 1.00, 0.1])),
        PrecisionPoint(4, logits=np.array([3.05, 0.98, 0.1])),
        PrecisionPoint(8, logits=np.array([3.08, 0.96, 0.1])),
    ]
    features = extract_precision_path_features(points)
    certificate = certify_observed_path_invariance(points)
    assert features.rank_flip_count == 0
    assert features.js_total_variation >= 0
    assert certificate.certified
    assert certificate.slack > 0


def test_unstable_path_detects_rank_flip_and_fails_certificate():
    points = [
        PrecisionPoint(2, logits=np.array([1.0, 3.0, 0.1])),
        PrecisionPoint(4, logits=np.array([2.1, 2.0, 0.1])),
        PrecisionPoint(8, logits=np.array([3.0, 1.0, 0.1])),
    ]
    features = extract_precision_path_features(points)
    certificate = certify_observed_path_invariance(points)
    assert features.rank_flip_count >= 1
    assert features.reference_disagreement_count >= 1
    assert not certificate.certified

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from qfuse.precision_path import PrecisionPoint, certify_observed_path_invariance, extract_precision_path_features


def main() -> None:
    points = [
        PrecisionPoint(2, logits=np.array([2.10, 1.65, 0.30]), latency_ms=18, memory_mb=2100),
        PrecisionPoint(4, logits=np.array([2.22, 1.58, 0.25]), latency_ms=27, memory_mb=3600),
        PrecisionPoint(8, logits=np.array([2.30, 1.51, 0.20]), latency_ms=38, memory_mb=6100),
        PrecisionPoint(16, logits=np.array([2.34, 1.48, 0.18]), latency_ms=61, memory_mb=11000),
    ]
    payload = {
        "features": extract_precision_path_features(points).to_dict(),
        "certificate": asdict(certify_observed_path_invariance(points)),
        "warning": "Demonstration values only; not paper results.",
    }
    out = Path("results/tpami_demo/precision_path_demo.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

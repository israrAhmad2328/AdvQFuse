from __future__ import annotations

import argparse
import json
from pathlib import Path

from qfuse.evaluation.tpami_visualizations import generate_tpami_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TPAMI-specific synthetic figure layouts.")
    parser.add_argument("--output", default="figures/tpami_demo")
    args = parser.parse_args()
    output = Path(args.output)
    paths = generate_tpami_suite(output)
    manifest = {
        "status": "synthetic_layout_only",
        "warning": "Do not report these values as experimental evidence.",
        "figure_count": len(paths) // 2,
        "files": [str(path) for path in paths],
    }
    manifest_path = Path("results/tpami_demo/manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

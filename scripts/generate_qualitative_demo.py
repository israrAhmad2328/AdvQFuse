from __future__ import annotations

import argparse
from pathlib import Path

from qfuse.evaluation import generate_qualitative_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic qualitative remote-sensing case figures.")
    parser.add_argument("--output-dir", default="figures/qualitative_demo")
    parser.add_argument("--metadata", default="results/qualitative_demo/case_metadata.csv")
    args = parser.parse_args()
    paths = generate_qualitative_figures(Path(args.output_dir), Path(args.metadata))
    print(f"Generated {len(paths)} qualitative PNG figures.")
    print("All panels are synthetic layout demonstrations and must be replaced by real test samples.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qfuse.evaluation import generate_all_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the full AdvQFuse-RS figure suite from a result CSV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="figures/real_results")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    paths = generate_all_figures(df, Path(args.output_dir))
    print(f"Generated {len(paths)} PNG figures plus PDF copies.")


if __name__ == "__main__":
    main()

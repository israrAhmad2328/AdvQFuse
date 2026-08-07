from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qfuse.evaluation import generate_all_figures, generate_extended_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate base and extended quantitative AdvQFuse-RS result figures.")
    parser.add_argument("--input", required=True, help="Prediction-level CSV following the project logging schema.")
    parser.add_argument("--base-output-dir", default="figures/real_results/base")
    parser.add_argument("--extended-output-dir", default="figures/real_results/extended")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    base = generate_all_figures(df, Path(args.base_output_dir))
    extended = generate_extended_figures(df, Path(args.extended_output_dir))
    print(f"Generated {len(base)} base figures and {len(extended)} extended multi-panel figures.")


if __name__ == "__main__":
    main()

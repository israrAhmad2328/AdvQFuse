from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qfuse.evaluation import generate_all_figures, generate_extended_figures, generate_qualitative_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the complete 36-figure AdvQFuse-RS demonstration suite.")
    parser.add_argument("--input", default="results/advanced_demo/advrs_synthetic_predictions.csv")
    parser.add_argument("--base-output", default="figures/advanced_demo")
    parser.add_argument("--extended-output", default="figures/extended_quantitative")
    parser.add_argument("--qualitative-output", default="figures/qualitative_demo")
    parser.add_argument("--qualitative-metadata", default="results/qualitative_demo/case_metadata.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    base = generate_all_figures(df, Path(args.base_output))
    extended = generate_extended_figures(df, Path(args.extended_output))
    qualitative = generate_qualitative_figures(Path(args.qualitative_output), Path(args.qualitative_metadata))
    print(f"Generated {len(base) + len(extended) + len(qualitative)} PNG figures and matching vector PDF copies.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = [
    "sample_id",
    "dataset",
    "split",
    "task",
    "modality",
    "path",
    "label",
    "question",
    "corruption",
    "severity",
    "missing",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an empty QFuseBench manifest template.")
    parser.add_argument("--output", default="data/qfusebench_manifest.csv")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=COLUMNS).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

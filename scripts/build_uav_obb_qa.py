from __future__ import annotations

import argparse
from pathlib import Path

from qfuse.data import build_uav_obb_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UAV-OBB-QA from the official YOLOv8-OBB folder.")
    parser.add_argument("--dataset-root", required=True, help="Any parent folder containing data.yaml and train/valid/test.")
    parser.add_argument("--output", default="data/derived/manifests/uav_obb_qa.jsonl")
    parser.add_argument("--positive-existence-only", action="store_true")
    args = parser.parse_args()
    records = build_uav_obb_manifest(
        args.dataset_root,
        include_negative_existence=not args.positive_existence_only,
    )
    write_manifest(records, args.output)
    print(f"Wrote {len(records):,} QA samples to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from qfuse.data import (
    build_earthvqa_manifest,
    build_floodnet_manifest,
    build_rsvqa_manifest,
    build_uav_obb_manifest,
    load_dataset_config,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized JSONL manifests for all available VQA datasets.")
    parser.add_argument("--config", default="configs/datasets.local.yaml")
    parser.add_argument("--output-dir", default="data/derived/manifests")
    parser.add_argument("--include-rsvqa-lr", action="store_true")
    args = parser.parse_args()
    roots = load_dataset_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    builders = [
        ("earthvqa", lambda: build_earthvqa_manifest(roots["earthvqa"])),
        ("floodnet", lambda: build_floodnet_manifest(roots["floodnet"])),
        ("rsvqa_hr", lambda: build_rsvqa_manifest(roots["rsvqa_hr"], high_resolution=True)),
        ("uav_obb_qa", lambda: build_uav_obb_manifest(roots["uav_obb"])),
    ]
    if args.include_rsvqa_lr:
        builders.append(("rsvqa_lr", lambda: build_rsvqa_manifest(roots["rsvqa_lr"], high_resolution=False)))

    total = 0
    for name, builder in builders:
        try:
            records = builder()
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[SKIP] {name}: {exc}")
            continue
        target = output / f"{name}.jsonl"
        write_manifest(records, target)
        print(f"[OK]   {name}: {len(records):,} records -> {target}")
        total += len(records)
    print(f"Total normalized records: {total:,}")


if __name__ == "__main__":
    main()

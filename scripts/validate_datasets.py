from __future__ import annotations

import argparse
from pathlib import Path

from qfuse.data import validate_all, write_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate every configured AdvQFuse dataset.")
    parser.add_argument("--config", default="configs/datasets.local.yaml")
    parser.add_argument("--report", default="results/dataset_validation.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless every dataset is ready.")
    args = parser.parse_args()

    statuses = validate_all(args.config)
    write_validation_report(statuses, args.report)
    print("\nAdvQFuse dataset audit")
    print("=" * 88)
    for status in statuses:
        marker = {"ready": "OK", "partial": "WARN", "missing": "MISS"}[status.state]
        print(f"[{marker:4}] {status.name:10} {status.root}")
        print(f"       {status.message}")
        if status.counts:
            print("       " + ", ".join(f"{key}={value}" for key, value in status.counts.items()))
        for item in status.missing:
            print(f"       missing: {item}")
        for item in status.warnings:
            print(f"       warning: {item}")
    print(f"\nMachine-readable report: {Path(args.report).resolve()}")
    if args.strict and any(status.state != "ready" for status in statuses):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

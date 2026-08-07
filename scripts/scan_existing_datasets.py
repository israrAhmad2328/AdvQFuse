from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import yaml

from qfuse.data import locate_uav_obb_root
from qfuse.data.dataset_layout import IMAGE_EXTENSIONS


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    return next((p.resolve() for p in candidates if p.exists()), None)


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _find_earthvqa(root: Path) -> Path | None:
    candidates = [root / "EarthVQA", root / "data/raw/earthvqa", root / "AdvQFuse_TPAMI_v4/data/raw/earthvqa"]
    for candidate in candidates:
        if (candidate / "Train_QA.json").exists() or (candidate / "Train/images_png").exists():
            return candidate.resolve()
    return None


def _find_floodnet(root: Path) -> Path | None:
    candidates = [root / "FloodNet_VQA", root / "data/raw/floodnet", root / "AdvQFuse_TPAMI_v4/data/raw/floodnet"]
    for candidate in candidates:
        if (candidate / "train_image/img").exists() and (candidate / "train_image/ann").exists():
            return candidate.resolve()
    return _first_existing(candidates)


def _rsvqa_score(path: Path, prefix: str) -> int:
    if not path.exists():
        return -1
    ann = path / "annotations" if (path / "annotations").exists() else path
    img = path / "images" if (path / "images").exists() else path
    q = len(list(ann.rglob(f"{prefix}_split_*_questions.json")))
    a = len(list(ann.rglob(f"{prefix}_split_*_answers.json")))
    return 100 * min(q, a) + min(_count_images(img), 99)


def _find_rsvqa(root: Path, high_resolution: bool) -> Path | None:
    name = "rsvqa_hr" if high_resolution else "rsvqa_lr"
    prefix = "USGS" if high_resolution else "LR"
    candidates = [
        root / name,
        root / "data/raw" / name,
        root / "AdvQFuse_TPAMI_v4/data/raw" / name,
    ]
    # Loose LR JSONs may initially live directly under /content.
    if not high_resolution:
        candidates.append(root)
    scored = sorted((( _rsvqa_score(p, prefix), p) for p in candidates), key=lambda item: item[0], reverse=True)
    return scored[0][1].resolve() if scored and scored[0][0] >= 0 else None


def _find_uav(root: Path) -> Path | None:
    direct_candidates = [
        root / "uav-obb-urban-vehicle-detection-dataset",
        root / "UAV-OBB",
        root / "data/raw/uav_obb",
        root / "AdvQFuse_TPAMI_v4/data/raw/uav_obb",
    ]
    for direct in direct_candidates:
        if direct.exists():
            found = locate_uav_obb_root(direct)
            if found:
                return found.resolve()
    # Last resort for archive-added nesting.
    for yaml_path in root.rglob("data.yaml"):
        found = locate_uav_obb_root(yaml_path.parent)
        if found:
            return found.resolve()
    return None


def _find_sen12ms(root: Path) -> Path | None:
    candidates = [root / "sen12ms_qa", root / "sen12ms", root / "data/raw/sen12ms", root / "AdvQFuse_TPAMI_v4/data/raw/sen12ms"]
    for candidate in candidates:
        if ((candidate / "optical").exists() and (candidate / "sar").exists()) or list(candidate.glob("ROIs*_s1")):
            return candidate.resolve()
    return _first_existing(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a Colab/workstation folder and write dataset paths without moving large files.")
    parser.add_argument("--search-root", default=".")
    parser.add_argument("--output", default="configs/datasets.local.yaml")
    args = parser.parse_args()
    root = Path(args.search_root).expanduser().resolve()

    found: dict[str, Path | None] = {
        "earthvqa": _find_earthvqa(root),
        "floodnet": _find_floodnet(root),
        "rsvqa_hr": _find_rsvqa(root, True),
        "rsvqa_lr": _find_rsvqa(root, False),
        "uav_obb": _find_uav(root),
        "sen12ms": _find_sen12ms(root),
    }
    payload = {"datasets": {name: {"root": str(path or (root / f"data/raw/{name}"))} for name, path in found.items()}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    for name, path in found.items():
        print(f"{name:10}: {path if path else 'not found'}")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()

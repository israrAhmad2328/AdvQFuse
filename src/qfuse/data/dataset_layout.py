from __future__ import annotations

import json
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(slots=True)
class DatasetStatus:
    name: str
    root: str
    state: str
    message: str
    counts: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.state in {"ready", "partial"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_dataset_config(path: str | Path) -> dict[str, Path]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    datasets = payload.get("datasets", payload)
    if not isinstance(datasets, dict):
        raise ValueError("dataset config must contain a mapping named 'datasets'")
    result: dict[str, Path] = {}
    for name, value in datasets.items():
        if isinstance(value, dict):
            value = value.get("root")
        if value:
            result[str(name)] = Path(str(value)).expanduser().resolve()
    return result


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _count_files(path: Path, patterns: tuple[str, ...]) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and any(p.match(pattern) for pattern in patterns))


def _find_child(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    return None


def locate_uav_obb_root(root: Path) -> Path | None:
    """Locate the actual YOLO-style UAV-OBB root even when archives add nested folders."""
    candidates = [root]
    candidates.extend(p.parent for p in root.rglob("data.yaml")) if root.exists() else None
    for candidate in candidates:
        train = candidate / "train"
        valid = candidate / "valid"
        test = candidate / "test"
        if all((split / "images").exists() and (split / "labels").exists() for split in (train, valid, test)):
            return candidate
    return None


def validate_earthvqa(root: Path) -> DatasetStatus:
    required = [
        "Train/images_png",
        "Val/images_png",
        "Test/images_png",
        "Train_QA.json",
        "Val_QA.json",
        "Test_QA.json",
    ]
    missing = [item for item in required if not (root / item).exists()]
    counts = {
        "train_images": _count_images(root / "Train/images_png"),
        "val_images": _count_images(root / "Val/images_png"),
        "test_images": _count_images(root / "Test/images_png"),
        "qa_json": _count_files(root, ("*_QA.json",)),
    }
    if not root.exists():
        return DatasetStatus("earthvqa", str(root), "missing", "Dataset root does not exist.", counts, required)
    if missing:
        return DatasetStatus(
            "earthvqa",
            str(root),
            "partial",
            "EarthVQA is incomplete or extracted into an unexpected folder.",
            counts,
            missing,
            ["The official download is access-controlled; do not use placeholder wget URLs."],
        )
    return DatasetStatus("earthvqa", str(root), "ready", "Official EarthVQA layout detected.", counts)


def validate_floodnet(root: Path) -> DatasetStatus:
    split_map = {"train": "train_image", "val": "valid_image", "test": "test_image"}
    missing: list[str] = []
    counts: dict[str, int] = {}
    for split, folder in split_map.items():
        img = root / folder / "img"
        ann = root / folder / "ann"
        if not img.exists():
            missing.append(f"{folder}/img")
        if not ann.exists():
            missing.append(f"{folder}/ann")
        counts[f"{split}_images"] = _count_images(img)
        counts[f"{split}_annotation_files"] = _count_files(ann, ("*.json", "*.txt"))
    if not root.exists():
        return DatasetStatus("floodnet", str(root), "missing", "Dataset root does not exist.", counts, missing)
    state = "ready" if not missing and counts.get("train_images", 0) else "partial"
    message = "FloodNet VQA folder layout detected." if state == "ready" else "FloodNet files are incomplete."
    return DatasetStatus("floodnet", str(root), state, message, counts, missing)


def _rsvqa_prefix(dataset: str) -> str:
    return "USGS" if dataset == "rsvqa_hr" else "LR"


def validate_rsvqa(root: Path, dataset: str) -> DatasetStatus:
    prefix = _rsvqa_prefix(dataset)
    annotation_root = _find_child(root, ("annotations", "text")) or root
    images_root = _find_child(root, ("images", "Images")) or root
    questions = list(annotation_root.rglob(f"{prefix}_split_*_questions.json")) if annotation_root.exists() else []
    answers = list(annotation_root.rglob(f"{prefix}_split_*_answers.json")) if annotation_root.exists() else []
    images = _count_images(images_root)
    archives = list(root.rglob("*.tar")) + list(root.rglob("*.tar.gz")) if root.exists() else []
    missing: list[str] = []
    warnings: list[str] = []
    if not questions:
        missing.append(f"{prefix}_split_*_questions.json")
    if not answers:
        missing.append(f"{prefix}_split_*_answers.json")
    if images == 0:
        missing.append("extracted image files")
        if archives:
            warnings.append("An image archive is present but has not been extracted.")
    if not root.exists():
        state = "missing"
    elif questions and answers and images:
        state = "ready"
    else:
        state = "partial"
    return DatasetStatus(
        dataset,
        str(root),
        state,
        f"{dataset.upper()} {'is ready' if state == 'ready' else 'needs organization or extraction'}.",
        {"question_files": len(questions), "answer_files": len(answers), "images": images, "archives": len(archives)},
        missing,
        warnings,
    )


def validate_uav_obb(root: Path) -> DatasetStatus:
    actual = locate_uav_obb_root(root)
    if actual is None:
        return DatasetStatus(
            "uav_obb",
            str(root),
            "missing" if not root.exists() else "partial",
            "Could not find a YOLO-style train/valid/test images+labels layout.",
            {},
            ["data.yaml", "train/images", "train/labels", "valid/images", "valid/labels", "test/images", "test/labels"],
        )
    counts: dict[str, int] = {}
    warnings: list[str] = []
    for split in ("train", "valid", "test"):
        counts[f"{split}_images"] = _count_images(actual / split / "images")
        counts[f"{split}_labels"] = _count_files(actual / split / "labels", ("*.txt",))
        if counts[f"{split}_images"] != counts[f"{split}_labels"]:
            warnings.append(f"{split}: image/label counts differ; empty-label images may be valid but should be audited.")
    return DatasetStatus("uav_obb", str(actual), "ready", "YOLOv8-OBB layout detected.", counts, warnings=warnings)


def validate_sen12ms(root: Path) -> DatasetStatus:
    optical = _find_child(root, ("optical", "s2", "S2"))
    sar = _find_child(root, ("sar", "s1", "S1"))
    landcover = _find_child(root, ("landcover", "lc", "labels"))
    # Official full release uses deeply nested ROIs*_s1/s2/lc folders.
    official_s1 = list(root.glob("ROIs*_s1")) if root.exists() else []
    official_s2 = list(root.glob("ROIs*_s2")) if root.exists() else []
    official_lc = list(root.glob("ROIs*_lc")) if root.exists() else []
    counts = {
        "optical_files": _count_images(optical) if optical else sum(_count_images(p) for p in official_s2),
        "sar_files": _count_images(sar) if sar else sum(_count_images(p) for p in official_s1),
        "landcover_files": _count_images(landcover) if landcover else sum(_count_images(p) for p in official_lc),
    }
    missing: list[str] = []
    warnings: list[str] = []
    if counts["optical_files"] == 0:
        missing.append("Sentinel-2/optical files")
    if counts["sar_files"] == 0:
        missing.append("Sentinel-1/SAR files")
    if counts["landcover_files"] == 0:
        missing.append("land-cover labels or labels.csv")
        warnings.append("Optical and SAR images alone are not an evaluable QA benchmark because no ground-truth answer is available.")
    if not root.exists():
        state = "missing"
    elif not missing:
        state = "ready"
    else:
        state = "partial"
    return DatasetStatus("sen12ms", str(root), state, "SEN12MS multi-sensor layout audit complete.", counts, missing, warnings)


def validate_all(config_path: str | Path) -> list[DatasetStatus]:
    roots = load_dataset_config(config_path)
    validators = {
        "earthvqa": validate_earthvqa,
        "floodnet": validate_floodnet,
        "rsvqa_hr": lambda path: validate_rsvqa(path, "rsvqa_hr"),
        "rsvqa_lr": lambda path: validate_rsvqa(path, "rsvqa_lr"),
        "uav_obb": validate_uav_obb,
        "sen12ms": validate_sen12ms,
    }
    statuses: list[DatasetStatus] = []
    for name, validator in validators.items():
        root = roots.get(name, Path(f"data/raw/{name}").resolve())
        statuses.append(validator(root))
    return statuses


def write_validation_report(statuses: list[DatasetStatus], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([status.to_dict() for status in statuses], indent=2), encoding="utf-8")

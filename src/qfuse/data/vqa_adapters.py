from __future__ import annotations

import ast
import json
import re
import tarfile
from pathlib import Path
from typing import Any, Iterable

from .advrs_manifest import AdvRSSample
from .dataset_layout import IMAGE_EXTENSIONS, locate_uav_obb_root
from .uav_obb_qa import generate_qa_for_image, load_class_mapping


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_dicts(nested)
    elif isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return
        yield from _iter_dicts(parsed)


def _case_get(record: dict[str, Any], *keys: str) -> Any:
    lower = {str(k).lower(): v for k, v in record.items()}
    for key in keys:
        if key.lower() in lower:
            return lower[key.lower()]
    return None


def _resolve_image(images_root: Path, image_hint: Any, fallback_stem: str | None = None) -> Path | None:
    hints: list[str] = []
    if image_hint is not None:
        hints.append(str(image_hint))
    if fallback_stem:
        hints.append(str(fallback_stem))
    for hint in hints:
        direct = images_root / hint
        if direct.is_file():
            return direct
        stem = Path(hint).stem
        for ext in IMAGE_EXTENSIONS:
            direct = images_root / f"{stem}{ext}"
            if direct.is_file():
                return direct
        matches = [p for p in images_root.rglob("*") if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTENSIONS]
        if matches:
            return matches[0]
    return None


def build_earthvqa_manifest(root: str | Path) -> list[AdvRSSample]:
    root = Path(root)
    records: list[AdvRSSample] = []
    mapping = [("Train", "train"), ("Val", "val"), ("Test", "test")]
    for folder, split in mapping:
        qa_path = root / f"{folder}_QA.json"
        images_root = root / folder / "images_png"
        if not qa_path.exists():
            continue
        payload = _load_json(qa_path)
        if not isinstance(payload, dict):
            continue
        for image_name, questions in payload.items():
            image_path = _resolve_image(images_root, image_name)
            if image_path is None:
                continue
            for idx, item in enumerate(questions if isinstance(questions, list) else []):
                question = _case_get(item, "Question", "question")
                answer = _case_get(item, "Answer", "answer")
                qtype = _case_get(item, "Type", "question_type") or "unknown"
                if question is None or answer is None:
                    continue
                records.append(
                    AdvRSSample(
                        sample_id=f"earthvqa:{split}:{Path(image_name).stem}:{idx}",
                        dataset="EarthVQA",
                        split=split,
                        image_paths=[str(image_path)],
                        question=str(question),
                        answer=str(answer),
                        question_type=str(qtype),
                        sensor_labels=["RGB remote-sensing image"],
                    )
                )
    return records


def _parse_maybe_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _floodnet_qa_pairs(payload: Any) -> list[dict[str, Any]]:
    """Normalize both compact and DatasetNinja-style FloodNet annotations.

    DatasetNinja exports often store question and answer in separate tags, each
    serialized as a Python-dict string. We join them by Question_ID instead of
    assuming that question and answer appear in the same nested object.
    """
    paired: dict[str, dict[str, Any]] = {}
    unkeyed: list[dict[str, Any]] = []

    def absorb(item: dict[str, Any]) -> None:
        qid = _case_get(item, "Question_ID", "question_id", "id")
        key = str(qid) if qid is not None else ""
        target = paired.setdefault(key, {}) if key else {}
        target.update(item)
        if not key:
            unkeyed.append(target)

    if isinstance(payload, dict) and isinstance(payload.get("tags"), list):
        for tag in payload["tags"]:
            if not isinstance(tag, dict):
                continue
            parsed = _parse_maybe_dict(tag.get("value"))
            if parsed:
                absorb(parsed)

    for item in _iter_dicts(payload):
        # Avoid absorbing wrapper/tag records that do not contain QA fields.
        if any(_case_get(item, key) is not None for key in ("Question", "question", "Ground_Truth", "answer", "Answer")):
            absorb(item)

    candidates = list(paired.values()) + unkeyed
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in candidates:
        question = _case_get(item, "Question", "question")
        answer = _case_get(item, "Ground_Truth", "ground_truth", "answer", "Answer")
        qid = _case_get(item, "Question_ID", "question_id", "id")
        if question is None or answer is None:
            continue
        signature = (str(qid), str(question), str(answer))
        if signature in seen:
            continue
        seen.add(signature)
        output.append(item)
    return output


def build_floodnet_manifest(root: str | Path) -> list[AdvRSSample]:
    root = Path(root)
    records: list[AdvRSSample] = []
    mapping = [("train_image", "train"), ("valid_image", "val"), ("test_image", "test")]
    for folder, split in mapping:
        images_root = root / folder / "img"
        ann_root = root / folder / "ann"
        if not images_root.exists() or not ann_root.exists():
            continue
        ann_files = list(ann_root.rglob("*.json"))
        for ann_path in ann_files:
            try:
                payload = _load_json(ann_path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            image_hint = _case_get(payload, "image", "image_name", "name", "file_name") if isinstance(payload, dict) else None
            # For files such as 6562.JPG.json, Path.stem is 6562.JPG, which
            # resolves directly to the corresponding image.
            image_path = _resolve_image(images_root, image_hint, ann_path.stem)
            local_index = 0
            for item in _floodnet_qa_pairs(payload):
                question = _case_get(item, "Question", "question")
                answer = _case_get(item, "Ground_Truth", "ground_truth", "answer", "Answer")
                qid = _case_get(item, "Question_ID", "question_id", "id")
                qtype = _case_get(item, "Question_Type", "question_type", "type") or "unknown"
                item_image = _case_get(item, "image", "image_name", "file_name")
                resolved = _resolve_image(images_root, item_image) or image_path
                if question is None or answer is None or resolved is None:
                    continue
                sid = qid if qid is not None else f"{ann_path.stem}:{local_index}"
                records.append(
                    AdvRSSample(
                        sample_id=f"floodnet:{split}:{sid}",
                        dataset="FloodNet",
                        split=split,
                        image_paths=[str(resolved)],
                        question=str(question),
                        answer=str(answer),
                        question_type=str(qtype),
                        sensor_labels=["UAV RGB image"],
                        metadata={"annotation_file": str(ann_path)},
                    )
                )
                local_index += 1
    return records


def _records_from_question_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("questions", "question", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        # Some RSVQA files are dicts keyed by question id.
        if all(isinstance(v, dict) for v in payload.values()):
            return [dict(v, _outer_id=k) for k, v in payload.items()]
    return []


def _answer_map(payload: Any) -> dict[str, str]:
    records = _records_from_question_payload(payload)
    if not records and isinstance(payload, dict):
        for key in ("answers", "annotations"):
            if isinstance(payload.get(key), list):
                records = [item for item in payload[key] if isinstance(item, dict)]
                break
    result: dict[str, str] = {}
    for item in records:
        qid = _case_get(item, "question_id", "id", "_outer_id")
        answer = _case_get(item, "answer", "multiple_choice_answer", "ground_truth")
        if qid is not None and answer is not None:
            result[str(qid)] = str(answer)
    return result


def build_rsvqa_manifest(root: str | Path, high_resolution: bool = True) -> list[AdvRSSample]:
    root = Path(root)
    prefix = "USGS" if high_resolution else "LR"
    dataset_name = "RSVQA-HR" if high_resolution else "RSVQA-LR"
    annotation_root = root / "annotations" if (root / "annotations").exists() else root
    images_root = root / "images" if (root / "images").exists() else root
    records: list[AdvRSSample] = []
    split_alias = {"train": "train", "val": "val", "test": "test", "test_phili": "test"}
    for question_path in sorted(annotation_root.rglob(f"{prefix}_split_*_questions.json")):
        match = re.search(r"_split_(.+?)_questions", question_path.name)
        raw_split = match.group(1) if match else "test"
        split = split_alias.get(raw_split, "test")
        answer_path = question_path.with_name(question_path.name.replace("_questions.json", "_answers.json"))
        answers = _answer_map(_load_json(answer_path)) if answer_path.exists() else {}
        q_records = _records_from_question_payload(_load_json(question_path))
        for idx, item in enumerate(q_records):
            qid = _case_get(item, "question_id", "id", "_outer_id")
            image_id = _case_get(item, "img_id", "image_id", "image", "image_name")
            question = _case_get(item, "question", "Question")
            qtype = _case_get(item, "type", "question_type", "category") or "unknown"
            answer = answers.get(str(qid)) if qid is not None else _case_get(item, "answer")
            image_path = _resolve_image(images_root, image_id)
            if question is None or answer is None or image_path is None:
                continue
            records.append(
                AdvRSSample(
                    sample_id=f"{dataset_name.lower()}:{raw_split}:{qid if qid is not None else idx}",
                    dataset=dataset_name,
                    split=split,
                    image_paths=[str(image_path)],
                    question=str(question),
                    answer=str(answer),
                    question_type=str(qtype),
                    sensor_labels=["RGB remote-sensing image"],
                    metadata={"official_split": raw_split},
                )
            )
    return records


def build_uav_obb_manifest(root: str | Path, include_negative_existence: bool = True) -> list[AdvRSSample]:
    actual = locate_uav_obb_root(Path(root))
    if actual is None:
        raise FileNotFoundError("Could not locate the UAV-OBB YOLO-style root")
    class_mapping = load_class_mapping(actual / "data.yaml")
    records: list[AdvRSSample] = []
    split_map = {"train": "train", "valid": "val", "test": "test"}
    for source_split, manifest_split in split_map.items():
        images_dir = actual / source_split / "images"
        labels_dir = actual / source_split / "labels"
        for image_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            records.extend(
                generate_qa_for_image(
                    image_path,
                    label_path,
                    split=manifest_split,
                    class_mapping=class_mapping,
                    include_negative_existence=include_negative_existence,
                )
            )
    return records

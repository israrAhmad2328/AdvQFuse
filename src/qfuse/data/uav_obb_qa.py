from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from .advrs_manifest import AdvRSSample

DEFAULT_CLASS_MAPPING: dict[int, str] = {
    0: "bike",
    1: "bus",
    2: "car",
    3: "other_vehicle",
    4: "taxi",
    5: "truck",
}


@dataclass(frozen=True)
class OBBObject:
    points: np.ndarray
    class_name: str
    difficult: int = 0

    @property
    def center(self) -> tuple[float, float]:
        return float(self.points[:, 0].mean()), float(self.points[:, 1].mean())

    @property
    def angle_deg(self) -> float:
        edge = self.points[1] - self.points[0]
        return math.degrees(math.atan2(float(edge[1]), float(edge[0]))) % 180.0


def load_class_mapping(path: str | Path) -> dict[int, str]:
    path = Path(path)
    if not path.exists():
        return dict(DEFAULT_CLASS_MAPPING)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = payload.get("names", DEFAULT_CLASS_MAPPING)
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    return dict(DEFAULT_CLASS_MAPPING)


def _center_angle_to_points(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    # Ultralytics angles are normally radians. Values outside a plausible radian
    # range are interpreted as degrees for compatibility with exported datasets.
    theta = angle if abs(angle) <= 2 * math.pi + 1e-6 else math.radians(angle)
    corners = np.asarray(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]],
        dtype=float,
    )
    rot = np.asarray([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return corners @ rot.T + np.asarray([cx, cy])


def parse_obb_label(
    path: str | Path,
    class_mapping: Mapping[int, str] | None = None,
) -> list[OBBObject]:
    """Parse UAV-OBB labels in YOLO-corner, YOLO-center-angle, or DOTA format.

    Supported lines:
      * YOLO OBB corners: ``class x1 y1 x2 y2 x3 y3 x4 y4``
      * YOLO OBB center: ``class cx cy w h angle``
      * DOTA: ``x1 y1 ... x4 y4 class [difficult]``
    Coordinates may be normalized or pixel-valued; counting/orientation generation
    does not require denormalization.
    """
    mapping = dict(class_mapping or DEFAULT_CLASS_MAPPING)
    objects: list[OBBObject] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        parts = line.strip().split()
        if not parts or parts[0].lower().startswith(("imagesource", "gsd")):
            continue
        try:
            # YOLO formats begin with a class id.
            class_id = int(float(parts[0]))
            if class_id not in mapping:
                raise ValueError("not a known YOLO class id")
            if len(parts) >= 9:
                coords = np.asarray([float(v) for v in parts[1:9]], dtype=float).reshape(4, 2)
            elif len(parts) >= 6:
                cx, cy, w, h, angle = (float(v) for v in parts[1:6])
                coords = _center_angle_to_points(cx, cy, w, h, angle)
            else:
                continue
            objects.append(OBBObject(coords, mapping[class_id], 0))
            continue
        except ValueError:
            pass

        # DOTA format begins with eight coordinates, followed by the class name.
        if len(parts) < 9:
            continue
        try:
            coords = np.asarray([float(v) for v in parts[:8]], dtype=float).reshape(4, 2)
        except ValueError:
            continue
        class_name = parts[8]
        difficult = int(parts[9]) if len(parts) > 9 and parts[9].isdigit() else 0
        objects.append(OBBObject(coords, class_name, difficult))
    return objects


# Backward-compatible alias.
parse_dota_label = parse_obb_label


def _orientation_bin(angle: float) -> str:
    if angle < 22.5 or angle >= 157.5:
        return "approximately horizontal"
    if angle < 67.5:
        return "diagonal rising"
    if angle < 112.5:
        return "approximately vertical"
    return "diagonal falling"


def _plural(name: str) -> str:
    readable = name.replace("_", " ")
    return readable if readable.endswith("s") else f"{readable}s"


def generate_qa_for_image(
    image_path: str | Path,
    label_path: str | Path,
    dataset: str = "UAV-OBB-QA",
    split: str = "train",
    class_mapping: Mapping[int, str] | None = None,
    include_negative_existence: bool = True,
) -> list[AdvRSSample]:
    """Convert an OBB annotation into deterministic and auditable VQA records."""
    image_path = Path(image_path)
    mapping = dict(class_mapping or DEFAULT_CLASS_MAPPING)
    objects = [o for o in parse_obb_label(label_path, mapping) if o.difficult == 0]
    all_classes = [mapping[idx] for idx in sorted(mapping)]
    present_classes = sorted({o.class_name for o in objects})
    stem = image_path.stem
    records: list[AdvRSSample] = []

    def add(suffix: str, question: str, answer: str | int, qtype: str, **metadata: object) -> None:
        records.append(
            AdvRSSample(
                sample_id=f"{dataset.lower()}:{split}:{stem}:{suffix}",
                dataset=dataset,
                split=split,
                image_paths=[str(image_path)],
                question=question,
                answer=str(answer),
                question_type=qtype,
                sensor_labels=["RGB aerial image"],
                metadata={"source_label": str(label_path), **metadata},
            )
        )

    add("total_count", "How many annotated vehicles are visible?", len(objects), "count")
    for cls in all_classes:
        count = sum(o.class_name == cls for o in objects)
        add(f"count_{cls}", f"How many {_plural(cls)} are visible?", count, "class_count", class_name=cls)
        if include_negative_existence or count > 0:
            add(
                f"exist_{cls}",
                f"Is any {cls.replace('_', ' ')} visible?",
                "yes" if count else "no",
                "existence",
                class_name=cls,
            )

    if objects:
        dominant = max(present_classes, key=lambda c: sum(o.class_name == c for o in objects))
        add("dominant", "Which vehicle class is most frequent?", dominant.replace("_", " "), "dominant_class")
        angles = [o.angle_deg for o in objects]
        # Circular mean for 180-degree OBB orientation.
        radians = np.deg2rad(np.asarray(angles) * 2.0)
        mean_angle = float((math.degrees(math.atan2(np.sin(radians).mean(), np.cos(radians).mean())) / 2.0) % 180.0)
        add(
            "orientation",
            "What is the dominant orientation of the annotated vehicles?",
            _orientation_bin(mean_angle),
            "orientation",
            mean_angle_deg=mean_angle,
        )
    return records

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AdvRSSample:
    sample_id: str
    dataset: str
    split: str
    image_paths: list[str]
    question: str
    answer: str
    question_type: str
    sensor_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.sample_id or not self.dataset:
            raise ValueError("sample_id and dataset are required")
        if self.split not in {"train", "cal", "val", "test"}:
            raise ValueError("split must be train, cal, val, or test")
        if not self.image_paths:
            raise ValueError("at least one image path is required")
        if self.sensor_labels and len(self.sensor_labels) != len(self.image_paths):
            raise ValueError("sensor_labels must match image_paths")


def write_manifest(samples: list[AdvRSSample], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for sample in samples:
            sample.validate()
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def read_manifest(path: str | Path) -> list[AdvRSSample]:
    samples: list[AdvRSSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                sample = AdvRSSample(**json.loads(line))
                sample.validate()
                samples.append(sample)
            except Exception as exc:
                raise ValueError(f"invalid manifest line {line_number}: {exc}") from exc
    return samples

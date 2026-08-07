from __future__ import annotations

import base64
import json
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

from qfuse.math_utils import normalize
from qfuse.types import ModelPrediction


@dataclass(slots=True)
class BonsaiEndpoint:
    url: str
    model: str = "local-model"
    timeout_seconds: int = 180


def _data_url(path: str | Path) -> str:
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError(f"model did not return valid JSON: {text[:300]}")


class OpenAICompatibleBonsaiClient:
    """Thin client for the llama.cpp OpenAI-compatible Bonsai server."""

    def __init__(self, endpoint: BonsaiEndpoint) -> None:
        self.endpoint = endpoint

    def classify(
        self,
        prompt: str,
        image_paths: list[str | Path],
        class_names: list[str],
        quality: float = 1.0,
        max_tokens: int = 256,
    ) -> ModelPrediction:
        content: list[dict[str, Any]] = [{"type": "text", "text": self._build_prompt(prompt, class_names)}]
        content.extend({"type": "image_url", "image_url": {"url": _data_url(path)}} for path in image_paths)
        body = {
            "model": self.endpoint.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
        }
        start = time.perf_counter()
        response = requests.post(self.endpoint.url, json=body, timeout=self.endpoint.timeout_seconds)
        response.raise_for_status()
        latency_ms = 1000.0 * (time.perf_counter() - start)
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
        parsed = _extract_json(text)
        probs = self._probabilities_from_json(parsed, class_names)
        return ModelPrediction(
            probabilities=probs,
            quality=quality,
            latency_ms=latency_ms,
            metadata={"raw": parsed, "timings": payload.get("timings", {})},
        )

    @staticmethod
    def _build_prompt(prompt: str, class_names: list[str]) -> str:
        names = ", ".join(class_names)
        return (
            f"{prompt}\n"
            f"Allowed labels: [{names}]. Return only JSON with keys 'label', 'confidence', "
            "and optional 'class_probabilities'. Confidence must be in [0,1]. "
            "If class_probabilities is supplied, include every allowed label and make values sum to 1."
        )

    @staticmethod
    def _probabilities_from_json(parsed: dict[str, Any], class_names: list[str]) -> np.ndarray:
        mapping = parsed.get("class_probabilities")
        if isinstance(mapping, dict):
            p = np.array([float(mapping.get(name, 0.0)) for name in class_names], dtype=float)
            if p.sum() > 0:
                return normalize(p)
        label = str(parsed.get("label", ""))
        confidence = float(parsed.get("confidence", 0.5))
        confidence = float(np.clip(confidence, 1.0 / len(class_names), 1.0))
        p = np.full(len(class_names), (1.0 - confidence) / max(len(class_names) - 1, 1))
        if label not in class_names:
            raise ValueError(f"unknown label returned by model: {label!r}")
        p[class_names.index(label)] = confidence
        return normalize(p)

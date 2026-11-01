"""Versioned settings for a separately trained, frozen camera detector."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..settings import _bool_env

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def runtime_version():
    """Return the installed ONNX Runtime version, or an explicit unavailable marker."""
    try:
        return version("onnxruntime")
    except PackageNotFoundError:
        return "unavailable"


@dataclass(frozen=True)
class DetectorConfig:
    model: Path
    model_sha256: str
    num_classes: int
    class_ids: tuple[int, ...]
    size: int
    confidence: float
    iou: float
    margin: float
    selection: str
    no_detection: str
    cache: bool

    def metadata(self):
        return {
            "enabled": True,
            "schema_version": 1,
            "format": "yolov8-raw-onnx-fp32",
            "model_sha256": self.model_sha256,
            "num_classes": self.num_classes,
            "class_ids": list(self.class_ids),
            "input_size": self.size,
            "confidence": self.confidence,
            "iou": self.iou,
            "margin": self.margin,
            "selection": self.selection,
            "no_detection": self.no_detection,
            "cache": self.cache,
            "provider": "CPUExecutionProvider",
            "runtime_version": runtime_version(),
        }

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.metadata(), sort_keys=True).encode()).hexdigest()


def _number(document, key, default, low, high):
    """Read a finite numeric detector setting and enforce its configured interval."""
    value = document.get(key, default)
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"detector {key} must be between {low} and {high}")
    return float(value)


@lru_cache(maxsize=4)
def load_config(path: Path) -> DetectorConfig:
    """Load an immutable model/configuration pair; restart workers after changing either artifact."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
    ):
        raise ValueError("detector configuration requires schema_version 1")
    filename = document.get("model")
    if not isinstance(filename, str) or not filename:
        raise ValueError("detector model must name a local ONNX file")
    model = (path.parent / filename).resolve()
    if model.suffix.lower() != ".onnx" or not model.is_file():
        raise ValueError(f"detector ONNX model is missing: {model}")
    with model.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if document.get("sha256") is not None and document["sha256"] != digest:
        raise ValueError("detector model SHA256 does not match its configuration")
    count = document.get("num_classes", 1)
    ids = document.get("class_ids", [0])
    size = document.get("input_size", 640)
    if type(count) is not int or not 1 <= count <= 4096:
        raise ValueError("detector num_classes must be between 1 and 4096")
    if not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < count for i in ids):
        raise ValueError("detector class_ids must select valid model class indices")
    if type(size) is not int or not 32 <= size <= 1536 or size % 32:
        raise ValueError("detector input_size must be a multiple of 32 between 32 and 1536")
    selection = document.get("selection", "largest")
    missing = document.get("no_detection", "full-image")
    if selection not in {"largest", "confidence"} or missing not in {"full-image", "zero", "error"}:
        raise ValueError("unsupported detector selection or no_detection policy")
    if type(document.get("cache", True)) is not bool:
        raise ValueError("detector cache must be a boolean")
    return DetectorConfig(
        model,
        digest,
        count,
        tuple(sorted(set(ids))),
        size,
        _number(document, "confidence", 0.25, 0, 1),
        _number(document, "iou", 0.45, 0, 1),
        _number(document, "margin", 0.05, 0, 0.5),
        selection,
        missing,
        document.get("cache", True),
    )


def detector_config(enabled=None):
    """Load the configured detector only when the optional detection flag is enabled."""
    if enabled is None:
        enabled = _bool_env("HARBOUR_FL_DETECTOR_ENABLED")
    if not enabled:
        return None
    path = Path(os.environ.get("HARBOUR_FL_DETECTOR_CONFIG", PROJECT_ROOT / "models/vessel-detector/detector.json"))
    return load_config(path.expanduser().resolve())


def preprocessing_signature(enabled=None):
    """Return the semantic detector identity used for compatibility checks, or the disabled signature."""
    config = detector_config(enabled)
    return (
        {key: value for key, value in config.metadata().items() if key != "cache"}
        if config is not None
        else {"enabled": False}
    )


def preprocessing_metadata():
    """Describe detector semantics and runtime/cache settings for diagnostic inspection."""
    config = detector_config()
    return config.metadata() if config is not None else {"enabled": False}

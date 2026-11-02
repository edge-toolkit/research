"""Portable prepared-feature loading and deterministic client batching."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

from .pretrained_data import load_rows, materialize, metadata
from .provenance import file_hash
from .settings import DEFAULT_DATA_ROOT
from .workload import workload


def _data_root() -> Path:
    """Resolve the prepared dataset root and require its manifests and encoder metadata."""
    workload()
    root = Path(os.environ.get("HARBOUR_FL_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()
    required = ("feature_manifest.csv", "class_summary.csv", "clients")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"dataset root {root} is missing {', '.join(missing)}")
    metadata(root)
    return root


def _labels(root: Path) -> list[str]:
    """Read class labels only after verifying the summary hash and its ordering against preparation metadata."""
    info = metadata(root)
    summary = root / "class_summary.csv"
    if info["files"].get(summary.name) != file_hash(summary):
        raise ValueError("prepared class summary changed")
    with summary.open(newline="", encoding="utf-8") as source:
        labels = [row["standard_label"] for row in csv.DictReader(source) if int(row["segments"]) > 0]
    if labels != info["labels"]:
        raise ValueError("prepared class labels do not match feature metadata")
    if len(labels) < 2:
        raise RuntimeError("the dataset must contain at least two classes")
    return labels


def _load_rows(root: Path, manifest: Path):
    return load_rows(root, manifest)


def _batches(rows, batch_size: int, seed: int, *, shuffle: bool):
    """Yield bounded batches, optionally shuffling row indices with a local deterministic seed."""
    indices = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [rows[index] for index in indices[start : start + batch_size]]


def _materialize(batch, labels: list[str], torch, condition=None):
    return materialize(batch, labels, torch, condition)

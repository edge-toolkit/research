"""Evaluate fixed final weights under separately named prediction-time input conditions."""

from __future__ import annotations

import time

from .learning import _evaluate

VERSION = 1
CONDITIONS = {
    "clean": {},
    "audio_snr10": {"audio_snr_db": 10},
    "audio_snr0": {"audio_snr_db": 0},
    "image_noise005": {"image_noise_std": 0.05},
    "image_noise010": {"image_noise_std": 0.1},
    "image_noise020": {"image_noise_std": 0.2},
    "missing_audio": {"missing_modality": "audio"},
    "missing_image": {"missing_modality": "image"},
}


def evaluate_inputs(model, rows, labels, torch, reference, emit, progress):
    """Reuse the final clean evaluation and vary only the inputs; no adaptation or training is performed."""
    results = {}
    for name, condition in CONDITIONS.items():
        started = time.monotonic()
        metrics = reference if name == "clean" else _evaluate(model, rows, labels, torch, input_condition=condition)
        results[name] = metrics
        emit(
            "quality-evaluation",
            condition=name,
            input_condition=condition,
            scope="prediction-time",
            progress=progress,
            samples=len(rows),
            metrics=metrics,
            seconds=None if name == "clean" else time.monotonic() - started,
        )
    return results

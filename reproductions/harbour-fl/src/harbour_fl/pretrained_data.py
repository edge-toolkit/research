"""Load verified cached pretrained embeddings and explicit prediction-input conditions."""

import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from .pretrained_assets import fingerprint
from .provenance import digest, file_hash
from .settings import _bool_env, _bounded_float_env


def metadata(root):
    """Verify preparation metadata integrity and require the current encoder identity and feature dimension."""
    path = Path(root) / "preparation.json"
    if not path.is_file():
        raise ValueError("pretrained profile requires prepared frozen features; run python -m harbour_fl.prepare")
    value = json.loads(path.read_text())
    if digest({key: item for key, item in value.items() if key != "fingerprint"}) != value["fingerprint"]:
        raise ValueError("prepared feature metadata changed")
    if value["input"]["pretrained_assets"] != fingerprint() or value["feature_dimension"] != 960:
        raise ValueError("prepared features use different pretrained encoders")
    return value


def load_rows(root, manifest):
    """Load a nonempty prepared partition after verifying its manifest and referenced feature-archive hashes."""
    root = Path(root)
    info = metadata(root)
    relative = str(Path(manifest).relative_to(root))
    if info["files"].get(relative) != file_hash(manifest):
        raise ValueError("prepared partition manifest changed")
    result = []
    with Path(manifest).open(newline="") as source:
        for row in csv.DictReader(source):
            feature = root / row["feature_path"]
            if info["files"].get(row["feature_path"]) != file_hash(feature):
                raise ValueError(f"prepared feature archive changed: {row['segment_id']}")
            result.append(
                {
                    "segment_id": row["segment_id"],
                    "label": row["standard_label"],
                    "feature": str(feature),
                    "image": str(feature),
                    "class_weights": info["class_weights"],
                }
            )
    if not result:
        raise ValueError("prepared partition is empty")
    return result


def configured_condition():
    """Read configured noise and missing-input faults without selecting a random augmentation condition."""
    condition = {
        "missing_modality": os.environ.get("HARBOUR_FL_MISSING_MODALITY", "none"),
        "audio_noise_std": float(os.environ.get("HARBOUR_FL_AUDIO_NOISE_STD", "0")),
        "image_noise_std": float(os.environ.get("HARBOUR_FL_IMAGE_NOISE_STD", "0")),
    }
    if "HARBOUR_FL_AUDIO_SNR_DB" in os.environ:
        condition["audio_snr_db"] = float(os.environ["HARBOUR_FL_AUDIO_SNR_DB"])
    return condition


def materialize(batch, labels, torch, condition=None):
    """Load the selected cached input condition, apply explicit missing-modality masks and return labelled tensors."""
    if _bool_env("HARBOUR_FL_DETECTOR_ENABLED"):
        raise ValueError("pretrained features use complete images; disable the optional camera detector")
    if condition is None:
        condition = configured_condition()
    if condition.get("audio_noise_std", 0):
        raise ValueError("pretrained audio uses waveform SNR; use audio_snr_db instead of audio_noise_std")
    key = condition.get("prepared_condition", "clean")
    noise = condition.get("image_noise_std", 0)
    if "audio_snr_db" in condition:
        if noise:
            raise ValueError("simultaneous audio and image corruption was not prepared")
        snr = condition["audio_snr_db"]
        if snr not in (0, 10, 20):
            raise ValueError("prepared waveform SNR must be 0, 10 or 20 dB")
        key = f"audio-snr{int(snr)}"
    elif noise:
        if noise not in (0.05, 0.1, 0.2):
            raise ValueError("prepared image noise standard deviation must be 0.05, 0.10 or 0.20")
        key = {0.05: "image-noise005", 0.1: "image-noise010", 0.2: "image-noise020"}[noise]
    audio, images = [], []
    for row in batch:
        delay_ms = _bounded_float_env("HARBOUR_FL_IO_READ_DELAY_MS", maximum=5000)
        if delay_ms:
            time.sleep(delay_ms / 1000)
        with np.load(row["feature"], allow_pickle=False) as archive:
            values = [archive[key + suffix].copy() for suffix in ("_audio", "_image")]
        if any(value.shape != (960,) or not np.isfinite(value).all() for value in values):
            raise ValueError("invalid pretrained feature vector")
        audio.append(torch.from_numpy(values[0]))
        images.append(torch.from_numpy(values[1]))
    audio, images = torch.stack(audio), torch.stack(images)
    missing = condition.get("missing_modality", "none")
    if missing not in ("none", "audio", "image"):
        raise ValueError("missing modality must be none, audio or image")
    if missing == "audio":
        audio.zero_()
    elif missing == "image":
        images.zero_()
    return audio, images, torch.tensor([labels.index(row["label"]) for row in batch], dtype=torch.long)

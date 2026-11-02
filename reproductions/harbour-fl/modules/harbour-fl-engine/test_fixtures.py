"""Small portable frozen-feature partitions for application tests; no downloaded weights are needed."""

import csv
from pathlib import Path

import numpy as np
from harbour_fl.pretrained_assets import fingerprint
from harbour_fl.provenance import digest, file_hash, save


def prepared(root):
    """Create hashed, portable feature partitions with distinct cached conditions and no downloaded model weights."""
    root = Path(root)
    root.mkdir(exist_ok=True)
    vectors = {}
    for index, key in enumerate(
        (
            "clean",
            "audio-snr20",
            "audio-snr10",
            "audio-snr0",
            "image-noise005",
            "image-noise010",
            "image-noise020",
        )
    ):
        audio_offset = index if key.startswith("audio-") else 0
        image_offset = index if key.startswith("image-") else 0
        vectors[key + "_audio"] = np.linspace(0, 1, 960, dtype=np.float32) + audio_offset
        vectors[key + "_image"] = np.linspace(1, 0, 960, dtype=np.float32) + image_offset
    np.savez(root / "sample.npz", **vectors)
    files = {"sample.npz": file_hash(root / "sample.npz")}
    paths = [root / "test_manifest.csv", root / "feature_manifest.csv"]
    paths += [
        root / "clients" / f"vessel_{client}" / f"{split}_manifest.csv"
        for client in (1, 2, 3)
        for split in ("train", "val")
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["segment_id", "standard_label", "feature_path"])
            writer.writeheader()
            for label in ("a", "b"):
                writer.writerow(
                    {
                        "segment_id": str(path.relative_to(root)) + label,
                        "standard_label": label,
                        "feature_path": "sample.npz",
                    }
                )
        files[str(path.relative_to(root))] = file_hash(path)
    (root / "class_summary.csv").write_text("standard_label,segments\na,1\nb,1\n")
    files["class_summary.csv"] = file_hash(root / "class_summary.csv")
    record = {
        "input": {"pretrained_assets": fingerprint()},
        "files": files,
        "feature_dimension": 960,
        "class_weights": [1.0, 1.0],
        "labels": ["a", "b"],
    }
    record["fingerprint"] = digest(record)
    save(root / "preparation.json", record)
    return root

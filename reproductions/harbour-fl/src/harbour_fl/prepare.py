"""Prepare portable three-client partitions from raw paired samples using frozen pretrained encoders."""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision

from .devices import select_device
from .prepare_dataset import prepare as import_dataset
from .prepare_dataset import validate_client_coverage
from .pretrained_assets import assets_root, fingerprint
from .pretrained_inputs import CONDITIONS, materialize
from .pretrained_model import Encoders
from .provenance import digest, file_hash, save

FEATURE_CONDITIONS = tuple(name for name, value in CONDITIONS.items() if "missing" not in value)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def encode(document, output, assets, device, seed=17):
    """Cache exact frozen embeddings; downstream FL trains and exchanges only the fusion head."""
    validate_client_coverage(document["rows"])
    output = Path(output)
    device = torch.device(device)
    identity = {
        "schema_version": 1,
        "source_fingerprint": document["fingerprint"],
        "pretrained_assets": fingerprint(),
        "seed": seed,
        "conditions": FEATURE_CONDITIONS,
        "device": str(device),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "encoder_source": {
            name: file_hash(Path(__file__).with_name(name)) for name in ("pretrained_model.py", "pretrained_inputs.py")
        },
    }
    identity = json.loads(json.dumps(identity))
    marker = output / "preparation.json"
    if marker.exists():
        existing = json.loads(marker.read_text())
        if existing["input"] != identity:
            raise ValueError("preparation settings changed; select a new output directory")
        for name, expected in existing["files"].items():
            if file_hash(output / name) != expected:
                raise ValueError(f"prepared artifact changed: {name}")
        return existing
    encoders = Encoders(assets).place(device)
    output.mkdir(parents=True, exist_ok=True)
    records, features, hashes = [], [], {}
    for index, row in enumerate(document["rows"]):
        relative = f"features/{index:08d}.npz"
        path = output / relative
        path.parent.mkdir(exist_ok=True)
        values = {}
        for condition in FEATURE_CONDITIONS:
            waveform, image = materialize(row, condition, seed)
            audio, camera = encoders(waveform[None], image[None].to(device))
            values[condition + "_audio"] = audio[0].cpu().numpy().copy()
            values[condition + "_image"] = camera[0].cpu().numpy().copy()
        np.savez_compressed(path, **values)
        hashes[relative] = file_hash(path)
        record = {"segment_id": row["segment_id"], "standard_label": row["label"], "feature_path": relative}
        records.append((row, record))
        features.append({"segment_id": row["segment_id"], "feature_path": relative})
        if (index + 1) % 100 == 0:
            print(f"Encoded {index + 1}/{len(document['rows'])} paired samples", flush=True)
    paths = []
    for client in ("vessel_1", "vessel_2", "vessel_3"):
        for split in ("train", "val"):
            selected = [
                record
                for row, record in records
                if row["split"] == split and (split == "val" or row["client"] == "client-" + client[-1])
            ]
            if not selected:
                raise ValueError(f"{client} has no {split} samples; use more recording groups or another split seed")
            path = output / "clients" / client / f"{split}_manifest.csv"
            write_csv(path, selected)
            paths.append(path)
    path = output / "test_manifest.csv"
    write_csv(path, [record for row, record in records if row["split"] == "test"])
    paths.append(path)
    path = output / "feature_manifest.csv"
    write_csv(path, features)
    paths.append(path)
    counts = Counter(row["label"] for row in document["rows"] if row["split"] == "train")
    path = output / "class_summary.csv"
    write_csv(
        path,
        [
            {"standard_label": label, "segments": sum(row["label"] == label for row in document["rows"])}
            for label in document["labels"]
        ],
    )
    paths.append(path)
    hashes.update({str(path.relative_to(output)): file_hash(path) for path in paths})
    result = {
        "input": identity,
        "labels": document["labels"],
        "class_weights": [sum(counts.values()) / (len(counts) * counts[label]) for label in document["labels"]],
        "files": hashes,
        "split_scope": document["scope"],
        "validation_scope": "shared held-out validation partition; no validation samples enter fitting",
        "feature_dimension": 960,
    }
    result["fingerprint"] = digest(result)
    save(marker, result)
    return result


def main():
    """Import paired recordings, select the extraction device and cache verified pretrained features and partitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Directory containing segment_manifest.csv")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--assets", type=Path, default=assets_root())
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if os.environ.get("HARBOUR_FL_DETECTOR_ENABLED", "false").lower() in {"1", "true", "yes", "on"}:
        parser.error("pretrained preparation uses complete images; disable the optional camera detector")
    torch.set_num_threads(2)
    device = select_device(torch, args.device)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(device.type != "mps", warn_only=True)
    source = import_dataset(args.dataset, args.output / "source", raw_root=args.raw_root, seed=args.seed)
    encode(source, args.output, args.assets, device, args.seed)
    print(f"Prepared pretrained features and partitions: {args.output}")


if __name__ == "__main__":
    main()

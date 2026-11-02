"""Measure actual three-client FL work on CPU, MPS or CUDA before selecting benchmark timeouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from .checkpoints import _checkpoint_bytes, _load_checkpoint
from .data import _data_root, _labels, _load_rows
from .devices import select_device, synchronize
from .learning import _evaluate, _fed_avg, _import_torch, _new_model, _train_local
from .settings import DEFAULT_CLIENTS, _bool_env
from .vision import preprocessing_metadata
from .workload import describe


def inventory(root, labels, limit=0, evaluation_limit=0):
    """Read the prepared partitions without duplicating data or changing the existing manifests."""
    clients, manifests = {}, {}
    for client in DEFAULT_CLIENTS:
        partitions = {}
        for split in ("train", "val"):
            path = root / "clients" / client / f"{split}_manifest.csv"
            rows = _load_rows(root, path)
            selected = rows[:limit] if limit else rows
            with path.open(newline="") as source:
                declared = sum(1 for _ in csv.DictReader(source))
            partitions[split] = selected
            manifests[str(path.relative_to(root))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "usable_samples": len(rows),
                "manifest_samples": declared,
                "excluded_samples": declared - len(rows),
                "selected_samples": len(selected),
                "class_counts": dict(Counter(row["label"] for row in selected)),
            }
        clients[client] = partitions
    path = root / "test_manifest.csv"
    test = _load_rows(root, path)
    selected_test = test[:evaluation_limit] if evaluation_limit else test
    with path.open(newline="") as source:
        declared_test = sum(1 for _ in csv.DictReader(source))
    manifests[path.name] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "usable_samples": len(test),
        "manifest_samples": declared_test,
        "excluded_samples": declared_test - len(test),
        "selected_samples": len(selected_test),
    }
    train_ids = {r["segment_id"] for values in clients.values() for r in values["train"]}
    validation_ids = {r["segment_id"] for values in clients.values() for r in values["val"]}
    test_ids = {r["segment_id"] for r in selected_test}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("training, validation and evaluation partitions contain overlapping segment IDs")
    return clients, selected_test, {"labels": labels, "manifests": manifests}


def measure(torch, device, clients, test, labels, rounds):
    """Execute clients sequentially with FedAvg; measure compute and serialization independently of transport."""
    global_model = _new_model(torch, len(labels))
    records = []
    for number in range(rounds):
        updates, measurements = [], []
        for name, partitions in clients.items():
            model = _new_model(torch, len(labels)).to(device)
            model.load_state_dict(global_model.state_dict())
            torch.manual_seed(int(os.environ.get("HARBOUR_FL_TRAINING_SEED", "17")) + number)
            metrics = _train_local(model, partitions["train"], labels, number, torch)
            validation = _evaluate(model, partitions["val"], labels, torch)
            synchronize(torch, device)
            started = time.monotonic()
            payload = _checkpoint_bytes(model, labels, number, torch)
            serialize_seconds = time.monotonic() - started
            updates.append({"samples": len(partitions["train"]), "checkpoint": _load_checkpoint(payload, torch)})
            measurements.append(
                {
                    "client": name,
                    **metrics,
                    "validation": validation,
                    "update_bytes": len(payload),
                    "serialization_seconds": serialize_seconds,
                }
            )
            del model
        started = time.monotonic()
        _fed_avg(global_model, updates, torch)
        aggregation_seconds = time.monotonic() - started
        evaluation = _evaluate(global_model, test, labels, torch)
        started = time.monotonic()
        payload = _checkpoint_bytes(global_model, labels, number + 1, torch)
        records.append(
            {
                "round": number,
                "clients": measurements,
                "aggregation_seconds": aggregation_seconds,
                "evaluation": evaluation,
                "checkpoint_bytes": len(payload),
                "checkpoint_serialization_seconds": time.monotonic() - started,
            }
        )
        print(f"Calibration round {number + 1}/{rounds} complete", flush=True)
    return global_model, records


def main():
    """Write a device-specific calibration report using real data, trainable parameters and serialized updates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--samples-per-client", type=int, default=0, help="0 uses each complete prepared partition")
    parser.add_argument("--evaluation-samples", type=int, default=0, help="0 uses the complete held-out test set")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name, upper in (("rounds", 10), ("local_epochs", 50), ("batch_size", 256), ("torch_threads", 32)):
        if not 1 <= getattr(args, name) <= upper:
            parser.error(f"{name} must be between 1 and {upper}")
    if min(args.samples_per_client, args.evaluation_samples) < 0:
        parser.error("sample limits must be nonnegative")
    os.environ.update(
        HARBOUR_FL_MODEL_PROFILE="pretrained",
        HARBOUR_FL_LOCAL_EPOCHS=str(args.local_epochs),
        HARBOUR_FL_BATCH_SIZE=str(args.batch_size),
        ET_BENCHMARK_GPU_FAULTS="0",
    )
    torch = _import_torch()
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(int(os.environ.get("HARBOUR_FL_TRAINING_SEED", "17")))
    device = select_device(torch, args.device)
    root = _data_root()
    labels = _labels(root)
    clients, test, dataset = inventory(root, labels, args.samples_per_client, args.evaluation_samples)
    # Inspect a separate model; timed training still includes first-use accelerator and data work.
    warmup = _new_model(torch, len(labels)).to(device)
    profile = describe(warmup, torch, device)
    del warmup
    seed = int(os.environ.get("HARBOUR_FL_TRAINING_SEED", "17"))
    torch.manual_seed(seed)
    started = time.monotonic()
    model, records = measure(torch, device, clients, test, labels, args.rounds)
    document = {
        "schema_version": 1,
        "execution": "sequential-native-calibration",
        "workload": profile,
        "training_seed": seed,
        "adaptation": _bool_env("HARBOUR_FL_ADAPTATION"),
        "dataset": dataset,
        "image_preprocessing": preprocessing_metadata(),
        "rounds": records,
        "elapsed_seconds": time.monotonic() - started,
        "sample_limited": bool(args.samples_per_client or args.evaluation_samples),
        "coordinator_device": str(next(model.parameters()).device),
        "cache_condition": "operator-controlled; first round includes initial data and accelerator work",
        "perturbations": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("HARBOUR_FL_")
            and key
            in {
                "HARBOUR_FL_MISSING_MODALITY",
                "HARBOUR_FL_AUDIO_NOISE_STD",
                "HARBOUR_FL_AUDIO_SNR_DB",
                "HARBOUR_FL_IMAGE_NOISE_STD",
                "HARBOUR_FL_IO_READ_DELAY_MS",
            }
        },
        "measurement_scope": (
            "Training, evaluation, aggregation and serialization; "
            "deployment and network recovery use deployed benchmarks."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"Calibration report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Multimodal model, local training, evaluation and FedAvg."""

from __future__ import annotations

import os
import time
from typing import Any

from .data import _batches, _materialize
from .devices import check_gpu_fault, synchronize
from .pretrained_data import configured_condition
from .settings import _bool_env, _positive_int_env
from .workload import workload


def _import_torch():
    """Import PyTorch lazily and explain the required setup when it is unavailable."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; run mise run setup:python in the Harbour-FL project") from exc
    return torch


def _new_model(_torch, class_count: int):
    from .pretrained_model import Fusion

    workload()
    return Fusion(class_count)


def _train_local(model, rows: list[dict[str, str]], labels: list[str], round_number: int, torch) -> dict[str, float]:
    """Train the fusion head with per-round AdamW, optional input augmentation and measured client work."""
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(os.environ.get("HARBOUR_FL_LEARNING_RATE", "0.001")), weight_decay=0.01
    )
    epochs = _positive_int_env("HARBOUR_FL_LOCAL_EPOCHS", 1)
    batch_size = _positive_int_env("HARBOUR_FL_BATCH_SIZE", 16)
    total_loss = 0.0
    total = 0
    correct = 0
    steps = 0
    materialization_ms: list[float] = []
    training_started = time.monotonic()
    device = next(model.parameters()).device
    weights = torch.tensor(rows[0]["class_weights"], device=device)
    for epoch in range(epochs):
        for batch in _batches(rows, batch_size, 10000 * round_number + epoch, shuffle=True):
            check_gpu_fault(device, torch)
            read_started = time.monotonic()
            if _bool_env("HARBOUR_FL_ADAPTATION"):
                condition = configured_condition()
                if (
                    not condition["audio_noise_std"]
                    and not condition["image_noise_std"]
                    and "audio_snr_db" not in condition
                ):
                    choices = ("clean", "audio-snr10", "image-noise010")
                    condition["prepared_condition"] = choices[int(torch.randint(3, ()))]
                audio, images, targets = _materialize(batch, labels, torch, condition)
            else:
                audio, images, targets = _materialize(batch, labels, torch)
            materialization_ms.extend([(time.monotonic() - read_started) * 1000.0 / len(batch)] * len(batch))
            audio, images, targets = (tensor.to(device) for tensor in (audio, images, targets))
            optimizer.zero_grad(set_to_none=True)
            logits = model(audio, images)
            loss = torch.nn.functional.cross_entropy(logits, targets, weight=weights)
            loss.backward()
            optimizer.step()
            steps += 1
            total_loss += float(loss.item()) * len(batch)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += len(batch)
    ordered_reads = sorted(materialization_ms)
    synchronize(torch, device)
    p95_index = round((len(ordered_reads) - 1) * 0.95)
    elapsed = max(time.monotonic() - training_started, 1e-09)
    return {
        "train_loss": total_loss / total,
        "train_accuracy": correct / total,
        "dataset_read_mean_ms": sum(materialization_ms) / len(materialization_ms),
        "dataset_read_p95_ms": ordered_reads[p95_index],
        "samples_per_second": total / elapsed,
        "training_seconds": elapsed,
        "processed_samples": total,
        "optimizer_steps": steps,
        "local_epochs": epochs,
        "batch_size": batch_size,
    }


def _evaluate(model, rows: list[dict[str, str]], labels: list[str], torch, *, input_condition=None) -> dict[str, Any]:
    """Evaluate fixed weights on selected input conditions and return loss, accuracy, macro F1 and confusion counts."""
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    confusion = [[0 for _ in labels] for _ in labels]
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in _batches(rows, _positive_int_env("HARBOUR_FL_BATCH_SIZE", 16), 0, shuffle=False):
            (audio, images, targets) = (
                _materialize(batch, labels, torch)
                if input_condition is None
                else _materialize(batch, labels, torch, input_condition)
            )
            audio, images, targets = (tensor.to(device) for tensor in (audio, images, targets))
            logits = model(audio, images)
            total_loss += float(torch.nn.functional.cross_entropy(logits, targets).item()) * len(batch)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            total += len(batch)
            for target, prediction in zip(targets.tolist(), predictions.tolist(), strict=True):
                confusion[target][prediction] += 1
    f1_values = []
    for index in range(len(labels)):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(len(labels)) if row != index)
        false_negative = sum(confusion[index][column] for column in range(len(labels)) if column != index)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "confusion_matrix": confusion,
    }


def _fed_avg(model, updates: list[dict[str, Any]], torch) -> None:
    """Aggregate head tensors by client sample count and reject updates with no contributing samples."""
    total_samples = sum(int(update["samples"]) for update in updates)
    if total_samples <= 0:
        raise ValueError("FedAvg received no samples")
    averaged = {}
    reference = model.state_dict()
    for name, reference_tensor in reference.items():
        tensors = [update["checkpoint"]["state_dict"][name] for update in updates]
        if reference_tensor.is_floating_point():
            value = torch.zeros_like(reference_tensor)
            for update, tensor in zip(updates, tensors, strict=True):
                value.add_(
                    tensor.to(device=value.device, dtype=value.dtype), alpha=int(update["samples"]) / total_samples
                )
            averaged[name] = value
        else:
            averaged[name] = tensors[0].to(reference_tensor.device)
    model.load_state_dict(averaged, strict=True)

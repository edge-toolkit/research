"""Model checkpoint serialization; persistence is provided by Core."""

from __future__ import annotations

import io
from typing import Any

from .settings import SCHEMA_VERSION
from .workload import signature, validate_checkpoint


def _checkpoint_bytes(model, labels: list[str], round_number: int, torch, **metadata) -> bytes:
    """Serialize CPU model tensors, class labels and workload identity together with round metadata."""
    buffer = io.BytesIO()
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "round": round_number,
            "labels": labels,
            "state_dict": {name: tensor.detach().cpu() for (name, tensor) in model.state_dict().items()},
            "workload": signature(),
            **metadata,
        },
        buffer,
    )
    return buffer.getvalue()


def _load_checkpoint(payload: bytes, torch) -> dict[str, Any]:
    """Load weights-only checkpoint data on CPU and reject invalid schemas or incompatible model identities."""
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != SCHEMA_VERSION
        or (not isinstance(checkpoint.get("state_dict"), dict))
    ):
        raise ValueError("invalid harbour FL checkpoint")
    validate_checkpoint(checkpoint)
    return checkpoint

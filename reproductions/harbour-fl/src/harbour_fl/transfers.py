"""Application payload-transfer outcomes, independent of training and serialization timings."""

from __future__ import annotations

import time
import uuid

VERSION = 1


def measure_transfer(runtime, direction, progress, operation, *, payload_bytes=None):
    """Record one SDK storage operation, retaining its original exception and an independently correlated outcome."""
    identifier = uuid.uuid4().hex
    runtime._event(
        "model-transfer-started",
        transfer_id=identifier,
        direction=direction,
        progress=progress,
        payload_bytes=payload_bytes,
    )
    started = time.monotonic()
    try:
        value = operation()
        if direction == "download":
            if value is None:
                raise RuntimeError("global model payload was not found")
            payload_bytes = len(value)
    except Exception as error:
        elapsed = time.monotonic() - started
        timeout = (
            isinstance(error, TimeoutError) or "timed out" in str(error).lower() or "timeout" in str(error).lower()
        )
        runtime._event(
            "model-transfer-finished",
            transfer_id=identifier,
            direction=direction,
            progress=progress,
            status="timeout" if timeout else "failure",
            seconds=elapsed,
            payload_bytes=payload_bytes,
            error_type=type(error).__name__,
        )
        raise
    elapsed = time.monotonic() - started
    runtime._event(
        "model-transfer-finished",
        transfer_id=identifier,
        direction=direction,
        progress=progress,
        status="success",
        seconds=elapsed,
        payload_bytes=payload_bytes,
    )
    return value, elapsed

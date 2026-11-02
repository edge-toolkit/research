"""Environment settings and wire-protocol constants."""

from __future__ import annotations

import os

ROUND_TYPE = "harbour-fl-round"
UPDATE_TYPE = "harbour-fl-update"
COMPLETE_TYPE = "harbour-fl-complete"
COMMAND_TYPE = "harbour-fl-command"
STATUS_TYPE = "harbour-fl-status"
CLIENT_STATUS_TYPE = "harbour-fl-client-status"
COORDINATOR_STATUS_TYPE = "harbour-fl-coordinator-status"
MLFLOW_TYPE = "mlflow-event"
SCHEMA_VERSION = 1
MLFLOW_SCHEMA_VERSION = 1
MLFLOW_PRODUCER = "harbour-fl-demo"
DEFAULT_DATA_ROOT = "data/prepared"
DEFAULT_CLIENTS = ("vessel_1", "vessel_2", "vessel_3")


def _bool_env(name: str, default: bool = False) -> bool:
    """Read an environment flag using the supported true spellings, or return its default when unset."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _positive_int_env(name: str, default: int) -> int:
    """Read an integer setting, falling back to the supplied default for nonpositive values."""
    value = int(os.environ.get(name, str(default)))
    return value if value > 0 else default


def _bounded_float_env(name: str, default: float = 0.0, maximum: float = 5.0) -> float:
    """Parse a bounded nonnegative numeric setting and reject malformed or out-of-range values."""
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if not 0.0 <= value <= maximum:
        raise RuntimeError(f"{name} must be between 0 and {maximum:g}")
    return value

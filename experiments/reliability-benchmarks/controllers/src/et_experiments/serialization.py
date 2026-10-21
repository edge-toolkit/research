"""Finite measurement checks and standard JSON for reports, exports and browser responses."""

from __future__ import annotations

import json
import math
from typing import Any


def finite_number(value: Any) -> bool:
    """Accept finite measurements without treating booleans as numeric samples."""
    return type(value) in (int, float) and math.isfinite(value)


def json_safe(value: Any) -> Any:
    """Copy JSON containers, replacing non-finite floats with None while retaining ordinary values."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def json_dumps(value: Any, **options: Any) -> str:
    """Serialize reports as standard JSON, representing non-finite measurements as null."""
    return json.dumps(json_safe(value), allow_nan=False, **options)

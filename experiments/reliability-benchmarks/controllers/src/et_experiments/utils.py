"""Small shared measurement and command helpers."""

from __future__ import annotations

import datetime as dt
import re
import subprocess

ANSI = re.compile(r"\x1b\[[0-9;]*m")


class BenchmarkError(RuntimeError):
    """Expected benchmark/configuration failure."""


def run(
    *args: str,
    check: bool = True,
    input_text: str | None = None,
    timeout: float | None = 600,
) -> subprocess.CompletedProcess[str]:
    """Capture command output with a ten-minute default deadline and optional caller override."""
    completed = subprocess.run(
        args,
        check=False,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BenchmarkError(f"{' '.join(args)} failed: {detail}")
    return completed


def utc_now() -> dt.datetime:
    """Return a timezone-aware UTC instant for report and fault timestamps."""
    return dt.datetime.now(dt.timezone.utc)


def iso(value: dt.datetime) -> str:
    """Format a timestamp as ISO 8601, writing a UTC offset as Z."""
    return value.isoformat().replace("+00:00", "Z")


def progress(started: dt.datetime, message: str) -> None:
    """Print and flush a progress message prefixed with elapsed wall-clock minutes and seconds."""
    elapsed = int((utc_now() - started).total_seconds())
    minutes, seconds = divmod(elapsed, 60)
    print(f"[{minutes:02d}:{seconds:02d}] {message}", flush=True)


def percentile(values: list[float], fraction: float) -> float | None:
    """Select the nearest indexed order statistic, returning None for an empty sample population."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def parse_cpu(value: str) -> float:
    """Convert a Kubernetes CPU quantity in cores or millicores to millicores."""
    return float(value[:-1]) if value.endswith("m") else float(value) * 1000.0


def parse_memory(value: str) -> float:
    """Convert byte or Ki/Mi/Gi/Ti memory quantities to mebibytes."""
    units = {"Ki": 1 / 1024, "Mi": 1, "Gi": 1024, "Ti": 1024 * 1024}
    for suffix, multiplier in units.items():
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * multiplier
    return float(value) / (1024 * 1024)


def log_timestamp(line: str) -> dt.datetime | None:
    """Parse an ISO timestamp at the start of a log line, returning None for unrelated text."""
    token = line.split(" ", 1)[0]
    try:
        return dt.datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None


def clean_log(line: str) -> str:
    """Remove ANSI styling escapes before parsing or displaying collected log text."""
    return ANSI.sub("", line)

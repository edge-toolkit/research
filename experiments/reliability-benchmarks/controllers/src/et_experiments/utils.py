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
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
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
    return dt.datetime.now(dt.timezone.utc)


def iso(value: dt.datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def progress(started: dt.datetime, message: str) -> None:
    elapsed = int((utc_now() - started).total_seconds())
    minutes, seconds = divmod(elapsed, 60)
    print(f"[{minutes:02d}:{seconds:02d}] {message}", flush=True)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def parse_cpu(value: str) -> float:
    return float(value[:-1]) if value.endswith("m") else float(value) * 1000.0


def parse_memory(value: str) -> float:
    units = {"Ki": 1 / 1024, "Mi": 1, "Gi": 1024, "Ti": 1024 * 1024}
    for suffix, multiplier in units.items():
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * multiplier
    return float(value) / (1024 * 1024)


def log_timestamp(line: str) -> dt.datetime | None:
    token = line.split(" ", 1)[0]
    try:
        return dt.datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None


def clean_log(line: str) -> str:
    return ANSI.sub("", line)

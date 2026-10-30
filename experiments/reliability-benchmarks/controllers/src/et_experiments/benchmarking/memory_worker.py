"""Bounded RAM occupancy inside the application's cgroup; executable through kubectl exec."""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

MIB = 1024 * 1024


def memory_snapshot(root=Path("/sys/fs/cgroup")):
    """Read charged memory usage and validate the container's finite cgroup limit."""
    if (root / "memory.max").exists():
        maximum = (root / "memory.max").read_text().strip()
        current = int((root / "memory.current").read_text())
        if maximum == "max":
            raise ValueError("client memory saturation requires a finite container memory limit")
        limit = int(maximum)
    else:
        root = root / "memory" if (root / "memory").is_dir() else root
        limit = int((root / "memory.limit_in_bytes").read_text())
        current = int((root / "memory.usage_in_bytes").read_text())
    if not 0 < limit <= 8 * 1024 * MIB:
        raise ValueError("client memory limit must be finite and at most 8 GiB for this bounded experiment")
    return current, limit


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_profile(options, snapshot=memory_snapshot, clock=time.monotonic, sleep=time.sleep):
    """Increase occupancy in small touched chunks, with headroom and a hard additional-allocation cap."""
    baseline, limit = snapshot()
    cap = int(options["max_extra_memory_mib"] * MIB)
    reserve = int(options["reserve_mib"] * MIB)
    stages, chunks = [], []
    peak = baseline
    guarded = False
    try:
        for fraction in options["levels"]:
            stage = {"target_utilization": fraction, "started_at": timestamp(), "samples": []}
            deadline = clock() + options["hold_seconds"]
            while clock() < deadline:
                current, observed_limit = snapshot()
                if observed_limit != limit:
                    raise ValueError("container memory limit changed during the profile")
                peak = max(peak, current)
                if current >= min(limit * 0.93, limit - reserve):
                    guarded = True
                    break
                allocated = sum(map(len, chunks))
                wanted = max(0, min(cap - allocated, int(limit * fraction) - current))
                if wanted:
                    size = min(wanted, 4 * MIB, int(limit * 0.93) - current, limit - reserve - current)
                    block = bytearray(max(0, size))
                    chunks.append(block)
                for block in chunks:
                    for offset in range(0, len(block), 4096):
                        block[offset] = (block[offset] + 1) % 256
                current, _limit = snapshot()
                peak = max(peak, current)
                stage["samples"].append(
                    {
                        "at": timestamp(),
                        "memory_bytes": current,
                        "utilization": current / limit,
                        "extra_bytes": sum(map(len, chunks)),
                    }
                )
                sleep(0.2)
            stage.update(finished_at=timestamp(), safety_stop=guarded)
            stages.append(stage)
            if guarded:
                break
    finally:
        chunks.clear()
        # Release the final chunk's local reference along with the allocation list.
        block = None
    released_at = timestamp()
    sleep(options.get("release_seconds", 2))
    after, _limit = snapshot()
    extra_peak = max((s["extra_bytes"] / MIB for stage in stages for s in stage["samples"]), default=0)
    reached = not guarded and extra_peak > 0 and peak / limit >= max(options["levels"]) - 0.03
    return {
        "status": "restored" if reached else "not-observed",
        "stages": stages,
        "baseline_memory_bytes": baseline,
        "memory_limit_bytes": limit,
        "released_at": released_at,
        "after_release_memory_bytes": after,
        "safety_stop": guarded,
        "metrics": {
            "memory_utilization_peak": peak / limit,
            "memory_extra_peak_mib": extra_peak,
            "pressure_observed": int(reached),
        },
        "measurement_note": "Cgroup RAM occupancy, including charged cache; not memory-bandwidth saturation.",
    }


if __name__ == "__main__":
    print(json.dumps(run_profile(json.loads(sys.argv[1]))), flush=True)

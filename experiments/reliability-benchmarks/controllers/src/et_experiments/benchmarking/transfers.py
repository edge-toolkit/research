"""Summarize correlated application transfers without mistaking missing outcomes for successes."""

from __future__ import annotations

import math

from et_experiments.recovery.statistics import distribution


def transfer_summary(events, workers):
    """Retain observed attempt populations, unknown outcomes, per-client durations and effective payload rates."""
    starts, finishes = {}, {}
    for event in events:
        if event.get("workload") not in workers or event.get("direction") not in {"download", "upload"}:
            continue
        key = (event["workload"], event.get("instance"), event.get("transfer_id"))
        if not all(key):
            continue
        if event["kind"] == "model-transfer-started":
            starts.setdefault(key, event)
        elif event["kind"] == "model-transfer-finished":
            finishes.setdefault(key, event)
    records = []
    for key, start in starts.items():
        end = finishes.get(key)
        valid = (
            end is not None
            and end["at"] >= start["at"]
            and end.get("progress") == start.get("progress")
            and end["direction"] == start["direction"]
            and end.get("status") in {"success", "failure", "timeout"}
        )
        seconds = end.get("seconds") if valid else None
        payload = end.get("payload_bytes") if valid else start.get("payload_bytes")
        rate = (
            payload / 1048576 / seconds
            if valid
            and end["status"] == "success"
            and type(seconds) in (int, float)
            and math.isfinite(seconds)
            and seconds > 0
            and type(payload) is int
            and payload >= 0
            else None
        )
        records.append(
            {
                "workload": key[0],
                "instance": key[1],
                "transfer_id": key[2],
                "direction": start["direction"],
                "progress": start.get("progress"),
                "started_at": start["at"],
                "status": end["status"] if valid else "unknown",
                "seconds": seconds,
                "payload_bytes": payload,
                "mib_per_second": rate,
            }
        )
    groups = {}
    orphan_finishes = [event for key, event in finishes.items() if key not in starts]
    for worker in workers:
        groups[worker] = {}
        for direction in ("download", "upload"):
            rows = [r for r in records if r["workload"] == worker and r["direction"] == direction]
            good = [r for r in rows if r["status"] == "success"]
            unknown = sum(r["status"] == "unknown" for r in rows)
            orphans = sum(e["workload"] == worker and e["direction"] == direction for e in orphan_finishes)
            complete = bool(rows) and not unknown and not orphans
            groups[worker][direction] = {
                "status": "measured" if complete else "incomplete" if rows or orphans else "not-observed",
                "attempts": len(rows),
                "successful": len(good),
                "failures": sum(r["status"] == "failure" for r in rows),
                "timeouts": sum(r["status"] == "timeout" for r in rows),
                "unknown": unknown,
                "orphan_outcomes": orphans,
                "success_ratio": len(good) / len(rows) if complete else None,
                "duration_seconds": distribution([r["seconds"] for r in good]),
                "payload_bytes": distribution([r["payload_bytes"] for r in good]),
                "mib_per_second": distribution([r["mib_per_second"] for r in good]),
            }
    return {
        "scope": "observed client SDK storage operations; serialization excluded; not network capacity",
        "records": records,
        "by_client": groups,
    }

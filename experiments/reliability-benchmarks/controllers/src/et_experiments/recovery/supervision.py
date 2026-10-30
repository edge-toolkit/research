"""Supporting recovery measurements without changing the first-useful-recovery boundary."""

from __future__ import annotations

from collections import Counter

from et_experiments.utils import BenchmarkError

CLOCKS = {"observer-monotonic-seconds", "observer-elapsed-seconds-with-restart-gaps"}
STABILITY_STATES = {"disabled", "pending", "stable", "unstable", "incomplete", "not-reported"}


def stability(record):
    """Preserve Core's assessment; an older report does not establish that stability was measured."""
    value = record.get("stability", {"status": "not-reported"})
    if not isinstance(value, dict) or value.get("status") not in STABILITY_STATES:
        raise BenchmarkError("Core report has a malformed recovery stability assessment")
    if value["status"] == "stable":
        verified, recovered = value.get("verified_at"), record.get("recovered")
        if (
            type(verified) is not int
            or type(recovered) is not int
            or verified < recovered
            or record.get("status") != "recovered"
        ):
            raise BenchmarkError("Core stability verification does not follow useful recovery")
    return dict(value)


def measurements(incidents, gaps=()):
    """Count verified stability among enabled recovered incidents; missing evidence never earns success."""
    recovered = [item for item in incidents if item.get("status") == "recovered"]
    counts = Counter(item.get("stability", {}).get("status", "not-reported") for item in recovered)
    eligible = sum(counts[key] for key in ("pending", "stable", "unstable", "incomplete"))
    closed = [gap for gap in gaps if type(gap.get("ended_at")) is int]
    return {
        "recovery_stability_eligible_incidents": eligible,
        "recovery_stability_verified_incidents": counts["stable"],
        "recovery_stability_unstable_incidents": counts["unstable"],
        "recovery_stability_pending_incidents": counts["pending"],
        "recovery_stability_incomplete_incidents": counts["incomplete"],
        "recovery_stability_disabled_incidents": counts["disabled"],
        "recovery_stability_unreported_incidents": counts["not-reported"],
        "recovery_stability_verified_ratio": counts["stable"] / eligible if eligible else None,
        "supervision_observation_gaps": len(gaps),
        "supervision_closed_gap_seconds": sum(gap["ended_at"] - gap["started_at"] for gap in closed),
        "supervision_open_gaps": len(gaps) - len(closed),
    }


def verify_continuation(previous, current):
    """Accept replacement only with the same durable context, preserved history and an explicit restart gap."""
    old, new = previous.get("context", {}), current.get("context", {})
    valid = (
        old.get("schema_version") == new.get("schema_version") == 1
        and old.get("configuration") == new.get("configuration")
        and old.get("jitter_seed") is not None
        and old["jitter_seed"] == new.get("jitter_seed")
        and type(new.get("restored_sessions")) is int
        and new["restored_sessions"] > old.get("restored_sessions", 0)
        and new.get("elapsed_seconds", -1) >= old.get("elapsed_seconds", 0)
        and current.get("clock_basis") == "observer-elapsed-seconds-with-restart-gaps"
        and any(gap.get("reason") == "supervisor restart" for gap in current.get("observation_gaps", []))
    )
    old_incidents, new_incidents = previous.get("incidents", []), current.get("incidents", [])
    valid = valid and len(new_incidents) >= len(old_incidents)
    for before, after in zip(old_incidents, new_incidents, strict=False):
        valid = valid and all(before[key] == after.get(key) for key in ("workload", "onset", "baseline"))
        valid = valid and after.get("actions", [])[: len(before["actions"])] == before["actions"]
        valid = valid and all(
            before.get(key) is None or before[key] == after.get(key) for key in ("detected", "recovered")
        )
    prior = previous.get("checkpoint_measurements", [])
    valid = valid and current.get("checkpoint_measurements", [])[: len(prior)] == prior
    for name, count in previous.get("restart_budget", {}).get("restarts", {}).items():
        valid = valid and current.get("restart_budget", {}).get("restarts", {}).get(name, -1) >= count
    if not valid:
        raise BenchmarkError("Core supervisor restarted without verifiable durable context continuity")

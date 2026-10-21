"""Parse diagnostic events, summarize repetitions and evaluate experimental SLOs."""

from __future__ import annotations

import json
import math
import statistics
from typing import Any

EVENT_PREFIX = "ET_EVENT "


def parse_events(text: str, job_id: str | None = None) -> list[dict[str, Any]]:
    """Extract valid diagnostic events while continuing past unrelated or malformed log entries."""
    events = {}
    for line in text.splitlines():
        if EVENT_PREFIX not in line:
            continue
        try:
            event = json.loads(line.split(EVENT_PREFIX, 1)[1])
            if not isinstance(event, dict) or not all(
                isinstance(event.get(key), str) for key in ("id", "instance", "kind", "job_id", "workload")
            ):
                continue
            if event.get("schema_version") != 1 or (job_id and event.get("job_id") != job_id):
                continue
            if not isinstance(event.get("at"), (int, float)) or not math.isfinite(event["at"]):
                continue
            if "progress" in event and (
                not isinstance(event["progress"], (int, float)) or not math.isfinite(event["progress"])
            ):
                continue
            events[event["id"]] = event
        except (ValueError, TypeError, KeyError):
            continue
    return sorted(events.values(), key=lambda item: item["at"])


def evaluate_slos(metrics: dict[str, Any], objectives: dict[str, Any]) -> dict[str, Any]:
    """Evaluate explicit experiment targets; missing evidence never silently passes."""
    results = {}
    for name, rule in objectives.items():
        value = metrics.get(name)
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            status = "not-measured"
        else:
            status = "passed" if rule.get("min", -math.inf) <= value <= rule.get("max", math.inf) else "failed"
        results[name] = {"value": value, "objective": rule, "status": status}
    return results


def checkpoint_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    saves = [event for event in events if event["kind"] == "checkpoint-saved"]
    restores = [event for event in events if event["kind"] == "checkpoint-restored"]
    resumed = [event for event in events if event["kind"] == "resume-completed"]
    attempts: dict[tuple, set] = {}
    for event in events:
        if event["kind"] == "operation-started":
            attempts.setdefault((event["workload"], event["progress"]), set()).add(event["instance"])
    repeated = sum(max(0, len(instances) - 1) for instances in attempts.values())
    return {
        "checkpoint_save_mean_seconds": statistics.fmean(e["seconds"] for e in saves) if saves else None,
        "checkpoint_load_mean_seconds": statistics.fmean(e["seconds"] for e in restores) if restores else None,
        "checkpoint_bytes_max": max((e["bytes"] for e in saves), default=None),
        "resume_mean_seconds": statistics.fmean(e["seconds"] for e in resumed) if resumed else None,
        "observed_repeated_work_units": repeated if attempts else None,
        "observed_resume_efficiency": len(attempts) / (len(attempts) + repeated) if attempts else None,
    }


def summarize_runs(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool incident durations across repetitions, retaining failed and unobserved outcomes."""
    incidents = [
        incident
        for result in results
        for incident in result.get("recovery", {}).get("incidents", [])
        if incident.get("status") not in {"suspected", "transient", "observing"}
    ]
    recovered = [i for i in incidents if i.get("status") == "recovered"]
    durations = [i["recovered_at"] - i["observed_failure_at"] for i in recovered]
    modes = sorted({result.get("attribution", {}).get("recovery_mode", "unspecified") for result in results})
    providers = sorted({result.get("recovery", {}).get("provider", "unspecified") for result in results})
    by_mode = {}
    for mode in modes:
        group = [
            result for result in results if result.get("attribution", {}).get("recovery_mode", "unspecified") == mode
        ]
        records = [
            item
            for result in group
            for item in result.get("recovery", {}).get("incidents", [])
            if item.get("status") not in {"suspected", "transient", "observing"}
        ]
        group_providers = {result.get("recovery", {}).get("provider", "unspecified") for result in group}
        times = [
            item["recovered_at"] - item["observed_failure_at"] for item in records if item["status"] == "recovered"
        ]
        by_mode[mode] = {
            "incidents": len(records),
            "recovered_incidents": len(times),
            "mttr_seconds": statistics.fmean(times) if times and len(group_providers) == 1 else None,
        }
    return {
        "repetitions": len(results),
        "passed_runs": sum(r.get("status") == "passed" for r in results),
        "experiment_error_ratio": sum(r.get("status") != "passed" for r in results) / len(results) if results else None,
        "failed_runs": sum(r.get("status") != "passed" for r in results),
        "incidents": len(incidents),
        "recovered_incidents": len(recovered),
        "unrecovered_incidents": len(incidents) - len(recovered),
        "mttr_seconds": statistics.fmean(durations) if durations and len(modes) == 1 and len(providers) == 1 else None,
        "recovery_success_ratio": len(recovered) / len(incidents) if incidents else None,
        "mttr_basis": "observer-first-evidence to verified useful progress; recovered incidents only",
        "recovery_modes": modes,
        "mixed_recovery_modes": len(modes) > 1,
        "recovery_providers": providers,
        "mixed_recovery_providers": len(providers) > 1,
        "by_recovery_mode": by_mode,
        "attribution_scope": "combined-deployed-system",
        "recovery_actors": sorted(
            {actor for result in results for actor in result.get("attribution", {}).get("recovery_actors", [])}
        ),
        "intervention_attempts": sum(len(result.get("attribution", {}).get("interventions", [])) for result in results),
    }

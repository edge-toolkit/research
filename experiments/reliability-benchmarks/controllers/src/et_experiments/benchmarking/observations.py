"""Workload-neutral completion and fault-recovery observation."""

from __future__ import annotations

import datetime as dt
from typing import Any


def work_finished(workload: dict[str, Any], recovery_enabled: bool) -> bool:
    """A failed application is terminal only when recovery is disabled."""
    if workload["failed"]:
        return not recovery_enabled
    return workload["completed_units"] >= workload["expected_units"]


def fault_incidents(report: dict[str, Any], fault: dict[str, Any]) -> list[dict[str, Any]]:
    """Select new target incidents without crediting pre-injection history."""
    targets = {child.get("target") for child in fault.get("faults", [])} or {fault.get("target")}
    return [
        item
        for item in report["incidents"]
        if item["workload"] in targets
        and item.get("core_incident_index", fault.get("core_incident_baseline", 0))
        >= fault.get("core_incident_baseline", 0)
    ]


def stability_outcome(report, scenario, fault):
    """Assess the separate stability requirement without moving the first-useful-recovery timestamp."""
    if not scenario.get("require_recovery_stability"):
        return "not-requested"
    incidents = fault_incidents(report, fault)
    if not incidents or any(item["status"] != "recovered" for item in incidents):
        return "pending"
    states = {item.get("stability", {}).get("status", "not-reported") for item in incidents}
    if "unstable" in states:
        return "unstable"
    if states.intersection({"incomplete", "disabled", "not-reported"}):
        return "incomplete"
    return "stable" if states == {"stable"} else "pending"


def recovery_outcome(report: dict[str, Any], scenario: dict[str, Any], fault: dict[str, Any]) -> str:
    """Require fault-target recovery evidence independently of workload completion."""
    incidents = fault_incidents(report, fault)
    if any(item["status"] == "unrecovered" for item in incidents):
        return "unrecovered"
    if (
        report.get("provider") == "edge-toolkit-core"
        and incidents
        and all(item["status"] == "transient" for item in incidents)
    ):
        return "not-detected"
    if not incidents or any(item["status"] != "recovered" for item in incidents):
        return "pending"
    if fault.get("type") == "combined" and not {c["target"] for c in fault["faults"]}.issubset(
        {item["workload"] for item in incidents}
    ):
        return "pending"
    if scenario.get("require_checkpoint_restore"):
        if report.get("provider") == "edge-toolkit-core":
            if not any(
                item.get("evidence", {}).get("checkpoint_restored_at") is not None
                and item.get("evidence", {}).get("resume_completed_at") is not None
                for item in incidents
            ):
                return "pending"
            return "recovered"
        started = dt.datetime.fromisoformat(fault["started_at"].replace("Z", "+00:00")).timestamp()
        kinds = {
            event["kind"]
            for event in report.get("events", [])
            if event["workload"] == fault["target"] and event["at"] >= started
        }
        if not {"checkpoint-restored", "resume-completed"}.issubset(kinds):
            return "pending"
    return "recovered"

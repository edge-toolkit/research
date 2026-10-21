"""Attribute observed recovery evidence to recorded platform and benchmark actions."""

from __future__ import annotations

import copy
from typing import Any

from et_experiments.utils import BenchmarkError


def recovery_mode(scenario: dict[str, Any]) -> str:
    mode = scenario.get("recovery_mode", "observe-only" if scenario.get("recovery_enabled", True) else "disabled")
    if mode not in {"observe-only", "controller-assisted", "disabled"}:
        raise BenchmarkError(f"unsupported recovery mode: {mode}")
    if not scenario.get("recovery_enabled", True) and mode != "disabled":
        raise BenchmarkError("recovery_enabled=false requires recovery_mode=disabled")
    if scenario.get("require_recovery") and mode == "disabled":
        raise BenchmarkError("required recovery cannot be measured with recovery_mode=disabled")
    if scenario.get("fault", {}).get("restart_bridge") and mode != "controller-assisted":
        raise BenchmarkError("forced bridge restart requires controller-assisted recovery")
    return mode


def describe_recovery(report: dict[str, Any], mode: str, fault: dict[str, Any], events=()) -> dict[str, Any]:
    result = copy.deepcopy(report)
    result["mode"] = mode
    interventions = list(fault.get("interventions", []))
    levels = set()
    actors = set()
    for incident in result["incidents"]:
        evidence = incident.setdefault("evidence", {})
        onset = incident["observed_failure_at"]
        end = incident.get("recovered_at") or float("inf")
        for field, kinds in {
            "connectivity_at": {"connected", "reconnected"},
            "checkpoint_restored_at": {"checkpoint-restored"},
            "resume_completed_at": {"resume-completed"},
        }.items():
            if result.get("provider") == "edge-toolkit-core" and field != "connectivity_at":
                # Core checkpoint evidence is already paired by process and incident during normalization.
                continue
            matching = [
                e["at"]
                for e in events
                if e["workload"] == incident["workload"]
                and e["kind"] in kinds
                and e.get("clock_basis", "controller-wall-clock")
                == incident.get("clock_basis", "controller-wall-clock")
                and onset <= e["at"] <= end
            ]
            if matching:
                evidence.setdefault(field, min(matching))
        recovered = incident["status"] == "recovered"
        evidence.update(
            {
                "pod_readiness_verified": incident.get("ready_at") is not None,
                "connectivity_verified": evidence.get("connectivity_at") is not None,
                "application_progress_verified": recovered,
                "checkpoint_restoration_verified": evidence.get("checkpoint_restored_at") is not None
                and evidence.get("resume_completed_at") is not None,
            }
        )
        for level, field in {
            "pod-readiness": "pod_readiness_verified",
            "connectivity": "connectivity_verified",
            "application-progress": "application_progress_verified",
            "checkpoint-restoration": "checkpoint_restoration_verified",
        }.items():
            if evidence[field]:
                levels.add(level)
        actions = incident.get("actions", [])
        interventions.extend({**a, "workload": incident["workload"], "purpose": "recovery-assistance"} for a in actions)
        actor = incident.get("recovery_actor", "unattributed")
        if recovered and any(a["type"] == "restart" for a in actions):
            actor = "edge-toolkit-core" if result.get("provider") == "edge-toolkit-core" else "research-controller"
        if (
            actor == "kubernetes"
            and incident["workload"] == fault.get("target")
            and fault.get("type") in {"rollout_restart", "temporary_scale_down", "worker_oom"}
        ):
            actor = "benchmark-requested-kubernetes-recovery"
        incident["recovery_actor"] = actor
        if recovered:
            actors.add(actor)
    result["attribution"] = {
        "scope": "combined-deployed-system",
        "recovery_provider": result.get("provider", "unspecified"),
        "recovery_clock_basis": result.get("clock_basis", "unspecified"),
        "components": ["edge-toolkit", "kubernetes", "application"],
        "recovery_mode": mode,
        "fault_actor": "benchmark" if fault.get("type") != "none" and fault.get("status") != "pending" else None,
        "fault_injection_status": fault.get("status"),
        "fault_removal_actor": "benchmark" if fault.get("status") == "restored" else None,
        "recovery_actors": sorted(actors),
        "evidence_levels": sorted(levels),
        "interventions": interventions,
        "intervention_attempted": bool(interventions),
        "intervention_occurred": any(
            a.get("type") in {"restart", "bridge-restart"} and a.get("status", "requested") == "requested"
            for a in interventions
        ),
        "core_only_recovery_claim": False,
        "basis": "Recorded commands and observed recovery evidence; not proof of exclusive causal responsibility.",
    }
    return result

"""Assess useful service continuity, participant return and checkpoint progress metadata."""

from __future__ import annotations

import datetime as dt


def wall_events(events):
    """Select events on the controller wall clock, keeping supervisor-relative times out of fault-time arithmetic."""
    return [e for e in events if e.get("clock_basis", "controller-wall-clock") == "controller-wall-clock"]


def processing_metrics(events, primary, expected):
    """Measure the full useful-work interval from explicit initial and final progress evidence."""
    records = [e for e in wall_events(events) if e["workload"] == primary]
    starts = [e["at"] for e in records if e["kind"] == "operation-started" and e.get("progress") == 0]
    finishes = [e["at"] for e in records if e["kind"] == "progress" and e.get("progress", -1) >= expected]
    seconds = min(finishes) - min(starts) if starts and finishes else None
    if seconds is not None and seconds <= 0:
        seconds = None
    return {
        "processing_time_seconds": seconds,
        "processing_units_per_minute": expected / seconds * 60 if seconds else None,
    }


def recovery_evidence(report, primary):
    """Assess committed work during the fault and fresh participant contributions after its completion.

    Pair worker and primary evidence by progress and process identity; checkpoint evidence is assessed separately.
    """
    fault = report.get("fault", {})
    start = fault.get("started_at")
    if not start or fault.get("type") == "none":
        return {"status": "not-applicable"}
    onset = dt.datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
    finish = dt.datetime.fromisoformat(fault.get("finished_at", start).replace("Z", "+00:00")).timestamp()
    events = wall_events(report.get("diagnostic_events", []))
    progress = [e for e in events if e["kind"] == "progress" and e["workload"] == primary and e["at"] >= onset]
    active_start = (
        dt.datetime.fromisoformat(fault["active_at"].replace("Z", "+00:00")).timestamp()
        if fault.get("active_at")
        else onset
    )
    active_finish = (
        dt.datetime.fromisoformat(fault["released_at"].replace("Z", "+00:00")).timestamp()
        if fault.get("released_at")
        else finish
    )
    during = [e for e in progress if active_start <= e["at"] <= active_finish]
    targets = [c.get("target") for c in fault.get("faults", [])] or [fault.get("target")]
    participants = {}
    for target in targets:
        accepted = [
            e
            for e in events
            if e["kind"] == "contribution-accepted"
            and e["at"] >= finish
            and e.get("participant_workload", e.get("participant")) == target
        ]
        ready = [
            e for e in events if e["kind"] == "contribution-ready" and e["workload"] == target and e["at"] >= finish
        ]
        accepted = [
            e
            for e in accepted
            if any(
                sent.get("progress") is not None
                and sent.get("progress") == e.get("progress")
                and sent["instance"] == e.get("participant_instance")
                and sent["at"] <= e["at"]
                for sent in ready
            )
        ]
        participants[target or "unspecified"] = {
            "status": "verified" if accepted else "not-observed",
            "first_accepted_at": min((e["at"] for e in accepted), default=None),
            "after_fault_seconds": min((e["at"] - onset for e in accepted), default=None),
        }
    return {
        "service_continuity": {
            "status": "observed" if during else "not-observed",
            "progress_events": len(during),
            "basis": "committed primary progress between confirmed fault apply/remove commands"
            if fault.get("active_at") and fault.get("released_at")
            else "committed primary progress during the fault-action interval",
        },
        "participant_recovery": participants,
        "checkpoint": checkpoint_evidence(events, primary, onset, report.get("workload", {}).get("expected_units")),
    }


def checkpoint_evidence(events, primary, onset, expected):
    """Pair restoration and resume in the same process using explicit progress metadata."""
    events = sorted((e for e in events if e["workload"] == primary), key=lambda e: e["at"])
    saves = [e for e in events if e["kind"] == "checkpoint-saved" and e["at"] <= onset]
    restores = [e for e in events if e["kind"] == "checkpoint-restored" and e["at"] >= onset]
    saved = saves[-1] if saves else None
    for restored in restores:
        if type(restored.get("progress")) not in (int, float):
            continue
        resumes = [
            e
            for e in events
            if e["kind"] == "resume-completed"
            and e["instance"] == restored["instance"]
            and e["at"] >= restored["at"]
            and e.get("progress") == restored.get("progress")
        ]
        if not resumes:
            continue
        value = restored.get("progress")
        before = [e.get("progress", 0) for e in events if e["kind"] == "progress" and e["at"] <= onset]
        later = [
            e
            for e in events
            if e["kind"] == "progress"
            and e["instance"] == restored["instance"]
            and e["at"] >= resumes[0]["at"]
            and e.get("progress", -1) > value
        ]
        return {
            "status": "verified-progress-metadata" if saved and saved.get("progress") == value else "unmatched-save",
            "checkpoint_age_at_fault_seconds": onset - saved["at"] if saved else None,
            "saved_progress": saved.get("progress") if saved else None,
            "restored_progress": value,
            "unsaved_completed_units": max(0, max(before, default=0) - value),
            "checkpoint_bytes": saved.get("bytes") if saved else None,
            "resumed_unfinished_work": bool(later and expected is not None and value < expected),
            "first_progress_after_resume_seconds": later[0]["at"] - resumes[0]["at"] if later else None,
            "basis": "application progress metadata; payload integrity is verified by the application's checkpoint SDK",
        }
    return {"status": "not-observed"}

"""Harbour-FL progress, tracking reconciliation and application measurements."""

from __future__ import annotations

import datetime as dt
import statistics
import time
from typing import Any

from et_experiments.utils import iso, percentile


def training_from_mlflow(state: dict[str, Any], rounds_expected: int, started: dt.datetime) -> dict[str, Any]:
    """Reconstruct round progress and timing distributions from step-indexed MLflow metrics."""
    by_step: dict[int, dict[str, Any]] = {}
    for key, values in state.get("history", {}).items():
        for metric in values:
            step = int(metric.get("step", 0))
            record = by_step.setdefault(step, {"round": step, "metrics": {}, "timestamp_ms": 0})
            record["metrics"][key] = float(metric["value"])
            record["timestamp_ms"] = max(record["timestamp_ms"], int(metric.get("timestamp", 0)))
    rounds = []
    previous = (
        dt.datetime.fromtimestamp(int(state["start_time"]) / 1000, tz=dt.timezone.utc)
        if state.get("start_time")
        else started
    )
    durations = []
    for step in sorted(by_step):
        record = by_step[step]
        completed = dt.datetime.fromtimestamp(record.pop("timestamp_ms") / 1000, tz=dt.timezone.utc)
        duration = record["metrics"].get("round_work_seconds", max(0.0, (completed - previous).total_seconds()))
        record.update({"completed_at": iso(completed), "duration_seconds": duration})
        durations.append(duration)
        rounds.append(record)
        previous = completed
    return {
        "completed_rounds": len(rounds),
        "expected_rounds": rounds_expected,
        "coordinator_failed": state.get("status") in {"FAILED", "KILLED"},
        "rounds": rounds,
        "round_duration_seconds": {
            "mean": statistics.fmean(durations) if durations else None,
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "max": max(durations) if durations else None,
        },
        "client_rounds": {},
        "checkpoint_write_seconds": [],
        "final_model_metrics": rounds[-1]["metrics"] if rounds else {},
    }


def observe_training(
    logged: dict[str, Any], state: dict[str, Any] | None, expected_rounds: int, started: dt.datetime
) -> dict[str, Any]:
    """Combine asynchronous tracking metrics with authoritative workload failure evidence."""
    tracked = training_from_mlflow(state, expected_rounds, started) if state else None
    if tracked and tracked["rounds"]:
        result = tracked
        by_round = {item["round"]: item for item in logged.get("rounds", [])}
        for item in result["rounds"]:
            source = by_round.pop(item["round"], {}).get("metrics", {})
            for key, value in source.items():
                item["metrics"].setdefault(key, value)
            if "round_work_seconds" in item["metrics"]:
                item["duration_seconds"] = item["metrics"]["round_work_seconds"]
        result["rounds"].extend(by_round.values())
        result["rounds"].sort(key=lambda item: item["round"])
        result["completed_rounds"] = len(result["rounds"])
        result["final_model_metrics"] = result["rounds"][-1]["metrics"]
        durations = [
            item["duration_seconds"]
            for item in result["rounds"]
            if item.get("duration_seconds") is not None and item["duration_seconds"] >= 0
        ]
        result["round_duration_seconds"] = {
            "mean": statistics.fmean(durations) if durations else None,
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "max": max(durations) if durations else None,
        }
    else:
        result = dict(logged)
    # Tracking can lag both a failure and a subsequent successful restart.
    result["coordinator_failed"] = bool(logged.get("coordinator_failed")) or (
        bool(tracked and tracked["coordinator_failed"]) and not logged.get("coordinator_recovered", False)
    )
    result["client_rounds"] = logged.get("client_rounds", {})
    result["checkpoint_write_seconds"] = logged.get("checkpoint_write_seconds", [])
    return result


def training_finished(training: dict[str, Any], state: dict[str, Any] | None, recovery_enabled: bool) -> bool:
    """Decide whether completion or a terminal failure ends observation under the selected recovery mode."""
    if training["coordinator_failed"]:
        return not recovery_enabled
    return training["completed_rounds"] >= training["expected_rounds"] or bool(
        state and state.get("status") == "FINISHED"
    )


def tracking_recovery(state, rounds: int, loss_key: str, fault: dict[str, Any], verified_at=None):
    """Verify post-fault ingestion, not merely the MLflow service's readiness."""
    values = (state or {}).get("history", {}).get(loss_key, [])
    steps = {int(item["step"]) for item in values}
    started = dt.datetime.fromisoformat(fault["started_at"].replace("Z", "+00:00")).timestamp()
    post_fault = any(float(item.get("timestamp", 0)) / 1000 >= started for item in values)
    verified = bool(state and state.get("status") == "FINISHED" and set(range(rounds)) <= steps and post_fault)
    at = verified_at if verified_at is not None else time.time() if verified else None
    assisted = any(item.get("type") == "bridge-restart" for item in fault.get("interventions", []))
    return {
        "status": "verified" if verified else "pending",
        "verified_at": at,
        "recovery_seconds": at - started if at is not None else None,
        "component": "core-mlflow-bridge",
        "forced_bridge_restart": assisted,
        "recovery_actor": ("benchmark-assisted-bridge" if assisted else "core-mlflow-bridge")
        if verified
        else "unattributed",
        "run_finished": bool(state and state.get("status") == "FINISHED"),
        "all_round_metrics_present": set(range(rounds)) <= steps,
        "post_fault_ingestion_verified": post_fault,
        "basis": "Post-fault training metric ingestion and run completion; not proof that every artifact was replayed.",
    }


def application_metrics(report):
    """Summarise FL participation, input I/O and checkpoint costs for tracking."""
    training = report["training"]
    resilience = report["resilience"]
    fault = report["fault"]
    durations = training.get("round_duration_seconds", {})
    client_metrics = [
        item.get("metrics", {})
        for rounds in report.get("training", {}).get("client_rounds", {}).values()
        for item in rounds
    ]
    checkpoint_writes = report.get("training", {}).get("checkpoint_write_seconds", [])

    def metric_values(key: str) -> list[float]:
        """Collect numeric observations of one client metric across the reported rounds."""
        return [float(item[key]) for item in client_metrics if isinstance(item.get(key), (int, float))]

    read_means = metric_values("dataset_read_mean_ms")
    read_p95s = metric_values("dataset_read_p95_ms")
    sample_rates = metric_values("samples_per_second")
    return {
        "dataset_read_mean_ms": statistics.fmean(read_means) if read_means else None,
        "dataset_read_p95_ms": max(read_p95s) if read_p95s else None,
        "samples_per_second": statistics.fmean(sample_rates) if sample_rates else None,
        "checkpoint_write_mean_seconds": statistics.fmean(checkpoint_writes) if checkpoint_writes else None,
        "checkpoint_write_p95_seconds": percentile(checkpoint_writes, 0.95),
        "checkpoint_write_failures": 1
        if fault.get("type") == "checkpoint_write_failure" and report.get("training", {}).get("coordinator_failed")
        else 0,
        "round_completion_ratio": resilience.get("round_completion_ratio"),
        "client_participation_ratio": resilience.get("mean_client_participation_ratio"),
        "completed_rounds_per_minute": resilience.get("completed_rounds_per_minute"),
        "round_duration_mean_seconds": durations.get("mean"),
        "round_duration_p50_seconds": durations.get("p50"),
        "round_duration_p95_seconds": durations.get("p95"),
        "round_duration_max_seconds": durations.get("max"),
        **{
            f"client_{key}_mean": statistics.fmean(values) if (values := metric_values(key)) else None
            for key in (
                "training_seconds",
                "processed_samples",
                "optimizer_steps",
                "update_bytes",
                "global_download_seconds",
                "update_serialization_seconds",
                "update_upload_seconds",
            )
        },
    }


def normalise_progress(training):
    """Translate the application's rounds into the engine's work-unit contract."""
    return {
        "completed_units": training["completed_rounds"],
        "expected_units": training["expected_rounds"],
        "failed": training["coordinator_failed"],
        "units": training["rounds"],
        "unit_duration_seconds": training["round_duration_seconds"],
        "final_metrics": training["final_model_metrics"],
        "details": training,
    }


def enrich_report(report, metric_aliases, worker_count):
    """Retain FL report compatibility outside the generic engine and supply domain SLIs."""
    training = report["workload"]["details"]
    report["training"] = training
    participation = [
        float(item["metrics"].get(metric_aliases["participation"], 0)) / max(worker_count, 1)
        for item in training["rounds"]
        if isinstance(item.get("metrics"), dict)
    ]
    report["resilience"].update(
        {
            "round_completion_ratio": report["resilience"]["completion_ratio"],
            "mean_client_participation_ratio": statistics.fmean(participation) if participation else None,
            "completed_rounds_per_minute": report["resilience"]["completed_units_per_minute"],
            "rejected_updates": sum(event["kind"] == "update-rejected" for event in report["events"]),
            "degraded_rounds": sum(event["kind"] == "degraded-participation" for event in report["events"]),
        }
    )
    report["application_metrics"] = application_metrics(report)
    report["application_tags"] = {
        f"fault_{key}": report["fault"][key]
        for key in ("round", "snr_db", "noise_std", "modality")
        if report["fault"].get(key) is not None
    }
    final = training["final_model_metrics"]
    return {
        "round_p95_seconds": training["round_duration_seconds"]["p95"],
        "model_accuracy": final.get(metric_aliases["accuracy"]),
    }

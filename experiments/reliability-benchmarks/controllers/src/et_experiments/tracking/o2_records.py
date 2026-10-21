"""Translate application-independent benchmark evidence into queryable O2 records."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any

from et_experiments.serialization import finite_number as numeric


def records(report: dict[str, Any], context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Represent metric names as row values using a stable O2 column schema."""
    context = context or {}
    attribution = report.get("attribution", {})
    recovery = report.get("recovery", {})
    timestamp = report.get("finished_at") or dt.datetime.now(dt.UTC).isoformat()
    base = {
        "schema_version": 1,
        "application": context.get("application", report.get("application", "unspecified")),
        "controller_run_id": context.get("controller_run_id", "standalone"),
        "repetition": context.get("repetition", 1),
        "job_id": report.get("job_id") or context.get("job_id", "unspecified"),
        "scenario": report.get("scenario") or context.get("scenario", "unspecified"),
        "namespace": report.get("namespace", "unspecified"),
        "recovery_mode": attribution.get("recovery_mode", "unspecified"),
        "recovery_provider": recovery.get("provider", "unspecified"),
    }
    result = []

    def emit(kind, values, at=None):
        """Append a timestamped record with a stable content identity and JSON copy of optional fields."""
        record = {**base, "record_type": kind, **values}
        identity = json.dumps(record, sort_keys=True, allow_nan=False)
        record["record_id"] = hashlib.sha256(identity.encode()).hexdigest()
        # O2 omits all-null columns from inferred schemas. A stable JSON string
        # lets dashboards extract missing optional fields as NULL.
        record["record_data"] = json.dumps(values, sort_keys=True, allow_nan=False)
        record["_timestamp"] = at or timestamp
        result.append(record)

    def measurements(kind, values, **labels):
        """Emit typed numeric measurements while leaving absent or non-finite values unexported."""
        for name, value in values.items():
            if numeric(value):
                emit(kind, {**labels, "metric_name": name, "metric_value": float(value)})

    emit(
        "run",
        {
            "status": report.get("status", "unknown"),
            "started_at": report.get("started_at"),
            "finished_at": report.get("finished_at"),
            "fault_type": report.get("fault", {}).get("type", "none"),
            "fault_target": report.get("fault", {}).get("target"),
            "fault_status": report.get("fault", {}).get("status"),
            "mlflow_run_id": report.get("mlflow_run_id"),
            "intervention_occurred": attribution.get("intervention_occurred"),
            "recovery_actors": ",".join(attribution.get("recovery_actors", [])),
            "evidence_levels": ",".join(attribution.get("evidence_levels", [])),
            "wall_time_seconds": report.get("wall_time_seconds"),
            "resource_metrics_available": report.get("resource_metrics_available"),
        },
    )
    measurements(
        "metric",
        {key: report.get(key) for key in ("wall_time_seconds", "total_pod_restarts", "total_pod_replacements")},
        metric_group="run",
    )
    for group in ("resilience", "sli", "application_metrics"):
        measurements("metric", report.get(group, {}), metric_group=group)
    workload = report.get("workload", {})
    measurements("metric", workload.get("unit_duration_seconds", {}), metric_group="work_unit_duration_seconds")
    measurements("metric", workload.get("final_metrics", {}), metric_group="model")
    measurements(
        "metric",
        {key: workload.get(key) for key in ("completed_units", "expected_units")},
        metric_group="workload",
    )
    measurements("metric", report.get("tracking_recovery", {}), metric_group="tracking_recovery")
    for index, unit in enumerate(workload.get("units", [])):
        measurements("unit_metric", unit.get("metrics", {}), unit_index=index)
        measurements("unit_metric", {"duration_seconds": unit.get("duration_seconds")}, unit_index=index)
    for name, assessment in report.get("slos", {}).items():
        emit(
            "slo",
            {
                "metric_name": name,
                "metric_value": assessment.get("value") if numeric(assessment.get("value")) else None,
                "status": assessment.get("status", "not-measured"),
                "objective_min": assessment.get("objective", {}).get("min"),
                "objective_max": assessment.get("objective", {}).get("max"),
            },
        )
    clock = recovery.get("clock_basis", "unspecified")
    for index, incident in enumerate(recovery.get("incidents", [])):
        onset, detected, recovered = (
            incident.get(key) for key in ("observed_failure_at", "detected_at", "recovered_at")
        )
        emit(
            "incident",
            {
                "incident_index": index,
                "workload": incident.get("workload"),
                "status": incident.get("status"),
                "clock_basis": incident.get("clock_basis", clock),
                "observer_onset_seconds": onset,
                "observer_detected_seconds": detected,
                "observer_recovered_seconds": recovered,
                "failure_detection_seconds": detected - onset if numeric(detected) and numeric(onset) else None,
                "recovery_seconds": recovered - onset
                if incident.get("status") == "recovered" and numeric(recovered) and numeric(onset)
                else None,
                "checkpoint_verified": bool(incident.get("evidence", {}).get("checkpoint_restored_at") is not None),
                "recovery_actor": incident.get("recovery_actor", "unattributed"),
            },
        )
        for action_index, action in enumerate(incident.get("actions", [])):
            emit(
                "action",
                {
                    "incident_index": index,
                    "action_index": action_index,
                    "workload": incident.get("workload"),
                    "event_kind": action.get("kind", action.get("type")),
                    "status": action.get("status"),
                    "actor": action.get("actor", "unspecified"),
                    "observer_offset_seconds": action.get("at"),
                    "clock_basis": clock,
                },
            )
    # Observer offsets are not wall-clock timestamps. Do not fabricate an epoch
    # by adding offsets to benchmark start: Core may have started later.
    for event in report.get("events", []):
        emit(
            "event",
            {
                "event_id": event.get("id"),
                "event_kind": event.get("kind"),
                "workload": event.get("workload"),
                "instance": event.get("instance"),
                "clock_basis": event.get("clock_basis", "application-defined"),
                "observer_offset_seconds": event.get("at"),
                **{key: event.get(key) for key in ("progress", "bytes", "seconds")},
            },
        )
    fault = report.get("fault", {})
    if fault.get("started_at"):
        emit("fault", {"event_kind": "fault-injected", "fault_type": fault.get("type")}, fault["started_at"])
    for sample_index, sample in enumerate(report.get("samples", [])):
        for app, pod in sample.get("pods", {}).items():
            emit(
                "resource_sample",
                {
                    "sample_index": sample_index,
                    "workload": app,
                    "pod": pod.get("name"),
                    "node": pod.get("node"),
                    "ready": pod.get("ready"),
                    "phase": pod.get("phase"),
                    **{
                        key: pod.get(key)
                        for key in ("restarts", "cpu_millicores", "memory_mib", "network_rx_bytes", "network_tx_bytes")
                    },
                },
                sample.get("at"),
            )
    for app, values in report.get("resources", {}).items():
        measurements("resource_metric", values, workload=app)
    # Do not ship raw application logs, dataset contents or arbitrary error text.
    # Kubernetes warning reasons are enough to group infrastructure symptoms.
    for index, warning in enumerate(report.get("warning_events", [])):
        emit(
            "warning",
            {"warning_index": index, "reason": warning.get("reason"), "object": warning.get("object")},
            warning.get("at"),
        )
    return result

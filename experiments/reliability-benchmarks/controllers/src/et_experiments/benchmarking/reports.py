"""Persist experiment reports independently of tracking availability."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


def write_reports(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    stem = report["job_id"]
    (output / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_path = os.environ.get("BENCHMARK_RESULT_FILE")
    if result_path:
        public = {
            key: report[key]
            for key in ("job_id", "status", "resilience", "recovery", "slos", "attribution", "tracking_recovery", "o2")
            if key in report
        }
        public["report"] = str(output / f"{stem}.json")
        Path(result_path).write_text(json.dumps(public, indent=2) + "\n")
    summary = output / "summary.csv"
    row = {
        "job_id": stem,
        "scenario": report["scenario"],
        "status": report["status"],
        "fault_type": report["fault"]["type"],
        "wall_time_seconds": report["wall_time_seconds"],
        "completed_units": report["workload"]["completed_units"],
        "work_unit_p50_seconds": report["workload"]["unit_duration_seconds"]["p50"],
        "work_unit_p95_seconds": report["workload"]["unit_duration_seconds"]["p95"],
        "pod_restarts": report["total_pod_restarts"],
        "pod_replacements": report["total_pod_replacements"],
        "recovery_seconds": report["resilience"]["recovery_seconds"],
        "completion_ratio": report["resilience"]["completion_ratio"],
        "warning_events": len(report["warning_events"]),
        "mlflow_run_id": report.get("mlflow_run_id"),
    }
    if summary.exists():
        with summary.open(newline="", encoding="utf-8") as source:
            if next(csv.reader(source), []) != list(row):
                summary = output / "workload-summary.csv"
    exists = summary.exists()
    with summary.open("a", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    attribution = report.get("attribution", {})
    recovery_row = {
        "job_id": stem,
        "scenario": report["scenario"],
        "status": report["status"],
        "recovery_mode": attribution.get("recovery_mode"),
        "scope": attribution.get("scope"),
        "recovery_actors": ",".join(attribution.get("recovery_actors", [])),
        "evidence_levels": ",".join(attribution.get("evidence_levels", [])),
        "intervention_occurred": attribution.get("intervention_occurred"),
        "intervention_attempts": len(attribution.get("interventions", [])),
        "tracking_recovery_status": report.get("tracking_recovery", {}).get("status"),
        "tracking_recovery_seconds": report.get("tracking_recovery", {}).get("recovery_seconds"),
    }
    recovery_summary = output / "recovery-summary.csv"
    exists = recovery_summary.exists()
    with recovery_summary.open("a", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(recovery_row))
        if not exists:
            writer.writeheader()
        writer.writerow(recovery_row)


def write_logs(output: Path, job_id: str, logs: dict[str, str]) -> None:
    directory = output / "logs" / job_id
    directory.mkdir(parents=True, exist_ok=True)
    for deployment, content in logs.items():
        (directory / f"{deployment}.log").write_text(content, encoding="utf-8")

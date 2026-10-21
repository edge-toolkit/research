"""Generate an importable OpenObserve dashboard for benchmark evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def dashboard(stream: str = "reliability_benchmarks") -> dict:
    """Use native v8 panels, with replay-safe record grouping in every query."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", stream):
        raise ValueError("stream must be an O2 identifier")

    def source(kind, columns):
        """Build a replay-deduplicated record source, decoding optional JSON fields and casting numeric values."""
        stable = {"_timestamp", "application", "scenario", "job_id", "recovery_mode", "recovery_provider"}
        numbers = {
            "metric_value",
            "objective_min",
            "objective_max",
            "wall_time_seconds",
            "failure_detection_seconds",
            "recovery_seconds",
            "observer_onset_seconds",
            "observer_detected_seconds",
            "observer_recovered_seconds",
            "observer_offset_seconds",
            "incident_index",
            "unit_index",
            "restarts",
            "cpu_millicores",
            "memory_mib",
            "progress",
            "bytes",
            "seconds",
        }
        expressions = []
        for name in dict.fromkeys(columns):
            expression = f'"{name}"' if name in stable else f"NULLIF(spath(record_data, '{name}'), 'null')"
            if name in numbers:
                expression = f"TRY_CAST({expression} AS DOUBLE)"
            expressions.append(f'max({expression}) AS "{name}"')
        selected = ", ".join(expressions)
        return f"(SELECT record_id, {selected} FROM \"{stream}\" WHERE record_type = '{kind}' GROUP BY record_id)"

    def table(title, kind, columns):
        return title, "table", f"SELECT {', '.join(columns)} FROM {source(kind, columns)}", []

    def metric_table(title, kind, group=None):
        """Build a metric table with its workload or work-unit identity and optional metric-group filter."""
        columns = ["application", "scenario", "job_id", "recovery_mode", "metric_name", "metric_value"]
        if kind == "resource_metric":
            columns.insert(4, "workload")
        if kind == "unit_metric":
            columns.insert(4, "unit_index")
        if kind == "metric":
            columns.insert(4, "metric_group")
        title, kind, query, fields = table(title, kind, columns)
        if group:
            query += f" WHERE metric_group = '{group}'"
        return title, kind, query, fields

    def duration_chart(title, column):
        """Compare mean Core incident durations by application, scenario and recovery mode."""
        columns = ["application", "scenario", "recovery_mode", "recovery_provider", column]
        query = (
            f"SELECT application || ' / ' || scenario AS x_axis_1, "
            f"recovery_mode AS breakdown_1, avg({column}) AS y_axis_1 "
            f"FROM {source('incident', columns)} WHERE {column} IS NOT NULL "
            "AND recovery_provider = 'edge-toolkit-core' GROUP BY x_axis_1, breakdown_1"
        )
        return title, "bar", query, ["application / scenario", "seconds", "recovery_mode"]

    def metric_chart(title, group, metric):
        """Compare one measured metric across applications and scenarios, separated by recovery mode."""
        columns = ["application", "scenario", "recovery_mode", "metric_group", "metric_name", "metric_value"]
        query = (
            "SELECT application || ' / ' || scenario AS x_axis_1, recovery_mode AS breakdown_1, "
            f"avg(metric_value) AS y_axis_1 FROM {source('metric', columns)} "
            f"WHERE metric_group = '{group}' AND metric_name = '{metric}' "
            "GROUP BY x_axis_1, breakdown_1"
        )
        return title, "bar", query, ["application / scenario", metric, "recovery_mode"]

    def resource_chart(title, column):
        """Plot observed resource values over time for each workload, retaining only measured values."""
        query = (
            "SELECT histogram(_timestamp) AS x_axis_1, workload AS breakdown_1, "
            f"avg({column}) AS y_axis_1 FROM {source('resource_sample', ['_timestamp', 'workload', column])} "
            f"WHERE {column} IS NOT NULL GROUP BY x_axis_1, breakdown_1 ORDER BY x_axis_1"
        )
        return title, "line", query, ["time", column, "workload"]

    definitions = {
        "Runs and SLOs": [
            (
                "Run outcomes by application and scenario",
                "bar",
                (
                    "SELECT application || ' / ' || scenario AS x_axis_1, status AS breakdown_1, "
                    f"count(DISTINCT job_id) AS y_axis_1 "
                    f"FROM {source('run', ['application', 'scenario', 'status', 'job_id'])} "
                    "GROUP BY x_axis_1, breakdown_1"
                ),
                ["application / scenario", "runs", "status"],
            ),
            table(
                "Run outcomes and recovery attribution",
                "run",
                [
                    "application",
                    "scenario",
                    "job_id",
                    "status",
                    "recovery_mode",
                    "fault_type",
                    "fault_target",
                    "wall_time_seconds",
                    "recovery_actors",
                    "evidence_levels",
                    "mlflow_run_id",
                ],
            ),
            table(
                "SLO assessment (missing evidence is not a pass)",
                "slo",
                ["job_id", "scenario", "metric_name", "metric_value", "objective_min", "objective_max", "status"],
            ),
            metric_table("Completion and throughput", "metric", "workload"),
            metric_chart("Mean completed work units per minute", "resilience", "completed_units_per_minute"),
            metric_chart("Mean per-run p95 work-unit duration (seconds)", "work_unit_duration_seconds", "p95"),
            metric_table("Measured SLIs", "metric", "sli"),
            metric_table("Work-unit duration percentiles (seconds)", "metric", "work_unit_duration_seconds"),
        ],
        "Recovery": [
            duration_chart("Mean detection latency by scenario and mode", "failure_detection_seconds"),
            duration_chart("Mean recovered-incident duration by scenario and mode", "recovery_seconds"),
            table(
                "Incidents, including failures without recovery",
                "incident",
                [
                    "job_id",
                    "workload",
                    "status",
                    "recovery_mode",
                    "failure_detection_seconds",
                    "recovery_seconds",
                    "checkpoint_verified",
                    "recovery_actor",
                    "clock_basis",
                ],
            ),
            table(
                "Core incident timeline (observer-relative seconds)",
                "incident",
                [
                    "job_id",
                    "workload",
                    "incident_index",
                    "observer_onset_seconds",
                    "observer_detected_seconds",
                    "observer_recovered_seconds",
                    "clock_basis",
                ],
            ),
            table(
                "Recovery actions",
                "action",
                ["job_id", "workload", "event_kind", "status", "actor", "observer_offset_seconds", "clock_basis"],
            ),
            metric_table("Recovery, checkpoints and resume efficiency", "metric", "resilience"),
            metric_chart(
                "Mean per-run checkpoint load duration (seconds)", "resilience", "checkpoint_load_mean_seconds"
            ),
            metric_table("Tracking-service recovery", "metric", "tracking_recovery"),
        ],
        "Infrastructure": [
            resource_chart("CPU usage (millicores, not percentage of capacity)", "cpu_millicores"),
            resource_chart("Memory working set (MiB)", "memory_mib"),
            table(
                "Sampled readiness and Pod restarts (not end-to-end availability)",
                "resource_sample",
                ["_timestamp", "job_id", "workload", "pod", "node", "ready", "phase", "restarts"],
            ),
            metric_table("Resource summaries, network deltas and sampled readiness", "resource_metric"),
            table("Kubernetes warning reasons", "warning", ["_timestamp", "job_id", "reason", "object"]),
            metric_table("Run duration, Pod replacements and restart totals", "metric", "run"),
        ],
        "Application and evidence": [
            metric_chart("Mean per-run held-out accuracy (adapter-defined)", "sli", "model_accuracy"),
            metric_table("Final model metrics", "metric", "model"),
            metric_table("Per-work-unit model metrics and duration", "unit_metric"),
            metric_table("Application-specific measurements", "metric", "application_metrics"),
            table(
                "Checkpoint and application events",
                "event",
                [
                    "job_id",
                    "workload",
                    "instance",
                    "event_kind",
                    "progress",
                    "seconds",
                    "bytes",
                    "observer_offset_seconds",
                    "clock_basis",
                ],
            ),
        ],
    }
    tabs = []
    for tab_index, (name, definitions_for_tab) in enumerate(definitions.items()):
        panels = []
        for index, (title, kind, query, axes) in enumerate(definitions_for_tab):
            fields = {axis: [] for axis in ("x", "y", "z", "breakdown")}
            if axes:
                for axis, alias, label in zip(
                    ("x", "y", "breakdown"), ("x_axis_1", "y_axis_1", "breakdown_1"), axes, strict=True
                ):
                    fields[axis] = [{"label": label, "alias": alias, "column": alias, "color": None}]
            panels.append(
                {
                    "id": f"benchmark_{tab_index}_{index}",
                    "type": kind,
                    "title": title,
                    "description": "Post-run benchmark evidence. Grouped by record_id to avoid replay double-counting.",
                    "config": {"show_legends": True, "legends_position": "bottom", "decimals": 3, "drilldown": []},
                    "queryType": "sql",
                    "queries": [
                        {
                            "query": query,
                            "customQuery": True,
                            "fields": {
                                "stream": stream,
                                "stream_type": "logs",
                                **fields,
                                "filter": {"filterType": "group", "logicalOperator": "AND", "conditions": []},
                            },
                            "config": {"promql_legend": ""},
                        }
                    ],
                    "layout": {"x": 0, "y": index * 10, "w": 96, "h": 10, "i": index},
                }
            )
        tabs.append({"tabId": f"benchmark_{tab_index}", "name": name, "panels": panels})
    return {
        "version": 8,
        "dashboardId": "",
        "title": "Reliability Benchmarks",
        "description": (
            "Post-run benchmark results using Core-deployed O2. Missing measurements remain unmeasured; "
            "observer times use their recorded clock basis. "
            "Compare runs with matching applications, recovery providers and modes."
        ),
        "role": "",
        "owner": "",
        "created": "2026-10-20T00:00:00Z",
        "variables": {"list": [], "showDynamicFilters": True},
        "defaultDatetimeDuration": {"type": "relative", "relativeTimePeriod": "24h"},
        "tabs": tabs,
    }


def main() -> int:
    """Write a native dashboard for import through O2's Dashboards page."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default="reliability_benchmarks")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dashboard(args.stream), indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Configurable MLflow observation and benchmark annotations."""

from __future__ import annotations

import json
import subprocess
import textwrap
from dataclasses import dataclass
from typing import Any

from et_experiments.serialization import finite_number


@dataclass(frozen=True)
class TrackingConfig:
    """Identify the application experiment, metric histories and deployed tracking and bridge services."""

    experiment: str
    metric_keys: tuple[str, ...]
    deployment: str = "mlflow"
    bridge_deployment: str = "mlflow-bridge"
    python: str = "/opt/cpython/bin/python3"


def mlflow_benchmark_payload(report: dict[str, Any], scenario_name: str) -> dict[str, Any]:
    """Map measured benchmark values and fault settings to namespaced MLflow metrics and tags."""
    resources = list(report.get("resources", {}).values())

    def maximum(key: str) -> float | None:
        values = [float(item[key]) for item in resources if finite_number(item.get(key))]
        return max(values) if values else None

    def total(key: str) -> float | None:
        values = [float(item[key]) for item in resources if finite_number(item.get(key))]
        return sum(values) if values else None

    fault = report.get("fault", {})
    resilience = report.get("resilience", {})
    durations = report.get("workload", {}).get("unit_duration_seconds", {})
    termination = fault.get("termination") or {}
    attribution = report.get("attribution", {})
    tracking = report.get("tracking_recovery", {})
    metrics = {
        "benchmark.wall_time_seconds": report.get("wall_time_seconds"),
        "benchmark.failure_detection_seconds": fault.get("failure_detection_seconds"),
        "benchmark.recovery_seconds": resilience.get("recovery_seconds"),
        "benchmark.fault_injection_duration_seconds": fault.get("injection_duration_seconds"),
        "benchmark.completion_ratio": resilience.get("completion_ratio"),
        "benchmark.completed_units_per_minute": resilience.get("completed_units_per_minute"),
        **{f"benchmark.work_unit_duration_{key}_seconds": durations.get(key) for key in ("mean", "p50", "p95", "max")},
        "benchmark.total_pod_restarts": report.get("total_pod_restarts"),
        "benchmark.total_pod_replacements": report.get("total_pod_replacements"),
        "benchmark.warning_events": len(report.get("warning_events", [])),
        "benchmark.cpu_peak_millicores": maximum("cpu_millicores_peak"),
        "benchmark.cpu_p95_millicores": maximum("cpu_millicores_p95"),
        "benchmark.memory_peak_mib": maximum("memory_mib_peak"),
        "benchmark.memory_p95_mib": maximum("memory_mib_p95"),
        "benchmark.network_rx_bytes": total("network_rx_bytes"),
        "benchmark.network_tx_bytes": total("network_tx_bytes"),
        "benchmark.recovery_intervention_attempts": len(attribution.get("interventions", [])),
        "benchmark.tracking_recovery_seconds": tracking.get("recovery_seconds"),
        "benchmark.tracking_recovery_verified": int(tracking["status"] == "verified")
        if tracking.get("status") in {"verified", "pending"}
        else None,
    }
    for key, value in report.get("application_metrics", {}).items():
        if finite_number(value):
            metrics[f"benchmark.{key}"] = value
    for key, value in fault.get("metrics", {}).items():
        if finite_number(value):
            metrics[f"benchmark.pressure.{key}"] = value
    for key, value in resilience.items():
        if finite_number(value):
            metrics.setdefault(f"benchmark.{key}", value)
    tags = {
        "benchmark.scenario": scenario_name,
        "benchmark.job_id": report.get("job_id"),
        "benchmark.fault_type": fault.get("type"),
        "benchmark.fault_target": fault.get("target"),
        "benchmark.fault_status": fault.get("status"),
        "benchmark.termination_reason": termination.get("reason"),
        "benchmark.recovery_mode": attribution.get("recovery_mode"),
        "benchmark.recovery_scope": attribution.get("scope"),
        "benchmark.recovery_provider": attribution.get("recovery_provider"),
        "benchmark.recovery_clock_basis": attribution.get("recovery_clock_basis"),
        "benchmark.recovery_actors": ",".join(attribution.get("recovery_actors", [])) or "unattributed",
        "benchmark.recovery_evidence": ",".join(attribution.get("evidence_levels", [])) or "not-measured",
        "benchmark.intervention_occurred": attribution.get("intervention_occurred"),
        "benchmark.tracking_recovery_actor": tracking.get("recovery_actor"),
        "benchmark.forced_bridge_restart": tracking.get("forced_bridge_restart"),
    }
    # Store configured fault intensity as searchable tags, separately from measured performance.
    # Tags support updates when annotation is retried.
    for key in (
        "duration_seconds",
        "latency_ms",
        "jitter_ms",
        "loss_percent",
        "rate_kbit",
        "memory_mib",
        "memory_limit_mib",
        "workers",
        "delay_ms",
    ):
        if fault.get(key) is not None:
            tags[f"benchmark.fault_{key}"] = fault[key]
    for key, value in report.get("application_tags", {}).items():
        tags.setdefault(f"benchmark.{key}", value)
    return {
        "metrics": {key: value for key, value in metrics.items() if finite_number(value)},
        "tags": {key: str(value) for key, value in tags.items() if value is not None},
    }


class MLflowTracking:
    """Read application runs and annotate them through commands executed in the tracking deployment."""

    def __init__(self, backend, config: TrackingConfig):
        self.backend, self.config = backend, config

    def mlflow_state(self, job_id: str) -> dict[str, Any] | None:
        """Read a run through MLflow's loopback REST API using the standard library."""
        script = r"""
    import json,sys,urllib.parse,urllib.request
    base="http://127.0.0.1:5000"
    def get(path,params):
        with urllib.request.urlopen(base+path+"?"+urllib.parse.urlencode(params),timeout=3) as response:
            return json.load(response)
    def post(path,payload):
        request=urllib.request.Request(
            base+path,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"}
        )
        with urllib.request.urlopen(request,timeout=3) as response:
            return json.load(response)
    experiment=get(
        "/api/2.0/mlflow/experiments/get-by-name",{"experiment_name":sys.argv[2]}
    )["experiment"]
    result=post(
        "/api/2.0/mlflow/runs/search",
        {"experiment_ids":[experiment["experiment_id"]],
            "filter":f"tags.mlflow.runName = '{sys.argv[1]}'","max_results":1},
    )
    runs=result.get("runs",[])
    if not runs: raise SystemExit(3)
    run=runs[0]
    history={}
    for key in json.loads(sys.argv[3]):
        values=get(
            "/api/2.0/mlflow/metrics/get-history",
            {"run_id":run["info"]["run_id"],"metric_key":key},
        ).get("metrics",[])
        history[key]=values
    print(json.dumps({
        "run_id":run["info"]["run_id"],"status":run["info"]["status"],
        "start_time":run["info"].get("start_time"),"history":history,
    }))
    """
        try:
            completed = self.backend.kubectl(
                "exec",
                "-n",
                self.backend.namespace,
                f"deployment/{self.config.deployment}",
                "--",
                self.config.python,
                "-c",
                textwrap.dedent(script),
                job_id,
                self.config.experiment,
                json.dumps(self.config.metric_keys),
                check=False,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            return None
        if completed.returncode or not completed.stdout.strip():
            return None
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            return None

    def annotate_mlflow(self, scenario_name: str, report: dict[str, Any]) -> str:
        """Add experiment metrics and tags to the matching run, returning an unavailable reason on command failure."""
        run_id = report.get("mlflow_run_id")
        if not run_id or run_id == "unavailable":
            return "unavailable: matching MLflow run not found"
        script = r"""
    import json,math,sys,time,urllib.request
    run_id,payload=sys.argv[1:3]
    data=json.loads(payload)
    base="http://127.0.0.1:5000/api/2.0/mlflow"
    def post(path,value):
        request=urllib.request.Request(
            base+path,data=json.dumps(value).encode(),headers={"Content-Type":"application/json"}
        )
        with urllib.request.urlopen(request,timeout=3): pass
    for key,value in data["tags"].items():
        post("/runs/set-tag",{"run_id":run_id,"key":key,"value":value})
    timestamp=int(time.time()*1000)
    for key,value in data["metrics"].items():
        value=float(value)
        if math.isfinite(value):
            post("/runs/log-metric",{"run_id":run_id,"key":key,"value":value,"timestamp":timestamp,"step":0})
    print(run_id)
    """
        payload = mlflow_benchmark_payload(report, scenario_name)
        try:
            completed = self.backend.kubectl(
                "exec",
                "-n",
                self.backend.namespace,
                f"deployment/{self.config.deployment}",
                "--",
                self.config.python,
                "-c",
                textwrap.dedent(script),
                str(run_id),
                json.dumps(payload, allow_nan=False),
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "unavailable: MLflow annotation timed out"
        return completed.stdout.strip() if completed.returncode == 0 else f"unavailable: {completed.stderr.strip()}"


class NoTracking:
    """Provide the engine's tracking interface when an application has no tracking service."""

    def mlflow_state(self, _job_id: str) -> None:
        return None

    def annotate_mlflow(self, _scenario_name: str, _report: dict[str, Any]) -> str:
        """Report explicitly that benchmark annotation is disabled for this application."""
        return "unavailable: tracking is not configured"

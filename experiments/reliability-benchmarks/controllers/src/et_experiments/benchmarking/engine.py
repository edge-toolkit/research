"""Run experiments with explicit application, deployment, and tracking dependencies."""

from __future__ import annotations

import datetime as dt
import logging
import os
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Any

from et_experiments.backends.core_recovery import CoreRecoveryMonitor, CoreRecoveryRuntime
from et_experiments.backends.kubernetes import KubernetesBackend
from et_experiments.contracts import ApplicationAdapter
from et_experiments.recovery.attribution import describe_recovery, recovery_mode
from et_experiments.recovery.core import (
    checkpoint_metrics,
    evaluate_slos,
    parse_events,
)
from et_experiments.serialization import json_dumps
from et_experiments.tracking.mlflow import MLflowTracking, NoTracking
from et_experiments.tracking.o2 import publish_report
from et_experiments.utils import BenchmarkError, clean_log, iso, progress, utc_now

from .design import FaultTrigger, validate_design
from .evidence import processing_metrics, recovery_evidence
from .faults import KubernetesFaults
from .observations import fault_incidents, recovery_outcome, stability_outcome, work_finished
from .reference import capture_reference
from .reports import save_report, write_logs, write_reports
from .sampling import Sampler, resource_summary
from .saturation import annotate_stages

logger = logging.getLogger(__name__)


def _restore_steps(steps):
    """Attempt every independent restoration action and report all failures together."""
    errors = []
    for name, restore in steps:
        try:
            restore()
        except Exception as error:  # noqa: BLE001 - remaining cleanup actions must still run
            errors.append(f"{name}: {type(error).__name__}: {error}")
    if errors:
        raise BenchmarkError("Restoration failed: " + "; ".join(errors))


class BenchmarkEngine:
    """Coordinate fault injection, Core recovery evidence, application measurements and deployment cleanup."""

    def __init__(self, application: ApplicationAdapter, backend=None, tracking=None):
        """Bind an application adapter, using default Kubernetes and tracking implementations unless supplied."""
        self.application = application
        self.config = application.config
        self.backend = backend or KubernetesBackend(
            self.config.context, self.config.namespace, self.config.primary, self.config.observed
        )
        self.tracking = tracking or (
            MLflowTracking(self.backend, application.tracking) if application.tracking else NoTracking()
        )
        self.faults = KubernetesFaults(self.backend, application)
        self.sampler_factory = lambda interval: Sampler(self.backend, interval)
        self.runtime_factory = CoreRecoveryRuntime
        self.monitor_factory = CoreRecoveryMonitor

    def configure_run(self, job_id, scenario, started):
        """Delegate job setup and application startup to the adapter with the shared fault helpers."""
        self.application.configure_run(self.backend, self.faults, job_id, scenario, started)

    def restore(self):
        """Restore resource limits, remove stress jobs and network helpers, then restore application settings."""
        _restore_steps(
            [
                ("saturation", self.faults.restore_saturation),
                ("scheduling", self.faults.scheduling.restore),
                (
                    "stress jobs",
                    partial(
                        self.backend.kubectl,
                        "delete",
                        "jobs",
                        "-n",
                        self.config.namespace,
                        "-l",
                        "app=et-benchmark",
                        "--ignore-not-found=true",
                        check=True,
                    ),
                ),
                *(
                    (f"network sidecar {deployment}", partial(self.faults.remove_network_fault_sidecar, deployment))
                    for deployment in self.config.workloads
                ),
                ("application", partial(self.application.restore, self.backend)),
            ]
        )

    def write_reports(self, output, report):
        """Attempt optional O2 export before saving detailed evidence and appending CSV summaries."""
        if os.environ.get("BENCHMARK_O2_DEFER") != "1":
            report["o2"] = publish_report(report, output)
        write_reports(output, report)

    def execute(
        self, name: str, scenario: dict[str, Any], output: Path, sample_interval: float, keep_configuration: bool
    ) -> dict[str, Any]:
        """Run one scenario and return its measurements, recovery attribution and SLO assessments.

        Validate controls before preparation, inject at the observed trigger and retain collected evidence. Restore the
        deployment unless keep_configuration is set; cleanup failures propagate and mark any saved report failed.
        """
        scenario = self.application.prepare_scenario(scenario)
        seed = validate_design(scenario, self.config.workers)
        trigger = FaultTrigger(scenario.get("fault", {}), seed)
        mode = recovery_mode(scenario)
        expected_units = self.application.expected_units(scenario)
        if type(expected_units) is not int or expected_units < 1:
            raise BenchmarkError("application expected_units must be a positive integer")
        if scenario.get("require_tracking_recovery") and self.application.tracking is None:
            raise BenchmarkError("tracking recovery requires a configured tracking service")
        self.faults.interventions.clear()
        if self.backend.kubectl("config", "current-context").stdout.strip() != self.config.context:
            raise BenchmarkError(f"kubectl context must be {self.config.context}")
        job_id = f"bench-{name}-{utc_now().strftime('%Y%m%d%H%M%S%f')}"
        started = utc_now()
        sampler = self.sampler_factory(sample_interval)
        recovery_config = {**self.application.recovery_config(), "started_at": started}
        recovery_config["deployment_resilience"] = recovery_config.get("resilience", {})
        recovery_config.setdefault("default_resilience", recovery_config["deployment_resilience"])
        recovery_config["resilience"] = scenario.get("recovery_resilience", recovery_config["deployment_resilience"])
        if scenario.get("require_recovery_stability") and (
            not scenario.get("require_recovery") or recovery_config["resilience"].get("stability_seconds", 0) <= 0
        ):
            raise BenchmarkError("recovery stability requires recovery verification and a positive stability window")
        recovery_runtime = self.runtime_factory(self.backend, recovery_config, job_id, mode)
        monitor = self.monitor_factory(recovery_runtime, sample_interval)
        monitor_started = False
        recovery_report = None
        fault = scenario.get("fault", {"type": "none"})
        fault_result: dict[str, Any] = {
            "type": fault.get("type", "none"),
            "status": "not-applicable" if fault.get("type", "none") == "none" else "pending",
        }
        fault_result["trigger"] = trigger.report()
        fault_injected = fault.get("type") == "none"
        tracking_evidence = {"status": "pending" if scenario.get("require_tracking_recovery") else "not-applicable"}
        tracking_verified_at = None
        observed_mlflow = None
        prepared = False
        evidence_wait_since = None
        stability_wait_since = None
        post_fault_wait_since = None
        measurement_note = None
        reference = {"missing": ["deployment-not-prepared"]}
        diagnostic_buffer = {}
        cleanup = {"status": "pending"}
        stored_report = None
        service_monitor = None
        service_result = None

        def finish_service_probes():
            """Retain raw probe outcomes separately and attach compact measurements to the job report."""
            nonlocal service_result
            if service_monitor is not None and service_result is None:
                result = service_monitor.finish()
                output.mkdir(parents=True, exist_ok=True)
                path = output / f"{job_id}-service-probes.json"
                path.write_text(json_dumps(result, indent=2) + "\n")
                service_result = {key: value for key, value in result.items() if key != "records"}
                service_result["observations_file"] = path.name
            return service_result

        assessments_complete = False
        post_fault = {"required": bool(scenario.get("require_post_fault_progress")), "status": "not-required"}
        if post_fault["required"]:
            post_fault["status"] = "pending"
            post_fault["timeout_seconds"] = float(
                scenario.get(
                    "post_fault_progress_timeout_seconds",
                    recovery_config.get("policy", {}).get("deadline_seconds", 240),
                )
            )

        def attributed(raw):
            """Attach recorded interventions and verified tracking recovery to the current recovery evidence."""
            fault_details = {**fault_result, "interventions": list(self.faults.interventions)}
            result = describe_recovery(raw, mode, fault_details, list(recovery_runtime.events.values()))
            if tracking_evidence["status"] == "verified":
                details = result["attribution"]
                details["recovery_actors"] = sorted(
                    set(details["recovery_actors"]) | {tracking_evidence["recovery_actor"]}
                )
                details["evidence_levels"] = sorted(set(details["evidence_levels"]) | {"tracking-ingestion"})
            return result

        try:
            progress(started, f"Preparing scenario '{name}'")
            progress(started, f"Core recovery mode: {mode}; research injects faults and measures the deployed system")
            # Observe before rollout so fast work cannot finish during later readiness checks.
            observation_start = utc_now()
            recovery_runtime.prepare()
            prepared = True
            self.configure_run(job_id, scenario, started)
            if scenario.get("service_probes", {}).get("enabled"):
                factory = getattr(self.application, "service_probe_monitor", None)
                if not callable(factory):
                    raise BenchmarkError("this application does not implement functional service probes")
                service_monitor = factory(self.backend, job_id, scenario)
                service_monitor.start()
            reference = capture_reference(self.application, scenario, self.backend)
            sampler.start()
            if mode != "disabled":
                monitor_started = True
                monitor.start()
            deadline = time.monotonic() + float(scenario.get("timeout_seconds", 900))
            workload_started = time.monotonic()
            progress(started, f"Workload started: 0/{expected_units} work units complete")
            last_completed = -1
            last_failed = False
            waiting_for_recovery = False
            last_update = 0.0
            logs, completed = "", 0
            while time.monotonic() < deadline:
                logs = self.backend.primary_logs(observation_start)
                diagnostic_buffer.update({e["id"]: e for e in parse_events(logs, job_id)})
                observed_mlflow = self.tracking.mlflow_state(job_id)
                parsed = self.application.observe(
                    {self.config.primary: logs}, observed_mlflow, expected_units, observation_start
                )
                if monitor_started and parsed["failed"] != last_failed:
                    message = (
                        "Primary workload failure detected; waiting for automatic recovery and resumed progress"
                        if parsed["failed"]
                        else "Primary workload resumed progress after failure"
                    )
                    progress(started, message)
                last_failed = parsed["failed"]
                completed = parsed["completed_units"]
                if (
                    post_fault_wait_since is not None
                    and post_fault["status"] == "pending"
                    and completed <= post_fault["completed_at_release"]
                    and time.monotonic() - post_fault_wait_since >= post_fault["timeout_seconds"]
                ):
                    post_fault.update(status="deadline-exceeded", primary_failed=parsed["failed"])
                    measurement_note = (
                        f"No useful progress within {post_fault['timeout_seconds']:g} seconds after fault release; "
                        f"primary workload {'reported failure' if parsed['failed'] else 'did not advance'}"
                    )
                    progress(started, measurement_note)
                    break
                if (
                    post_fault.get("completed_at_release") is not None
                    and completed > post_fault["completed_at_release"]
                    and post_fault["status"] != "verified"
                ):
                    verified = utc_now()
                    post_fault.update(
                        status="verified",
                        verified_at=iso(verified),
                        seconds=(
                            verified
                            - dt.datetime.fromisoformat(post_fault["observed_release_at"].replace("Z", "+00:00"))
                        ).total_seconds(),
                    )
                if completed != last_completed:
                    if completed:
                        progress(started, f"Work unit {completed}/{expected_units} completed")
                    last_completed = completed
                    last_update = time.monotonic()
                elif time.monotonic() - last_update >= 15:
                    pod_state = sampler.samples[-1]["pods"] if sampler.samples else {}
                    ready = sum(bool(item.get("ready")) for item in pod_state.values())
                    recovering = waiting_for_recovery or parsed["failed"]
                    activity = "Waiting for recovery" if recovering else "Workload in progress"
                    if completed >= expected_units and tracking_evidence["status"] == "pending":
                        activity = "Waiting for post-fault tracking ingestion"
                    progress(
                        started,
                        f"{activity}: "
                        f"{completed}/{expected_units} work units complete, "
                        f"{ready}/{len(pod_state)} observed pods ready",
                    )
                    last_update = time.monotonic()
                stage_events = []
                if not fault_injected and trigger.settings.get("event_kind"):
                    stage_events = [
                        event
                        for text in self.backend.collect_logs(observation_start).values()
                        for event in parse_events(text, job_id)
                    ]
                    diagnostic_buffer.update({e["id"]: e for e in stage_events})
                if not fault_injected and trigger.ready(completed, time.monotonic() - workload_started, stage_events):
                    for text in self.backend.collect_logs(observation_start).values():
                        diagnostic_buffer.update({e["id"]: e for e in parse_events(text, job_id)})
                    baseline = len(monitor.snapshot()["incidents"]) if monitor_started else 0
                    progress(started, f"Injecting fault: {fault.get('type')} target={fault.get('target', 'node')}")
                    fault_result.update(target=fault.get("target"), started_at=iso(utc_now()), status="injecting")
                    fault_result = self.faults.inject_fault(fault, job_id)
                    fault_result["trigger"] = trigger.report()
                    event_at = (trigger.evidence or {}).get("event_at")
                    if event_at is not None:
                        lag = (
                            dt.datetime.fromisoformat(fault_result["started_at"].replace("Z", "+00:00")).timestamp()
                            - event_at
                        )
                        fault_result["trigger"]["event_to_injection_seconds"] = lag if lag >= 0 else None
                    fault_result["core_incident_baseline"] = baseline
                    fault_injected = True
                    progress(started, f"Fault action finished with status={fault_result['status']}")
                    if post_fault["required"]:
                        boundary = self.application.observe(
                            {self.config.primary: self.backend.primary_logs(observation_start)},
                            self.tracking.mlflow_state(job_id),
                            expected_units,
                            observation_start,
                        )
                        post_fault.update(
                            completed_at_release=boundary["completed_units"], observed_release_at=iso(utc_now())
                        )
                        post_fault_wait_since = time.monotonic()
                        if boundary["completed_units"] >= expected_units:
                            measurement_note = "Workload finished before post-release progress could be measured"
                            progress(started, measurement_note)
                            break
                tracking_ready = not scenario.get("require_tracking_recovery")
                if not tracking_ready and fault_injected:
                    tracking_evidence = self.application.tracking_recovery(
                        observed_mlflow,
                        expected_units,
                        fault_result,
                        tracking_verified_at,
                    )
                    tracking_verified_at = tracking_evidence["verified_at"]
                    tracking_ready = tracking_evidence["status"] == "verified"
                recovery_status = "recovered"
                if scenario.get("require_recovery"):
                    if not monitor_started:
                        raise BenchmarkError("require_recovery needs recovery_enabled")
                    snapshot = monitor.snapshot() if fault_injected else {"incidents": []}
                    recovery_status = (
                        recovery_outcome(snapshot, scenario, fault_result) if fault_injected else "pending"
                    )
                    if (
                        snapshot.get("provider") == "edge-toolkit-core"
                        and completed >= expected_units
                        and recovery_status == "pending"
                        and (
                            not fault_incidents(snapshot, fault_result)
                            or all(item["status"] == "recovered" for item in fault_incidents(snapshot, fault_result))
                        )
                    ):
                        if evidence_wait_since is None:
                            evidence_wait_since = time.monotonic()
                        if time.monotonic() - evidence_wait_since >= float(
                            scenario.get("recovery_evidence_timeout_seconds", 30)
                        ):
                            measurement_note = (
                                "No new Core target incident observed after workload completed; MTTR unmeasured"
                                if not fault_incidents(snapshot, fault_result)
                                else "Core recovered, but required checkpoint/resume evidence was not verified in time"
                            )
                            progress(started, measurement_note)
                            break
                stability_status = "not-requested"
                if scenario.get("require_recovery_stability") and recovery_status == "recovered":
                    stability_status = stability_outcome(snapshot, scenario, fault_result)
                    if stability_status == "pending" and completed >= expected_units:
                        if stability_wait_since is None:
                            stability_wait_since = time.monotonic()
                            progress(started, "Useful recovery verified; observing sustained recovery separately")
                        stability_timeout = recovery_config["resilience"]["stability_seconds"] + max(
                            30, 2 * recovery_config.get("poll_seconds", 2)
                        )
                        if time.monotonic() - stability_wait_since >= stability_timeout:
                            measurement_note = "Recovery stability was not verified within its observation window"
                            progress(started, measurement_note)
                            break
                    elif stability_status in {"unstable", "incomplete"}:
                        measurement_note = f"Useful recovery observed; recovery stability is {stability_status}"
                        progress(started, measurement_note)
                        break
                if recovery_status in {"unrecovered", "not-detected"}:
                    message = (
                        "Core classified the interruption as transient; MTTR was not measured"
                        if recovery_status == "not-detected"
                        else "Recovery deadline exceeded; collecting the failed experiment evidence"
                    )
                    progress(started, message)
                    measurement_note = message
                    break
                post_fault_ready = not post_fault["required"] or post_fault["status"] == "verified"
                completion_ready = not scenario.get("completion_event") or any(
                    event["kind"] == scenario["completion_event"]
                    and event["workload"] == self.config.primary
                    and event.get("progress", -1) >= expected_units
                    for event in diagnostic_buffer.values()
                )
                if (
                    work_finished(parsed, monitor_started)
                    and completion_ready
                    and recovery_status == "recovered"
                    and stability_status in {"not-requested", "stable"}
                    and tracking_ready
                    and (post_fault_ready or parsed["failed"])
                ):
                    if not fault_injected:
                        measurement_note = "Workload finished before the configured fault trigger was observed"
                    if parsed["failed"]:
                        progress(
                            started,
                            "Primary workload failure detected; automatic recovery is disabled for this scenario",
                        )
                    break
                if completed >= expected_units and not waiting_for_recovery:
                    progress(
                        started, "Workload complete; waiting for verified application/tracking recovery before cleanup"
                    )
                    waiting_for_recovery = True
                time.sleep(2)
            else:
                pods = self.backend.kubectl(
                    "get", "pods", "-n", self.config.namespace, "--no-headers", check=False
                ).stdout.strip()
                tail = "\n".join(clean_log(line) for line in logs.splitlines()[-12:])
                raise BenchmarkError(
                    f"benchmark timed out after {scenario.get('timeout_seconds', 900)} seconds "
                    f"with {completed}/{expected_units} work units complete\nPods:\n{pods or '<none>'}\n"
                    f"Primary workload log tail:\n{tail or '<no primary workload logs>'}\n"
                    f"Tracking recovery evidence: {json_dumps(tracking_evidence)}"
                )

            finish_service_probes()
            progress(started, "Collecting logs, Kubernetes events, and resource summaries")
            logs = self.backend.collect_logs(observation_start)
            diagnostic_buffer.update(
                {event["id"]: event for text in logs.values() for event in parse_events(text, job_id)}
            )
            for text in logs.values():
                for event in parse_events(text, job_id):
                    if not monitor_started or event["kind"] not in {
                        "checkpoint-saved",
                        "checkpoint-restored",
                        "resume-completed",
                    }:
                        recovery_runtime.events[event["id"]] = event
            write_logs(output, job_id, logs)
            final_mlflow = self.tracking.mlflow_state(job_id)
            workload = self.application.observe(logs, final_mlflow, expected_units, observation_start)
            sampler.finish()
            recovery_report = attributed(monitor.finish() if monitor_started else {"incidents": [], "metrics": {}})
            if measurement_note:
                recovery_report["measurement_note"] = measurement_note
            monitor_started = False
            resources = resource_summary(sampler.samples)
            if tracking_evidence["status"] == "verified":
                bridge = resources.get(self.application.tracking.bridge_deployment, {})
                replaced = bool(bridge.get("peak_restarts", 0) or bridge.get("pod_replacements", 0))
                tracking_evidence["bridge_process_replacement_observed"] = replaced
                if replaced:
                    recovery_report["attribution"]["recovery_actors"].append("kubernetes-bridge-replacement-observed")
            restarts = sum(item["peak_restarts"] for item in resources.values())
            replacements = sum(item["pod_replacements"] for item in resources.values())
            completion_ratio = workload["completed_units"] / expected_units
            incidents = fault_incidents(recovery_report, fault_result)
            recovered = [i for i in incidents if i["status"] == "recovered"]
            recovery_seconds = (
                recovered[-1]["recovered_at"] - recovered[-1]["observed_failure_at"] if recovered else None
            )
            events = list(recovery_runtime.events.values())
            evidence = checkpoint_metrics(events)
            fault_restored = fault_result.get("status") not in {"not-observed", "restore-failed"}
            successful = (
                workload["completed_units"] >= expected_units
                and not workload["failed"]
                and fault_restored
                and (not post_fault["required"] or post_fault["status"] == "verified")
                and (not scenario.get("require_tracking_recovery") or tracking_evidence["status"] == "verified")
            )
            if scenario.get("require_recovery"):
                successful = (
                    successful
                    and recovery_outcome({**recovery_report, "events": events}, scenario, fault_result) == "recovered"
                )
            stable = stability_outcome(recovery_report, scenario, fault_result)
            stability_ready = stable in {"not-requested", "stable"}
            successful = successful and stability_ready
            expected_failure = scenario.get("expected_failure_message")
            allowed_failure = (
                bool(scenario.get("allow_run_failure"))
                and workload["failed"]
                and (not expected_failure or expected_failure in logs.get(self.config.primary, ""))
            )
            report: dict[str, Any] = {
                "schema_version": 1,
                "job_id": job_id,
                "application": self.config.namespace,
                "namespace": self.config.namespace,
                "scenario": name,
                "effective_scenario": scenario,
                "service_probes": service_result or {"status": "disabled"},
                "fault_configuration": scenario.get("fault", {"type": "none"}),
                "cleanup": cleanup,
                "reference": reference,
                "experiment_seed": seed,
                "description": scenario.get("description"),
                "started_at": iso(started),
                "finished_at": iso(utc_now()),
                "wall_time_seconds": (utc_now() - started).total_seconds(),
                "status": "passed"
                if (successful or allowed_failure)
                and stability_ready
                and fault_injected
                and (not post_fault["required"] or post_fault["status"] == "verified")
                else "failed",
                "expected_failure_observed": allowed_failure,
                "fault": fault_result,
                "post_fault_progress": post_fault,
                "workload": workload,
                "resource_metrics_available": sampler.metrics_available,
                "resources": resources,
                "total_pod_restarts": restarts,
                "total_pod_replacements": replacements,
                "resilience": {
                    **recovery_report["metrics"],
                    **evidence,
                    "completion_ratio": completion_ratio,
                    "post_fault_progress_seconds": post_fault.get("seconds"),
                    "post_fault_progress_verified": int(post_fault["status"] == "verified")
                    if post_fault["required"]
                    else None,
                    "recovery_seconds": recovery_seconds,
                    "recovered_after_fault": recovery_seconds is not None if fault.get("type") != "none" else None,
                    "completed_units_per_minute": (
                        workload["completed_units"] / max((utc_now() - started).total_seconds(), 0.001) * 60
                    ),
                },
                "recovery": recovery_report,
                "recovery_stability": {"required": bool(scenario.get("require_recovery_stability")), "status": stable},
                "attribution": recovery_report["attribution"],
                "tracking_recovery": tracking_evidence,
                "events": events,
                "diagnostic_events": sorted(diagnostic_buffer.values(), key=lambda e: e["at"]),
                "warning_events": self.backend.warning_events(observation_start),
                "samples": sampler.samples,
            }
            stored_report = report
            annotate_stages(report)
            application_slis = self.application.enrich_report(report)
            report["resilience"].update(
                processing_metrics(report["diagnostic_events"], self.config.primary, expected_units)
            )
            report["useful_recovery"] = recovery_evidence(report, self.config.primary)
            if scenario.get("require_participant_recovery"):
                returns = report["useful_recovery"].get("participant_recovery", {})
                if not returns or any(item["status"] != "verified" for item in returns.values()):
                    report["status"] = "failed"
            if scenario.get("require_unfinished_resume") and not report["useful_recovery"].get("checkpoint", {}).get(
                "resumed_unfinished_work"
            ):
                report["status"] = "failed"
            slo_metrics = {
                **report["resilience"],
                **{f"pressure_{key}": value for key, value in fault_result.get("metrics", {}).items()},
                "work_unit_p95_seconds": workload["unit_duration_seconds"]["p95"],
                **application_slis,
            }
            report["sli"] = slo_metrics
            report["slos"] = evaluate_slos(slo_metrics, scenario.get("slos", {}))
            if any(item["status"] != "passed" for item in report["slos"].values()):
                report["status"] = "failed"
            assessments_complete = True
            # Persist the benchmark evidence before optional MLflow enrichment so
            # an unavailable tracking service can never discard a completed run.
            report["mlflow_run_id"] = final_mlflow.get("run_id") if final_mlflow else "unavailable"
            output.mkdir(parents=True, exist_ok=True)
            report_path = output / f"{job_id}.json"
            report_path.write_text(json_dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            progress(started, "Annotating the MLflow run (best effort)")
            annotation = self.tracking.annotate_mlflow(name, report)
            if not annotation.startswith("unavailable:"):
                report["mlflow_run_id"] = annotation
            else:
                report["mlflow_annotation"] = annotation
            if os.environ.get("BENCHMARK_O2_URL") and os.environ.get("BENCHMARK_O2_DEFER") != "1":
                progress(started, "Exporting results to OpenObserve (best effort)")
            progress(started, "Measurements collected; finalising cleanup and reports")
            return report
        except (BenchmarkError, subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            if monitor_started:
                try:
                    recovery_report = attributed(monitor.finish())
                except (BenchmarkError, subprocess.TimeoutExpired) as observation_error:
                    recovery_report = attributed({**monitor.last_report, "observation_error": str(observation_error)})
                monitor_started = False
            failed = stored_report if stored_report is not None else {}
            failed.update(
                {
                    "schema_version": 1,
                    "job_id": job_id,
                    "application": self.config.namespace,
                    "namespace": self.config.namespace,
                    "started_at": iso(started),
                    "finished_at": iso(utc_now()),
                    "scenario": name,
                    "effective_scenario": scenario,
                    "fault_configuration": scenario.get("fault", {"type": "none"}),
                    "cleanup": cleanup,
                    "reference": reference,
                    "experiment_seed": seed,
                    "status": "failed",
                    "error": str(exc),
                    "fault": fault_result,
                    "post_fault_progress": post_fault,
                    "recovery": recovery_report or {"incidents": [], "metrics": {}},
                    "attribution": (recovery_report or attributed({"incidents": [], "metrics": {}}))["attribution"],
                    "tracking_recovery": tracking_evidence,
                    "events": list(recovery_runtime.events.values()),
                    "diagnostic_events": sorted(diagnostic_buffer.values(), key=lambda e: e["at"]),
                    "mlflow_run_id": observed_mlflow.get("run_id") if observed_mlflow else "unavailable",
                    "wall_time_seconds": (utc_now() - started).total_seconds(),
                }
            )
            output.mkdir(parents=True, exist_ok=True)
            stored_report = failed
            report_path = output / f"{job_id}.json"
            report_path.write_text(json_dumps(failed, indent=2) + "\n")
            if os.environ.get("BENCHMARK_RESULT_FILE"):
                Path(os.environ["BENCHMARK_RESULT_FILE"]).write_text(json_dumps(failed, indent=2) + "\n")
            failed["mlflow_annotation"] = self.tracking.annotate_mlflow(name, failed)
            report_path.write_text(json_dumps(failed, indent=2) + "\n")
            raise
        finally:
            try:
                finish_service_probes()
                if stored_report is not None and service_result is not None:
                    stored_report["service_probes"] = service_result
            except Exception:
                logger.warning("Could not finalize functional probe observations", exc_info=True)
            if stored_report is not None and not assessments_complete:
                stored_report["status"] = "failed"
                stored_report.setdefault("error", "Report enrichment or success-gate evaluation did not complete")
            if monitor_started:
                try:
                    recovery_report = attributed(monitor.finish())
                # Finalization must preserve the original exception and allow deployment cleanup.
                except Exception:
                    logger.warning("Could not finalize recovery observations", exc_info=True)
            if recovery_report is not None:
                try:
                    output.mkdir(parents=True, exist_ok=True)
                    (output / f"{job_id}-recovery.json").write_text(json_dumps(recovery_report, indent=2) + "\n")
                # Serialization and filesystem failures must also leave cleanup reachable.
                except Exception:
                    logger.warning("Could not write the recovery report", exc_info=True)
            sampler.finish()
            try:
                if prepared and not keep_configuration:
                    progress(started, "Restoring the normal manual-start application configuration")
                    _restore_steps(
                        [
                            ("stop supervisor", recovery_runtime.stop),
                            ("workloads", self.restore),
                            ("supervisor defaults", recovery_runtime.restore_defaults),
                        ]
                    )
                    cleanup["status"] = "restored"
                else:
                    cleanup["status"] = "retained" if keep_configuration else "not-required"
            except (BenchmarkError, subprocess.TimeoutExpired, OSError) as cleanup_error:
                cleanup.update(status="failed", error=str(cleanup_error))
                if stored_report is not None:
                    stored_report["status"] = "failed"
                raise
            finally:
                if stored_report is not None:
                    if "workload" in stored_report:
                        self.write_reports(output, stored_report)
                    else:
                        if os.environ.get("BENCHMARK_O2_DEFER") != "1":
                            stored_report["o2"] = publish_report(stored_report, output)
                        save_report(output, stored_report)
                    progress(
                        started, f"Benchmark {stored_report['status']}; report written to {output / f'{job_id}.json'}"
                    )

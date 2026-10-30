"""Configure Core supervision for experiments and collect recovery reports."""

from __future__ import annotations

import json
import re
import time

from et_experiments.recovery.core import parse_events
from et_experiments.recovery.supervision import CLOCKS, measurements, stability, verify_continuation
from et_experiments.utils import BenchmarkError


def checkpoint_evidence(raw, index):
    """Pair restoration/resume from one process, allowing delivery after Core closes an incident."""
    record = raw["incidents"][index]
    end = min(
        (item["onset"] for item in raw["incidents"][index + 1 :] if item["workload"] == record["workload"]),
        default=float("inf"),
    )
    measurements = [item for item in raw.get("checkpoint_measurements", []) if item["workload"] == record["workload"]]
    old_instances = {item["instance"] for item in measurements if item["observed_at"] < record["onset"]}
    # A durable checkpoint may precede the incident baseline because completed work since the save was unsaved.
    candidates = [
        item
        for item in measurements
        if record["onset"] <= item["observed_at"] < end
        and item.get("instance")
        and item["instance"] not in old_instances
    ]
    for restored in candidates:
        if restored["kind"] != "checkpoint-restored":
            continue
        for resumed in candidates:
            if (
                resumed["kind"] == "resume-completed"
                and resumed["instance"] == restored["instance"]
                and resumed["observed_at"] >= restored["observed_at"]
                and resumed["progress"] >= restored["progress"]
            ):
                return {
                    "checkpoint_restored_at": restored["observed_at"],
                    "resume_completed_at": resumed["observed_at"],
                    "checkpoint_instance": restored["instance"],
                }
    return {}


def normalize_report(raw, job_id, mode):
    """Normalize Core's recovery decisions while preserving its observer clock and progress evidence."""
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or raw.get("job_id") != job_id:
        raise BenchmarkError("Core recovery report has the wrong schema or job identity")
    if raw.get("mode") != mode or raw.get("clock_basis") not in CLOCKS:
        raise BenchmarkError("Core recovery report has an unexpected mode or clock basis")
    context = raw.get("context", {})
    gaps = raw.get("observation_gaps", [])
    if not isinstance(gaps, list) or any(
        not isinstance(gap, dict)
        or type(gap.get("started_at")) is not int
        or gap["started_at"] < 0
        or (
            gap.get("ended_at") is not None
            and (type(gap["ended_at"]) is not int or gap["ended_at"] < gap["started_at"])
        )
        for gap in gaps
    ):
        raise BenchmarkError("Core report has malformed observation gaps")
    if raw["clock_basis"] == "observer-elapsed-seconds-with-restart-gaps" and (
        context.get("schema_version") != 1
        or type(context.get("restored_sessions")) is not int
        or context["restored_sessions"] < 1
        or context.get("configuration", {}).get("job_id") != job_id
        or not any(gap.get("reason") == "supervisor restart" for gap in raw.get("observation_gaps", []))
    ):
        raise BenchmarkError("Core restored observer clock lacks durable recovery context")
    incidents = []
    for index, record in enumerate(raw["incidents"]):
        evidence = checkpoint_evidence(raw, index)
        actions = []
        for action in record["actions"]:
            actions.append(
                {
                    **action,
                    "actor": "edge-toolkit-core",
                    "type": "restart" if action["kind"] == "restart-requested" else action["kind"],
                    "status": "requested" if action["kind"] == "restart-requested" else action["kind"],
                }
            )
        incidents.append(
            {
                "core_incident_index": index,
                "workload": record["workload"],
                "status": record["status"],
                "observed_failure_at": record["onset"],
                "detected_at": record["detected"],
                "recovered_at": record["recovered"],
                "progress_before": record["baseline"],
                "ready_at": None,
                "actions": actions,
                "stability": stability(record),
                "evidence": evidence,
                "onset_basis": "core-first-unhealthy-observation",
                "clock_basis": raw["clock_basis"],
                "recovery_actor": "edge-toolkit-core"
                if any(a["type"] == "restart" for a in actions)
                else "unattributed",
            }
        )
    detected = [item for item in incidents if item["detected_at"] is not None]
    recovered = [item for item in detected if item["status"] == "recovered"]
    checkpoint_events = [
        {
            "id": f"core-checkpoint-{index}",
            "kind": item["kind"],
            "workload": item["workload"],
            "at": item["observed_at"],
            "clock_basis": raw["clock_basis"],
            "instance": item["instance"],
            "progress": item["progress"],
            "bytes": item["bytes"],
            "seconds": item["duration_millis"] / 1000,
        }
        for index, item in enumerate(raw.get("checkpoint_measurements", []))
    ]
    return {
        "provider": "edge-toolkit-core",
        "clock_basis": raw["clock_basis"],
        "job_id": job_id,
        "mode": mode,
        "incidents": incidents,
        "events": checkpoint_events,
        "core_report": raw,
        "assessments": raw.get("assessments", {}),
        "observation_gaps": raw.get("observation_gaps", []),
        "resilience": context.get("configuration", {}).get("resilience", {}),
        "restored_sessions": context.get("restored_sessions", 0),
        "metrics": {
            **measurements(incidents, raw.get("observation_gaps", [])),
            "mttr_seconds": raw["mttr_seconds"],
            "failure_detection_mean_seconds": raw["mean_failure_detection_seconds"],
            "recovery_success_ratio": len(recovered) / len(detected) if detected else None,
            "detected_incidents": len(detected),
            "unrecovered_incidents": len(detected) - len(recovered),
            "recovery_attempts": sum(a["kind"] == "restart-reserved" for i in incidents for a in i["actions"]),
            "observation_errors": raw["unknown_observations"],
            "observed_availability_ratio": None,
        },
    }


class CoreRecoveryRuntime:
    """Prepare experiment supervision using Core's recovery actions and durable restart budgets."""

    def __init__(self, backend, config, job_id, mode):
        """Validate the recovery mode and allocate a job-specific state path without changing the deployment."""
        if mode not in {"observe-only", "controller-assisted", "disabled"}:
            raise BenchmarkError("unsupported Core recovery mode")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", job_id):
            raise BenchmarkError("benchmark job identity is not safe for a supervisor state filename")
        self.backend, self.config, self.job_id, self.mode = backend, config, job_id, mode
        self.deployment = config.get("supervisor_deployment", "")
        self.state_file = f"/app/state/{job_id}.json"
        self.events = {}
        self.session = None
        self.active = False
        self.last_report = {"provider": "edge-toolkit-core", "incidents": [], "metrics": {}}

    def prepare(self):
        """Stop supervision before intentional workload reconfiguration, even for a disabled experiment."""
        if not self.deployment:
            if self.mode != "disabled":
                raise BenchmarkError("the application must configure its Core supervisor deployment")
            return
        self.backend.kubectl("get", f"deployment/{self.deployment}", "-n", self.backend.namespace, "-o", "name")
        self.stop()
        stopped = self.backend.kubectl(
            "wait",
            "--for=delete",
            "pod",
            "-n",
            self.backend.namespace,
            "-l",
            f"app={self.deployment}",
            "--timeout=120s",
            check=False,
            timeout=125,
        )
        if stopped.returncode:
            remaining = self.backend.kubectl(
                "get", "pods", "-n", self.backend.namespace, "-l", f"app={self.deployment}", "-o", "json", timeout=15
            )
            if json.loads(remaining.stdout)["items"]:
                raise BenchmarkError(
                    "the previous Core supervisor did not stop; refusing overlapping observation sessions"
                )

    def start(self):
        """Use a distinct durable state file per new job; never reset an existing job's restart budget."""
        if self.mode == "disabled":
            return
        self.permissions(self.mode)
        values = [
            f"RECOVERY_JOB_ID={self.job_id}",
            f"RECOVERY_MODE={self.mode}",
            f"RECOVERY_STATE_FILE={self.state_file}",
            "RECOVERY_RESILIENCE=" + json.dumps(self.config.get("resilience", {})),
        ]
        self.backend.kubectl("set", "env", f"deployment/{self.deployment}", "-n", self.backend.namespace, *values)
        self.backend.kubectl("scale", f"deployment/{self.deployment}", "-n", self.backend.namespace, "--replicas=1")
        self.active = True
        self.backend.wait_rollouts((self.deployment,))
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            result = self.read_state(check=False)
            if result.returncode == 0 and result.stdout.strip():
                self.snapshot()
                return
            time.sleep(0.5)
        raise BenchmarkError("Core supervisor did not publish its job report within 60 seconds")

    def read_state(self, *, check=True):
        """Read this job's durable supervisor report with a bounded kubectl command."""
        return self.backend.kubectl(
            "exec",
            f"deployment/{self.deployment}",
            "-n",
            self.backend.namespace,
            "--",
            "cat",
            self.state_file,
            check=check,
            timeout=15,
        )

    def snapshot(self):
        """Collect sender-scoped evidence, verifying durable continuity after supervisor replacement.

        Core incident times remain supervisor-relative; application diagnostics retain their own clock basis.
        """
        pods = self.backend.kubectl(
            "get", "pods", "-n", self.backend.namespace, "-l", f"app={self.deployment}", "-o", "json", timeout=15
        )
        try:
            live = [pod for pod in json.loads(pods.stdout)["items"] if not pod["metadata"].get("deletionTimestamp")]
            if len(live) != 1:
                raise BenchmarkError("Core supervisor has no unique running instance")
            pod = live[0]
            session = (
                pod["metadata"]["uid"],
                sum(s.get("restartCount", 0) for s in pod["status"]["containerStatuses"]),
            )
            replaced = self.session is not None and session != self.session
            if replaced and not self.last_report.get("core_report", {}).get("context"):
                raise BenchmarkError("Core supervisor restarted; observer-session evidence cannot be pooled safely")
            report = normalize_report(json.loads(self.read_state().stdout), self.job_id, self.mode)
            requested = self.config.get("resilience", {})
            if any(report["resilience"].get(key) != value for key, value in requested.items()):
                raise BenchmarkError("Core supervisor lacks the requested resilience settings; rebuild and deploy Core")
            if replaced:
                verify_continuation(self.last_report["core_report"], report["core_report"])
            self.session = session
            allowed = set(self.config.get("workloads", []))
            if any(item["workload"] not in allowed for item in report["incidents"] + report["events"]):
                raise BenchmarkError("Core report contains a workload outside the application allowlist")
        except (ValueError, KeyError, TypeError) as exc:
            raise BenchmarkError(f"Core recovery report is missing or malformed: {exc}") from exc
        for event in report["events"]:
            self.events[event["id"]] = event
        logs = self.backend.collect_logs(self.config["started_at"])
        for text in logs.values():
            for event in parse_events(text, self.job_id):
                if event["kind"] not in {"checkpoint-saved", "checkpoint-restored", "resume-completed"}:
                    self.events[event["id"]] = event
        report["events"] = list(self.events.values())
        self.last_report = report
        return report

    def stop(self):
        """Scale supervision to zero when configured and mark the runtime inactive."""
        if self.deployment:
            self.backend.kubectl("scale", f"deployment/{self.deployment}", "-n", self.backend.namespace, "--replicas=0")
        self.active = False

    def permissions(self, mode):
        """Select Core's mode during experiment setup without expanding its workload allowlist."""
        role_name = self.config.get("supervisor_role", self.deployment)
        result = self.backend.kubectl("get", f"role/{role_name}", "-n", self.backend.namespace, "-o", "json")
        try:
            rules = json.loads(result.stdout)["rules"]
            dependencies = {
                name
                for names in self.config.get("deployment_resilience", self.config.get("resilience", {}))
                .get("dependencies", {})
                .values()
                for name in names
            } - set(self.config["workloads"])
            valid = (
                len(rules) == (2 if dependencies else 1)
                and rules[0]["apiGroups"] == ["apps"]
                and rules[0]["resources"] == ["deployments"]
                and set(rules[0]["resourceNames"]) == set(self.config["workloads"])
                and set(rules[0]["verbs"]) <= {"get", "patch"}
            )
            if dependencies:
                valid = valid and (
                    rules[1]["apiGroups"] == ["apps"]
                    and rules[1]["resources"] == ["deployments"]
                    and set(rules[1]["resourceNames"]) == dependencies
                    and rules[1]["verbs"] == ["get"]
                )
        except (ValueError, KeyError, TypeError) as exc:
            raise BenchmarkError("Core recovery Role is malformed") from exc
        if not valid:
            raise BenchmarkError("Core recovery Role does not match the scenario workload allowlist")
        verbs = ["get", "patch"] if mode == "controller-assisted" else ["get"]
        self.backend.kubectl(
            "patch",
            f"role/{role_name}",
            "-n",
            self.backend.namespace,
            "--type=json",
            "-p",
            json.dumps([{"op": "replace", "path": "/rules/0/verbs", "value": verbs}]),
        )

    def restore_defaults(self):
        """Restore supervisor permissions and default job settings, then restart it after workload rollouts."""
        if not self.deployment:
            return
        self.permissions(self.config.get("mode", "observe-only"))
        self.backend.kubectl(
            "set",
            "env",
            f"deployment/{self.deployment}",
            "-n",
            self.backend.namespace,
            f"RECOVERY_JOB_ID={self.config.get('job_id', 'demo')}",
            f"RECOVERY_MODE={self.config.get('mode', 'observe-only')}",
            "RECOVERY_STATE_FILE=/app/state/recovery.json",
            "RECOVERY_RESILIENCE="
            + json.dumps(self.config.get("default_resilience", self.config.get("resilience", {}))),
        )
        self.backend.wait_rollouts(tuple(self.config["workloads"]))
        self.backend.kubectl("scale", f"deployment/{self.deployment}", "-n", self.backend.namespace, "--replicas=1")
        self.backend.wait_rollouts((self.deployment,))


class CoreRecoveryMonitor:
    """Collect evidence from the independently running Core supervisor."""

    def __init__(self, runtime, _interval):
        self.runtime = runtime

    @property
    def last_report(self):
        """Expose the latest successfully collected evidence for reporting observation failures."""
        return self.runtime.last_report

    def start(self):
        self.runtime.start()

    def snapshot(self):
        return self.runtime.snapshot()

    def finish(self):
        """Take a final observation and stop supervision even when collection fails."""
        try:
            return self.snapshot()
        finally:
            self.runtime.stop()

"""Configure Core supervision for experiments and collect recovery reports."""

from __future__ import annotations

import json
import re
import time

from et_experiments.recovery.core import parse_events
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
    candidates = [
        item
        for item in measurements
        if record["onset"] <= item["observed_at"] < end
        and item.get("instance")
        and item["instance"] not in old_instances
        and item["progress"] >= record["baseline"]
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
    if raw.get("mode") != mode or raw.get("clock_basis") != "observer-monotonic-seconds":
        raise BenchmarkError("Core recovery report has an unexpected mode or clock basis")
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
        "metrics": {
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
            if self.session is not None and session != self.session:
                raise BenchmarkError("Core supervisor restarted; observer-session evidence cannot be pooled safely")
            self.session = session
            report = normalize_report(json.loads(self.read_state().stdout), self.job_id, self.mode)
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
        if self.deployment:
            self.backend.kubectl("scale", f"deployment/{self.deployment}", "-n", self.backend.namespace, "--replicas=0")
        self.active = False

    def permissions(self, mode):
        """Select Core's mode during experiment setup without expanding its workload allowlist."""
        role_name = self.config.get("supervisor_role", self.deployment)
        result = self.backend.kubectl("get", f"role/{role_name}", "-n", self.backend.namespace, "-o", "json")
        try:
            rules = json.loads(result.stdout)["rules"]
            valid = (
                len(rules) == 1
                and rules[0]["apiGroups"] == ["apps"]
                and rules[0]["resources"] == ["deployments"]
                and set(rules[0]["resourceNames"]) == set(self.config["workloads"])
                and set(rules[0]["verbs"]) <= {"get", "patch"}
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
        )
        self.backend.wait_rollouts(tuple(self.config["workloads"]))
        self.backend.kubectl("scale", f"deployment/{self.deployment}", "-n", self.backend.namespace, "--replicas=1")


class CoreRecoveryMonitor:
    """Collect evidence from the independently running Core supervisor."""

    def __init__(self, runtime, _interval):
        self.runtime = runtime

    @property
    def last_report(self):
        return self.runtime.last_report

    def start(self):
        self.runtime.start()

    def snapshot(self):
        return self.runtime.snapshot()

    def finish(self):
        try:
            return self.snapshot()
        finally:
            self.runtime.stop()

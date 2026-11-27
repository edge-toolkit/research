"""Bounded hub outages and rejected restart commands for Harbour-FL recovery evaluations."""

import json
import subprocess
import time
from urllib.parse import urlparse

from et_experiments.benchmarking.faults import KubernetesFaults
from et_experiments.recovery.supervision import verify_continuation
from et_experiments.utils import BenchmarkError, iso, utc_now


def validate_fault(fault, workers):
    """Reject unsupported targets and durations before a recovery experiment changes deployments."""
    kind = fault.get("type")
    if kind not in {"hub_outage", "restart_command_rejected"}:
        return
    targets = {"ws-server"} if kind == "hub_outage" else set(workers)
    if fault.get("target") not in targets:
        raise BenchmarkError("Recovery fault target is outside the deployment allowlist")
    duration = fault.get("duration_seconds", 30 if kind == "hub_outage" else 70)
    if type(duration) not in (int, float) or not 1 <= duration <= 120:
        raise BenchmarkError("Recovery fault duration must be between 1 and 120 seconds")
    return duration


class HarbourRecoveryFaults(KubernetesFaults):
    """Reuse shared infrastructure faults and restore every temporary deployment/RBAC change."""

    def __init__(self, backend, application):
        super().__init__(backend, application)
        self._pending = []

    def restore_recovery(self):
        """Retry incomplete cleanup during engine finalization; unresolved changes must fail cleanup."""
        errors = []
        for operation in self._pending:
            try:
                operation()
            except (BenchmarkError, OSError, subprocess.TimeoutExpired) as error:
                errors.append(str(error))
        if errors:
            raise BenchmarkError("Recovery fault cleanup failed: " + "; ".join(errors))
        self._pending.clear()

    def inject_fault(self, fault, job_id):
        kind = fault.get("type")
        if kind not in {"hub_outage", "restart_command_rejected"}:
            return super().inject_fault(fault, job_id)
        self.restore_recovery()
        duration = validate_fault(fault, self.application.config.workers)
        target = fault["target"]
        namespace = self.application.config.namespace

        def call(*args, **options):
            options.setdefault("timeout", 60)
            return self.backend.kubectl("-n", namespace, *args, **options).stdout

        original = json.loads(call("get", "deployment/" + target, "-o", "json"))
        if original["spec"].get("replicas", 1) != 1:
            raise BenchmarkError("Recovery fault requires exactly one original target replica")
        if kind == "hub_outage":
            spec = original["spec"]["template"]["spec"]
            container = spec["containers"][0]
            env = {item["name"]: item.get("value", "") for item in container.get("env", [])}
            url = env.get("STORAGE_URL", "file:///app/storage")
            if url.startswith("file:"):
                path = urlparse(url).path
                volumes = {volume["name"]: volume for volume in spec.get("volumes", [])}
                persistent = any(
                    (path == mount["mountPath"] or path.startswith(mount["mountPath"].rstrip("/") + "/"))
                    and any(key in volumes.get(mount["name"], {}) for key in ("persistentVolumeClaim", "hostPath"))
                    for mount in container.get("volumeMounts", [])
                )
                if not persistent:
                    raise BenchmarkError("Hub outage requires persistent checkpoint storage")
            elif not url.startswith(("s3:", "s3+http:", "s3+https:")):
                raise BenchmarkError("Hub storage persistence is not established")
        result = {"type": kind, "target": target, "started_at": iso(utc_now()), "actor": "benchmark-controller"}
        role = self.application.recovery_config().get("supervisor_role", "recovery-supervisor")
        supervisor = self.application.recovery_config().get("supervisor_deployment", "recovery-supervisor")
        rules = None
        if kind == "restart_command_rejected":
            rules = json.loads(call("get", "role/" + role, "-o", "json"))["rules"]
            changed = [
                i
                for i, rule in enumerate(rules)
                if "deployments" in rule.get("resources", [])
                and "patch" in rule.get("verbs", [])
                and (not rule.get("resourceNames") or target in rule["resourceNames"])
            ]
            if not changed:
                raise BenchmarkError("Supervisor has no restart permission to reject")
        failure = None
        try:
            if rules is None:
                call("scale", "deployment/" + target, "--replicas=0")
                call("wait", "--for=delete", "pod", "-l", "app=" + target, "--timeout=90s", timeout=100)
                result["active_at"] = iso(utc_now())
                time.sleep(duration)
            else:
                call(
                    "patch",
                    "role/" + role,
                    "--type=json",
                    "-p",
                    json.dumps(
                        [
                            {
                                "op": "replace",
                                "path": f"/rules/{i}/verbs",
                                "value": [v for v in rules[i]["verbs"] if v != "patch"],
                            }
                            for i in changed
                        ]
                    ),
                )
                result.update(self.scheduling.inject(target, duration, job_id))

                def state():
                    return json.loads(
                        call("exec", "deployment/" + supervisor, "--", "cat", f"/app/state/{job_id}.json", timeout=15)
                    )

                before = state()
                consumed = before.get("restart_budget", {}).get("restarts", {}).get(target, 0)
                rejected = any(
                    action.get("kind") == "restart-failed"
                    for incident in before.get("incidents", [])
                    if incident.get("workload") == target
                    for action in incident.get("actions", [])
                )
                if consumed != 1 or not rejected:
                    raise BenchmarkError("Rejected restart and consumed budget were not observed")
                call("rollout", "restart", "deployment/" + supervisor)
                call("rollout", "status", "deployment/" + supervisor, "--timeout=180s", timeout=190)
                deadline = time.monotonic() + 30
                while True:
                    after = state()
                    if after.get("context", {}).get("restored_sessions", 0) > before.get("context", {}).get(
                        "restored_sessions", 0
                    ):
                        break
                    if time.monotonic() >= deadline:
                        raise BenchmarkError("Supervisor replacement did not expose restored context")
                    time.sleep(1)
                verify_continuation(before, after)
                if after.get("restart_budget", {}).get("restarts", {}).get(target) != consumed:
                    raise BenchmarkError("Restart budget changed across supervisor replacement")
                result.update(budget_before=before, budget_after=after, durable_budget_verified=True)
        except BaseException as error:
            failure = error
            raise
        finally:
            operations = (
                [lambda: call("scale", "deployment/" + target, "--replicas=1")]
                if rules is None
                else [
                    lambda: call(
                        "patch",
                        "role/" + role,
                        "--type=json",
                        "-p",
                        json.dumps([{"op": "replace", "path": "/rules", "value": rules}]),
                    ),
                    self.scheduling.restore,
                ]
            )
            operations.append(lambda: call("rollout", "status", "deployment/" + target, "--timeout=300s", timeout=310))
            self._pending = operations
            try:
                self.restore_recovery()
            except BenchmarkError as cleanup:
                if failure is not None:
                    raise BenchmarkError(f"{type(failure).__name__}: {failure}; cleanup failed: {cleanup}") from failure
                raise
            result["released_at"] = iso(utc_now())
        return {**result, "status": "restored"}

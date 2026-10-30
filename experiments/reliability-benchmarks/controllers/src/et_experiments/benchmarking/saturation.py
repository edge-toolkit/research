"""Client-scoped pressure profiles, temporary resource envelopes, and bounded cleanup."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import secrets
import subprocess
from pathlib import Path

from et_experiments.utils import BenchmarkError, iso, percentile, utc_now

MEMORY_TYPE = "client_memory_saturation"
NETWORK_TYPE = "client_network_saturation"


def number(fault, key, default, minimum, maximum):
    """Reject non-finite or out-of-range pressure settings before resource changes."""
    try:
        value = float(fault.get(key, default))
    except (ValueError, TypeError) as error:
        raise BenchmarkError(f"invalid saturation setting: {key}") from error
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise BenchmarkError(f"{key} must be between {minimum} and {maximum}")
    return value


def profile_options(fault):
    """Validate pressure levels, stage durations and resource bounds before running a profile."""
    memory = fault["type"] == MEMORY_TYPE
    levels = fault.get("levels", [0.6, 0.75, 0.9] if memory else [0.25, 0.5, 0.85, 1.25, 1.5])
    if not isinstance(levels, list) or not 2 <= len(levels) <= 6:
        raise BenchmarkError("saturation profiles require two to six increasing levels")
    parsed = [number({"level": value}, "level", 0, 0.1, 0.9 if memory else 2) for value in levels]
    if parsed != sorted(set(parsed)):
        raise BenchmarkError("saturation levels must be strictly increasing")
    options = {"levels": parsed, "hold_seconds": number(fault, "hold_seconds", 10 if memory else 5, 1, 15)}
    if memory:
        options.update(
            max_extra_memory_mib=number(fault, "max_extra_memory_mib", 512, 16, 1024),
            reserve_mib=number(fault, "reserve_mib", 64, 32, 256),
            release_seconds=2,
        )
    else:
        options["rate_kbit"] = number(fault, "rate_kbit", 1024, 128, 4096)
    return options


def worker_source(name):
    """Read a packaged standalone pressure helper for execution inside the target container."""
    return Path(__file__).with_name(name).read_text()


def annotate_stages(report):
    """Group timestamped application completions into pressure windows, leaving absent evidence unmeasured."""
    units = report.get("workload", {}).get("units", [])
    timed = []
    for unit in units:
        if not isinstance(unit.get("completed_at"), str):
            continue
        try:
            at = dt.datetime.fromisoformat(unit["completed_at"].replace("Z", "+00:00"))
            if at.tzinfo is not None:
                timed.append((at, unit))
        except (KeyError, ValueError, TypeError):
            continue
    for stage in report.get("fault", {}).get("stages", []):
        start = dt.datetime.fromisoformat(stage["started_at"].replace("Z", "+00:00"))
        end = dt.datetime.fromisoformat(stage["finished_at"].replace("Z", "+00:00"))
        completed = [unit for at, unit in timed if start <= at < end]
        durations = [
            unit["duration_seconds"] for unit in completed if isinstance(unit.get("duration_seconds"), (int, float))
        ]
        stage["application"] = {
            "completed_units": len(completed) if timed else None,
            "completed_units_per_second": len(completed) / max(0.001, (end - start).total_seconds()) if timed else None,
            "unit_duration_p95_seconds": percentile(durations, 0.95),
            "latency_sample_count": len(durations),
        }


class SaturationFaults:
    """Prepare client pressure profiles and track resource settings needed for restoration."""

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.original_resources = {}

    def validate(self, fault):
        """Require an allowlisted worker target and bounded, ordered profile settings."""
        if fault.get("target") not in self.config.workers:
            raise BenchmarkError("saturation requires an allowlisted worker target")
        return profile_options(fault)

    def prepare(self, fault):
        """Capture original container resources and optionally apply the reference memory limit."""
        if fault.get("type") not in {MEMORY_TYPE, NETWORK_TYPE}:
            return
        self.validate(fault)
        if fault["type"] != MEMORY_TYPE or "memory_limit_mib" not in fault:
            return
        limit = int(number(fault, "memory_limit_mib", 768, 256, 2048))
        target = fault["target"]
        result = self.backend.kubectl("get", f"deployment/{target}", "-n", self.backend.namespace, "-o", "json")
        containers = json.loads(result.stdout)["spec"]["template"]["spec"]["containers"]
        index = next((index for index, item in enumerate(containers) if item["name"] == target), None)
        if index is None:
            raise BenchmarkError("saturation target has no matching workload container")
        original = containers[index].get("resources", {})
        self.original_resources.setdefault(target, (index, original))
        resources = json.loads(json.dumps(original))
        resources.setdefault("limits", {})["memory"] = f"{limit}Mi"
        resources.setdefault("requests", {})["memory"] = f"{min(64, limit)}Mi"
        self.set_resources(target, index, resources)

    def set_resources(self, target, index, resources):
        """Replace the selected deployment container's resource settings with a JSON patch."""
        self.backend.kubectl(
            "patch",
            f"deployment/{target}",
            "-n",
            self.backend.namespace,
            "--type=json",
            "-p",
            json.dumps(
                [{"op": "add", "path": f"/spec/template/spec/containers/{index}/resources", "value": resources}]
            ),
        )

    def restore(self):
        """Restore saved resource settings, retaining entries that fail so restoration can be retried."""
        for target, (index, resources) in list(self.original_resources.items()):
            self.set_resources(target, index, resources)
            del self.original_resources[target]

    def execute(self, target, script, options, mode=None):
        """Run a pressure helper with a profile-based deadline and require a recognised JSON outcome."""
        args = [
            "exec",
            "-n",
            self.backend.namespace,
            f"deployment/{target}",
            "-c",
            target,
            "--",
            "/opt/cpython/bin/python3",
            "-c",
            script,
        ]
        if mode:
            args.append(mode)
        result = self.backend.kubectl(
            *args,
            json.dumps(options),
            timeout=len(options["levels"]) * (options["hold_seconds"] + 1) + 30,
            check=False,
        )
        if result.returncode:
            raise BenchmarkError(f"saturation helper failed: {result.stderr[-3000:]}")
        try:
            report = json.loads(result.stdout)
        except (ValueError, TypeError) as error:
            raise BenchmarkError("saturation helper did not return a complete measurement") from error
        if not isinstance(report, dict) or report.get("status") not in {"restored", "not-observed"}:
            raise BenchmarkError("invalid saturation measurement")
        return report

    def inject(self, fault, job_id):
        """Dispatch validated memory or network pressure and return the helper's measured result."""
        options = self.validate(fault)
        if fault["type"] == MEMORY_TYPE:
            return self.execute(fault["target"], worker_source("memory_worker.py"), options)
        return self.network(fault["target"], options, job_id)

    def receiver(self, name, options):
        """Build a private UDP receiver Pod and Service with resource limits and a bounded lifetime."""
        labels = {"app": "et-benchmark-saturation", "benchmark-receiver": name}
        container = {
            "name": "receiver",
            "image": self.config.stress_image,
            "imagePullPolicy": "IfNotPresent",
            "command": [
                "/opt/cpython/bin/python3",
                "-c",
                worker_source("network_worker.py"),
                "serve",
                json.dumps(options),
            ],
            "resources": {"requests": {"cpu": "10m", "memory": "16Mi"}, "limits": {"cpu": "500m", "memory": "64Mi"}},
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "readOnlyRootFilesystem": True,
                "capabilities": {"drop": ["ALL"]},
            },
        }
        container["readinessProbe"] = {
            "exec": {
                "command": [
                    "/opt/cpython/bin/python3",
                    "-c",
                    worker_source("network_worker.py"),
                    "probe",
                    json.dumps(options),
                ]
            },
            "periodSeconds": 2,
            "timeoutSeconds": 2,
        }
        metadata = {"name": name, "namespace": self.backend.namespace, "labels": labels}
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": metadata,
            "spec": {
                "restartPolicy": "Never",
                "activeDeadlineSeconds": options["lifetime_seconds"] + 30,
                "securityContext": {
                    "runAsUser": 10001,
                    "runAsGroup": 10001,
                    "runAsNonRoot": True,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "containers": [container],
            },
        }
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": metadata,
            "spec": {
                "selector": {"benchmark-receiver": name},
                "ports": [{"port": options["port"], "targetPort": options["port"], "protocol": "UDP"}],
            },
        }
        return {"apiVersion": "v1", "kind": "List", "items": [pod, service]}

    def network(self, target, options, job_id):
        """Apply a client egress bottleneck and remove its qdisc and private receiver on exit.

        Refuse pre-existing traffic-control policies. Record restoration failures separately from measured pressure.
        """
        name = "et-net-" + hashlib.sha256(job_id.encode()).hexdigest()[:12]
        lifetime = math.ceil(len(options["levels"]) * (options["hold_seconds"] + 1)) + 150
        options = {**options, "host": name, "port": 19090, "token": secrets.token_hex(16), "lifetime_seconds": lifetime}
        script = worker_source("network_worker.py")
        applied = False
        report = {}
        try:
            self.backend.kubectl("apply", "-f", "-", input_text=json.dumps(self.receiver(name, options)))
            self.backend.kubectl(
                "wait",
                f"pod/{name}",
                "-n",
                self.backend.namespace,
                "--for=condition=Ready",
                "--timeout=60s",
                timeout=70,
            )
            previous = self.netem(target, "show", "dev", "eth0")
            if "noqueue" not in previous.stdout:
                raise BenchmarkError("refusing to replace a pre-existing client traffic-control policy")
            reachable = self.backend.kubectl(
                "exec",
                "-n",
                self.backend.namespace,
                f"deployment/{target}",
                "-c",
                target,
                "--",
                "/opt/cpython/bin/python3",
                "-c",
                script,
                "path-probe",
                json.dumps(options),
                timeout=15,
                check=False,
            )
            if reachable.returncode:
                raise BenchmarkError(
                    f"private network receiver is unreachable before pressure: {reachable.stderr[-1500:]}"
                )
            applied = True
            self.netem(
                target,
                "replace",
                "dev",
                "eth0",
                "root",
                "netem",
                "limit",
                "64",
                "rate",
                f"{options['rate_kbit']:g}kbit",
            )
            report = self.execute(target, script, options, "run")
            if report.get("pressure_classification") == "probe-path-error":
                try:
                    details = self.backend.kubectl(
                        "get", f"pod/{name}", "-n", self.backend.namespace, "-o", "json", check=False, timeout=10
                    )
                    status = json.loads(details.stdout).get("status", {})
                    report["receiver_status"] = {
                        "phase": status.get("phase"),
                        "reason": status.get("reason"),
                        "containers": [
                            {"name": item.get("name"), "restarts": item.get("restartCount"), "state": item.get("state")}
                            for item in status.get("containerStatuses", [])
                        ],
                    }
                except (BenchmarkError, subprocess.TimeoutExpired, ValueError, TypeError):
                    report["receiver_status"] = {"status": "unavailable"}
        finally:
            try:
                if applied:
                    removed = self.netem(target, "del", "dev", "eth0", "root", check=False)
                    report["released_at"] = iso(utc_now())
                    if removed.returncode:
                        report.update(status="restore-failed", restore_error=removed.stderr)
            finally:
                removed_receiver = self.backend.kubectl(
                    "delete",
                    f"pod/{name}",
                    f"service/{name}",
                    "-n",
                    self.backend.namespace,
                    "--ignore-not-found=true",
                    "--wait=false",
                    check=False,
                    timeout=15,
                )
                if removed_receiver.returncode:
                    report.update(status="restore-failed", restore_error=removed_receiver.stderr)
        report["configured_rate_kbit"] = options["rate_kbit"]
        return report

    def netem(self, target, *args, **kwargs):
        """Run a bounded traffic-control command inside the worker's isolated network helper."""
        return self.backend.kubectl(
            "exec",
            "-n",
            self.backend.namespace,
            f"deployment/{target}",
            "-c",
            "benchmark-netem",
            "--",
            "tc",
            "qdisc",
            *args,
            timeout=15,
            **kwargs,
        )

"""Infrastructure fault injection, bounded targets, and teardown."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from typing import Any

from et_experiments.utils import BenchmarkError, iso, progress, utc_now

from .gpu import inject_gpu_failure
from .saturation import MEMORY_TYPE, NETWORK_TYPE, SaturationFaults

NETWORK_FAULT_TYPES = {"network_latency", "network_bandwidth", "network_loss", "network_partition", "degraded_network"}
NETWORK_FAULT_SIDECAR = "benchmark-netem"
NETWORK_FAULT_IMAGE = "nicolaka/netshoot:v0.13"


def bounded_number(fault: dict[str, Any], key: str, minimum: float, maximum: float, default: float) -> float:
    try:
        value = float(fault.get(key, default))
    except (TypeError, ValueError) as error:
        raise BenchmarkError(f"invalid {key}: {fault.get(key)!r}") from error
    return max(minimum, min(value, maximum))


def memory_pressure_manifest(job_id: str, memory_mib: int, duration: int, image: str, namespace: str) -> dict[str, Any]:
    allocation = max(64, min(int(memory_mib), 2048))
    limit = allocation + 128
    script = (
        "import sys,time\n"
        "size=int(sys.argv[1])*1024*1024\n"
        "block=bytearray(size)\n"
        "for offset in range(0,size,4096): block[offset]=1\n"
        "print(f'allocated_and_touched_bytes={len(block)}',flush=True)\n"
        "time.sleep(int(sys.argv[2]))\n"
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"et-bench-memory-{job_id[-8:]}",
            "namespace": namespace,
            "labels": {"app": "et-benchmark", "benchmark-job": job_id, "benchmark-fault": "memory-pressure"},
        },
        "spec": {
            "ttlSecondsAfterFinished": 120,
            "template": {
                "metadata": {"labels": {"app": "et-benchmark", "benchmark-job": job_id}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "memory-pressure",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/opt/cpython/bin/python3", "-c"],
                            "args": [script, str(allocation), str(duration)],
                            "resources": {
                                "requests": {"cpu": "10m", "memory": "32Mi"},
                                "limits": {"cpu": "100m", "memory": f"{limit}Mi"},
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "runAsNonRoot": True,
                                "capabilities": {"drop": ["ALL"]},
                            },
                        }
                    ],
                    "securityContext": {
                        "runAsUser": 10001,
                        "runAsGroup": 10001,
                        "runAsNonRoot": True,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                },
            },
        },
    }


def network_fault_arguments(fault: dict[str, Any]) -> tuple[str, ...]:
    """Translate an allowlisted network fault into shell-free tc-netem arguments."""
    fault_type = str(fault.get("type"))
    if fault_type == "network_latency":
        latency = bounded_number(fault, "latency_ms", 1, 5000, 250)
        jitter = bounded_number(fault, "jitter_ms", 0, latency, 50)
        return ("delay", f"{latency:g}ms", f"{jitter:g}ms", "distribution", "normal")
    if fault_type == "network_bandwidth":
        rate = int(bounded_number(fault, "rate_kbit", 8, 1_000_000, 256))
        return ("rate", f"{rate}kbit")
    if fault_type == "network_loss":
        loss = bounded_number(fault, "loss_percent", 0.1, 99, 10)
        return ("loss", f"{loss:g}%")
    if fault_type == "network_partition":
        return ("loss", "100%")
    if fault_type == "degraded_network":
        latency = bounded_number(fault, "latency_ms", 1, 5000, 200)
        jitter = bounded_number(fault, "jitter_ms", 0, latency, 50)
        loss = bounded_number(fault, "loss_percent", 0.1, 99, 5)
        rate = int(bounded_number(fault, "rate_kbit", 8, 1_000_000, 512))
        return (
            "delay",
            f"{latency:g}ms",
            f"{jitter:g}ms",
            "distribution",
            "normal",
            "loss",
            f"{loss:g}%",
            "rate",
            f"{rate}kbit",
        )
    raise BenchmarkError(f"unsupported network fault type: {fault_type}")


class KubernetesFaults:
    def __init__(self, backend, application):
        self.backend, self.application = backend, application
        self.safe_targets = set(application.config.observed)
        self.interventions: list[dict[str, Any]] = []
        self.saturation = SaturationFaults(backend, application.config)

    def prepare_fault(self, fault, started):
        self.saturation.prepare(fault)
        self.prepare_network_fault(fault, started)

    def restore_saturation(self):
        self.saturation.restore()

    def prepare_network_fault(self, fault: dict[str, Any], started: dt.datetime) -> None:
        if fault.get("type") in NETWORK_FAULT_TYPES | {NETWORK_TYPE}:
            target = str(fault.get("target", ""))
            if target not in set(self.application.config.workers):
                raise BenchmarkError(f"network faults are not allowed for {target}")
            progress(started, f"Adding an isolated network-fault helper to {target}")
            self.remove_network_fault_sidecar(target)
            sidecar = {
                "name": NETWORK_FAULT_SIDECAR,
                "image": NETWORK_FAULT_IMAGE,
                "imagePullPolicy": "IfNotPresent",
                "command": ["sleep", "infinity"],
                "resources": {"requests": {"cpu": "5m", "memory": "16Mi"}, "limits": {"cpu": "100m", "memory": "64Mi"}},
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    # tc needs CAP_NET_ADMIN in the initial user namespace. The
                    # application Pod defaults to uid 10001, so override the identity
                    # for this short-lived helper only; it is not privileged and
                    # receives no other capabilities.
                    "runAsUser": 0,
                    "runAsGroup": 0,
                    "runAsNonRoot": False,
                    "capabilities": {"drop": ["ALL"], "add": ["NET_ADMIN"]},
                },
            }
            self.backend.kubectl(
                "patch",
                f"deployment/{target}",
                "-n",
                self.backend.namespace,
                "--type=json",
                "-p",
                json.dumps([{"op": "add", "path": "/spec/template/spec/containers/-", "value": sidecar}]),
            )

    def inject_fault(self, fault: dict[str, Any], job_id: str) -> dict[str, Any]:
        fault_type = fault.get("type", "none")
        started = utc_now()
        result: dict[str, Any] = {"type": fault_type, "started_at": iso(started), "target": fault.get("target")}
        if fault_type == "none":
            result["status"] = "not-applicable"
        elif fault_type == "gpu_failure":
            result.update(inject_gpu_failure(self.backend, self.application.config.workers, fault, job_id))
        elif fault_type in {MEMORY_TYPE, NETWORK_TYPE}:
            result.update(self.saturation.inject(fault, job_id))
        elif fault_type == "delete_pod":
            target = str(fault["target"])
            if target not in self.safe_targets:
                raise BenchmarkError(f"unsafe fault target: {target}")
            self.backend.kubectl("delete", "pod", "-n", self.backend.namespace, "-l", f"app={target}", "--wait=false")
            result["status"] = "injected"
        elif fault_type == "temporary_scale_down":
            target = str(fault["target"])
            if target not in set(self.application.config.workers):
                raise BenchmarkError(f"temporary scale-down is not allowed for {target}")
            self.backend.kubectl("scale", f"deployment/{target}", "-n", self.backend.namespace, "--replicas=0")
            time.sleep(float(fault.get("duration_seconds", 15)))
            self.backend.kubectl("scale", f"deployment/{target}", "-n", self.backend.namespace, "--replicas=1")
            result["status"] = "restored"
        elif fault_type == "rollout_restart":
            target = str(fault["target"])
            if target not in self.safe_targets:
                raise BenchmarkError(f"unsafe fault target: {target}")
            self.backend.kubectl("rollout", "restart", f"deployment/{target}", "-n", self.backend.namespace)
            result["status"] = "injected"
        elif fault_type == "cpu_stress":
            workers = max(1, min(int(fault.get("workers", 2)), 8))
            duration = max(1, min(int(fault.get("duration_seconds", 30)), 300))
            manifest = {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "name": f"et-bench-stress-{job_id[-8:]}",
                    "namespace": self.backend.namespace,
                    "labels": {"app": "et-benchmark", "benchmark-job": job_id},
                },
                "spec": {
                    "ttlSecondsAfterFinished": 120,
                    "template": {
                        "metadata": {"labels": {"app": "et-benchmark", "benchmark-job": job_id}},
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "cpu-stress",
                                    "image": self.application.config.stress_image,
                                    "imagePullPolicy": "IfNotPresent",
                                    "command": ["/opt/cpython/bin/python3", "-c"],
                                    "args": [
                                        (
                                            "import multiprocessing as m,time\n"
                                            "end=time.monotonic()+int(__import__('sys').argv[1])\n"
                                            "def burn():\n while time.monotonic()<end: pass\n"
                                            "ps=[m.Process(target=burn) "
                                            "for _ in range(int(__import__('sys').argv[2]))]\n"
                                            "[p.start() for p in ps]\n[p.join() for p in ps]"
                                        ),
                                        str(duration),
                                        str(workers),
                                    ],
                                    "resources": {
                                        "requests": {"cpu": "100m", "memory": "64Mi"},
                                        "limits": {"cpu": str(workers), "memory": "256Mi"},
                                    },
                                    "securityContext": {
                                        "allowPrivilegeEscalation": False,
                                        "readOnlyRootFilesystem": True,
                                        "runAsNonRoot": True,
                                        "capabilities": {"drop": ["ALL"]},
                                    },
                                }
                            ],
                            "securityContext": {
                                "runAsUser": 10001,
                                "runAsGroup": 10001,
                                "runAsNonRoot": True,
                                "seccompProfile": {"type": "RuntimeDefault"},
                            },
                        },
                    },
                },
            }
            self.backend.kubectl("apply", "-f", "-", input_text=json.dumps(manifest))
            result.update({"status": "injected", "workers": workers, "duration_seconds": duration})
        elif fault_type == "memory_pressure":
            memory_mib = int(bounded_number(fault, "memory_mib", 64, 2048, 512))
            duration = max(1, min(int(fault.get("duration_seconds", 30)), 300))
            manifest = memory_pressure_manifest(
                job_id, memory_mib, duration, self.application.config.stress_image, self.backend.namespace
            )
            self.backend.kubectl("apply", "-f", "-", input_text=json.dumps(manifest))
            result.update({"status": "injected", "memory_mib": memory_mib, "duration_seconds": duration})
        elif fault_type == "worker_oom":
            target = str(fault.get("target", ""))
            if target not in set(self.application.config.workers):
                raise BenchmarkError(f"worker OOM is not allowed for {target}")
            limit_mib = int(bounded_number(fault, "memory_limit_mib", 32, 1024, 96))
            detection_timeout = max(10, min(int(fault.get("detection_timeout_seconds", 90)), 180))
            deployment = self.backend.kubectl("get", f"deployment/{target}", "-n", self.backend.namespace, "-o", "json")
            containers = json.loads(deployment.stdout)["spec"]["template"]["spec"]["containers"]
            container_index = next(index for index, item in enumerate(containers) if item.get("name") == target)
            original_resources = containers[container_index].get("resources", {})
            limited_resources = json.loads(json.dumps(original_resources))
            limited_resources.setdefault("limits", {})["memory"] = f"{limit_mib}Mi"
            # Kubernetes requires request <= limit. Keep a small nonzero request
            # for scheduling while the deliberately low limit triggers the fault.
            limited_resources.setdefault("requests", {})["memory"] = f"{min(limit_mib, 64)}Mi"
            detected_at: dt.datetime | None = None
            termination: dict[str, Any] | None = None
            try:
                self.backend.kubectl(
                    "patch",
                    f"deployment/{target}",
                    "-n",
                    self.backend.namespace,
                    "--type=json",
                    "-p",
                    json.dumps(
                        [
                            {
                                "op": "replace",
                                "path": f"/spec/template/spec/containers/{container_index}/resources",
                                "value": limited_resources,
                            }
                        ]
                    ),
                )
                deadline = time.monotonic() + detection_timeout
                while time.monotonic() < deadline and detected_at is None:
                    pods = self.backend.kubectl(
                        "get", "pods", "-n", self.backend.namespace, "-l", f"app={target}", "-o", "json", check=False
                    )
                    if pods.returncode == 0:
                        for pod in json.loads(pods.stdout).get("items", []):
                            for status in pod.get("status", {}).get("containerStatuses", []):
                                if status.get("name") != target:
                                    continue
                                terminated = status.get("state", {}).get("terminated") or status.get(
                                    "lastState", {}
                                ).get("terminated")
                                if terminated and terminated.get("reason") == "OOMKilled":
                                    detected_at = utc_now()
                                    termination = {
                                        "pod": pod["metadata"]["name"],
                                        "reason": terminated.get("reason"),
                                        "exit_code": terminated.get("exitCode"),
                                        "finished_at": terminated.get("finishedAt"),
                                    }
                                    break
                            if detected_at:
                                break
                    if detected_at is None:
                        time.sleep(1)
            finally:
                restored = self.backend.kubectl(
                    "patch",
                    f"deployment/{target}",
                    "-n",
                    self.backend.namespace,
                    "--type=json",
                    "-p",
                    json.dumps(
                        [
                            {
                                "op": "replace",
                                "path": f"/spec/template/spec/containers/{container_index}/resources",
                                "value": original_resources,
                            }
                        ]
                    ),
                    check=False,
                )
                if restored.returncode == 0:
                    self.backend.kubectl(
                        "rollout",
                        "status",
                        f"deployment/{target}",
                        "-n",
                        self.backend.namespace,
                        "--timeout=180s",
                        check=False,
                    )
            result.update(
                {
                    "status": "restored"
                    if detected_at and restored.returncode == 0
                    else "not-observed"
                    if not detected_at
                    else "restore-failed",
                    "memory_limit_mib": limit_mib,
                    "detection_timeout_seconds": detection_timeout,
                    "injection_to_oom_observation_seconds": (detected_at - started).total_seconds()
                    if detected_at
                    else None,
                    "failure_detection_seconds": (
                        (
                            detected_at - dt.datetime.fromisoformat(termination["finished_at"].replace("Z", "+00:00"))
                        ).total_seconds()
                        if detected_at and termination and termination.get("finished_at")
                        else None
                    ),
                    "termination": termination,
                    "restore_error": restored.stderr.strip() if restored.returncode else None,
                }
            )
        elif fault_type == "mlflow_outage":
            if self.application.tracking is None:
                raise BenchmarkError("tracking outage requires tracking configuration")
            duration = max(1, min(int(fault.get("duration_seconds", 15)), 300))
            self.backend.kubectl(
                "scale",
                f"deployment/{self.application.tracking.deployment}",
                "-n",
                self.backend.namespace,
                "--replicas=0",
            )
            time.sleep(duration)
            self.backend.kubectl(
                "scale",
                f"deployment/{self.application.tracking.deployment}",
                "-n",
                self.backend.namespace,
                "--replicas=1",
            )
            self.backend.kubectl(
                "rollout",
                "status",
                f"deployment/{self.application.tracking.deployment}",
                "-n",
                self.backend.namespace,
                "--timeout=180s",
            )
            if fault.get("restart_bridge", False):
                intervention = {
                    "at": utc_now().timestamp(),
                    "type": "bridge-restart",
                    "actor": "benchmark",
                    "workload": self.application.tracking.bridge_deployment,
                    "purpose": "recovery-assistance",
                    "status": "attempted",
                }
                self.interventions.append(intervention)
                try:
                    self.backend.kubectl(
                        "rollout",
                        "restart",
                        f"deployment/{self.application.tracking.bridge_deployment}",
                        "-n",
                        self.backend.namespace,
                    )
                except (BenchmarkError, subprocess.TimeoutExpired, OSError):
                    intervention["status"] = "failed-or-unknown"
                    raise
                intervention["status"] = "requested"
                self.backend.kubectl(
                    "rollout",
                    "status",
                    f"deployment/{self.application.tracking.bridge_deployment}",
                    "-n",
                    self.backend.namespace,
                    "--timeout=180s",
                )
            result.update({"status": "restored", "duration_seconds": duration})
        elif fault_type in NETWORK_FAULT_TYPES:
            target = str(fault.get("target", ""))
            if target not in set(self.application.config.workers):
                raise BenchmarkError(f"network faults are not allowed for {target}")
            duration = max(1, min(int(fault.get("duration_seconds", 30)), 300))
            netem = network_fault_arguments(fault)
            command = ("tc", "qdisc", "replace", "dev", "eth0", "root", "netem", *netem)
            applied = self.backend.kubectl(
                "exec",
                "-n",
                self.backend.namespace,
                f"deployment/{target}",
                "-c",
                NETWORK_FAULT_SIDECAR,
                "--",
                *command,
            )
            observed = self.backend.kubectl(
                "exec",
                "-n",
                self.backend.namespace,
                f"deployment/{target}",
                "-c",
                NETWORK_FAULT_SIDECAR,
                "--",
                "tc",
                "qdisc",
                "show",
                "dev",
                "eth0",
            )
            try:
                time.sleep(duration)
            finally:
                removed = self.backend.kubectl(
                    "exec",
                    "-n",
                    self.backend.namespace,
                    f"deployment/{target}",
                    "-c",
                    NETWORK_FAULT_SIDECAR,
                    "--",
                    "tc",
                    "qdisc",
                    "del",
                    "dev",
                    "eth0",
                    "root",
                    check=False,
                )
            result.update(
                {
                    "status": "restored" if removed.returncode == 0 else "restore-failed",
                    "duration_seconds": duration,
                    "netem_arguments": list(netem),
                    "apply_output": applied.stdout.strip(),
                    "observed_qdisc": observed.stdout.strip(),
                    "restore_error": removed.stderr.strip() if removed.returncode else None,
                }
            )
        else:
            details = self.application.application_fault(fault)
            if details is None:
                raise BenchmarkError(f"unsupported fault type: {fault_type}")
            result.update(details)
        result["finished_at"] = iso(utc_now())
        result["interventions"] = list(self.interventions)
        result["injection_duration_seconds"] = (utc_now() - started).total_seconds()
        return result

    def remove_network_fault_sidecar(self, deployment: str) -> None:
        response = self.backend.kubectl(
            "get", f"deployment/{deployment}", "-n", self.backend.namespace, "-o", "json", check=False
        )
        if response.returncode:
            return
        containers = (
            json.loads(response.stdout).get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        )
        index = next(
            (index for index, item in enumerate(containers) if item.get("name") == NETWORK_FAULT_SIDECAR), None
        )
        if index is not None:
            self.backend.kubectl(
                "patch",
                f"deployment/{deployment}",
                "-n",
                self.backend.namespace,
                "--type=json",
                "-p",
                json.dumps([{"op": "remove", "path": f"/spec/template/spec/containers/{index}"}]),
                check=False,
            )

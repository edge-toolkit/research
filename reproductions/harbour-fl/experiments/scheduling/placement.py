"""Optional placement policy and backend for the scheduling experiments."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any, Protocol


class OrchestrationError(RuntimeError):
    """An observation, policy, or runtime failure."""


@dataclass(frozen=True)
class NodeState:
    name: str
    ready: bool
    schedulable: bool
    labels: dict[str, str]
    allocatable_cpu_millicores: float
    allocatable_memory_mib: float
    requested_cpu_millicores: float
    requested_memory_mib: float

    @property
    def free_cpu_millicores(self) -> float:
        return self.allocatable_cpu_millicores - self.requested_cpu_millicores

    @property
    def free_memory_mib(self) -> float:
        return self.allocatable_memory_mib - self.requested_memory_mib


@dataclass(frozen=True)
class WorkloadState:
    workload_id: str
    pod_name: str
    node_name: str
    ready: bool
    restart_count: int


@dataclass(frozen=True)
class PlacementDecision:
    workload_id: str
    action: str
    source_node: str
    target_node: str
    reason: str
    candidate_nodes: tuple[str, ...]


class RuntimeAdapter(Protocol):
    def nodes(self) -> list[NodeState]: ...
    def workload(self, workload_id: str) -> WorkloadState: ...
    def placement(self, workload_id: str) -> dict[str, str] | None: ...
    def relocate(self, workload_id: str, target_node: str, timeout_seconds: int) -> WorkloadState: ...
    def restart(self, workload_id: str, timeout_seconds: int) -> WorkloadState: ...
    def restore_placement(self, workload_id: str, placement: dict[str, str] | None, timeout_seconds: int) -> None: ...


class LeastAllocatedPolicy:
    """Choose an eligible alternate node, preferring free memory then CPU."""

    def __init__(self, required_labels: dict[str, str], allow_same_node_restart: bool) -> None:
        self.required_labels = required_labels
        self.allow_same_node_restart = allow_same_node_restart

    def decide(self, workload: WorkloadState, nodes: list[NodeState]) -> PlacementDecision:
        """Choose an eligible alternate node by available capacity, allowing same-node restart only when configured."""
        eligible = [
            node
            for node in nodes
            if node.ready
            and node.schedulable
            and all(node.labels.get(key) == value for key, value in self.required_labels.items())
        ]
        alternate = [node for node in eligible if node.name != workload.node_name]
        if alternate:
            target = max(alternate, key=lambda node: (node.free_memory_mib, node.free_cpu_millicores, node.name))
            return PlacementDecision(
                workload.workload_id,
                "relocate",
                workload.node_name,
                target.name,
                "eligible alternate node has the highest estimated free capacity",
                tuple(sorted(node.name for node in eligible)),
            )
        if self.allow_same_node_restart and any(node.name == workload.node_name for node in eligible):
            return PlacementDecision(
                workload.workload_id,
                "restart",
                workload.node_name,
                workload.node_name,
                "no eligible alternate node; applying configured same-node recovery fallback",
                tuple(sorted(node.name for node in eligible)),
            )
        raise OrchestrationError("no eligible target node and same-node recovery is disabled")


def parse_cpu(value: str) -> float:
    """Convert a Kubernetes CPU quantity into millicores for placement comparisons."""
    if value.endswith("n"):
        return float(value[:-1]) / 1_000_000
    if value.endswith("u"):
        return float(value[:-1]) / 1_000
    if value.endswith("m"):
        return float(value[:-1])
    return float(value) * 1000


def parse_memory(value: str) -> float:
    """Convert supported Kubernetes memory quantities into MiB for placement comparisons."""
    units = {
        "Ki": 2**10,
        "Mi": 2**20,
        "Gi": 2**30,
        "Ti": 2**40,
        "Pi": 2**50,
        "Ei": 2**60,
        "k": 10**3,
        "M": 10**6,
        "G": 10**9,
        "T": 10**12,
        "P": 10**15,
        "E": 10**18,
    }
    for suffix, multiplier in units.items():
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * multiplier / (1024 * 1024)
    return float(value) / (1024 * 1024)


class KubernetesRuntime:
    """Restricted Kubernetes adapter; all mutations are deployment-scoped."""

    def __init__(self, context: str, namespace: str) -> None:
        self.context = context
        self.namespace = namespace

    def _run(
        self, *arguments: str, check: bool = True, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Execute context-scoped kubectl with captured output and raise on checked command failures."""
        command = ("kubectl", "--context", self.context, *arguments)
        result = subprocess.run(command, check=False, text=True, capture_output=True, timeout=timeout)
        if check and result.returncode:
            raise OrchestrationError(result.stderr.strip() or result.stdout.strip() or "kubectl failed")
        return result

    def _json(self, *arguments: str) -> dict[str, Any]:
        return json.loads(self._run(*arguments, "-o", "json").stdout)

    def nodes(self) -> list[NodeState]:
        """Combine node readiness and allocatable resources with active pod requests for placement decisions."""
        node_items = self._json("get", "nodes").get("items", [])
        pods = self._json("get", "pods", "--all-namespaces").get("items", [])
        requests: dict[str, tuple[float, float]] = {}
        for pod in pods:
            if pod.get("status", {}).get("phase") in {"Succeeded", "Failed"}:
                continue
            node_name = pod.get("spec", {}).get("nodeName")
            if not node_name:
                continue
            cpu, memory = requests.get(node_name, (0.0, 0.0))
            for container in pod.get("spec", {}).get("containers", []):
                requested = container.get("resources", {}).get("requests", {})
                cpu += parse_cpu(str(requested.get("cpu", "0")))
                memory += parse_memory(str(requested.get("memory", "0")))
            requests[node_name] = (cpu, memory)
        result = []
        for node in node_items:
            metadata = node.get("metadata", {})
            spec = node.get("spec", {})
            status = node.get("status", {})
            conditions = {item.get("type"): item.get("status") for item in status.get("conditions", [])}
            cpu, memory = requests.get(metadata["name"], (0.0, 0.0))
            allocatable = status.get("allocatable", {})
            result.append(
                NodeState(
                    name=metadata["name"],
                    ready=conditions.get("Ready") == "True",
                    schedulable=not bool(spec.get("unschedulable")),
                    labels={str(key): str(value) for key, value in metadata.get("labels", {}).items()},
                    allocatable_cpu_millicores=parse_cpu(str(allocatable.get("cpu", "0"))),
                    allocatable_memory_mib=parse_memory(str(allocatable.get("memory", "0"))),
                    requested_cpu_millicores=cpu,
                    requested_memory_mib=memory,
                )
            )
        return result

    def workload(self, workload_id: str) -> WorkloadState:
        """Describe the newest active workload pod, rejecting workloads without an active instance."""
        items = self._json("get", "pods", "-n", self.namespace, "-l", f"app={workload_id}").get("items", [])
        active = [item for item in items if item.get("metadata", {}).get("deletionTimestamp") is None]
        if not active:
            raise OrchestrationError(f"no active pod found for {workload_id}")
        pod = max(active, key=lambda item: item.get("metadata", {}).get("creationTimestamp", ""))
        statuses = pod.get("status", {}).get("containerStatuses", [])
        return WorkloadState(
            workload_id=workload_id,
            pod_name=pod["metadata"]["name"],
            node_name=pod.get("spec", {}).get("nodeName", ""),
            ready=bool(statuses) and all(item.get("ready", False) for item in statuses),
            restart_count=sum(int(item.get("restartCount", 0)) for item in statuses),
        )

    def placement(self, workload_id: str) -> dict[str, str] | None:
        """Return a copy of the current deployment node selector, or None when no selector is set."""
        deployment = self._json("get", f"deployment/{workload_id}", "-n", self.namespace)
        selector = deployment.get("spec", {}).get("template", {}).get("spec", {}).get("nodeSelector")
        return dict(selector) if selector else None

    def _rollout(self, workload_id: str, timeout_seconds: int) -> WorkloadState:
        """Wait for a bounded rollout and require the replacement pod to report ready."""
        self._run(
            "rollout",
            "status",
            f"deployment/{workload_id}",
            "-n",
            self.namespace,
            f"--timeout={timeout_seconds}s",
            timeout=timeout_seconds + 10,
        )
        state = self.workload(workload_id)
        if not state.ready:
            raise OrchestrationError(f"replacement pod for {workload_id} is not ready")
        return state

    def relocate(self, workload_id: str, target_node: str, timeout_seconds: int) -> WorkloadState:
        patch = {"spec": {"template": {"spec": {"nodeSelector": {"kubernetes.io/hostname": target_node}}}}}
        self._run("patch", f"deployment/{workload_id}", "-n", self.namespace, "--type=merge", "-p", json.dumps(patch))
        return self._rollout(workload_id, timeout_seconds)

    def restart(self, workload_id: str, timeout_seconds: int) -> WorkloadState:
        self._run("rollout", "restart", f"deployment/{workload_id}", "-n", self.namespace)
        return self._rollout(workload_id, timeout_seconds)

    def restore_placement(self, workload_id: str, placement: dict[str, str] | None, timeout_seconds: int) -> None:
        """Restore the original selector and wait for rollout, propagating failures from either command."""
        selector = None if placement is None else {"kubernetes.io/hostname": None, **placement}
        patch = {"spec": {"template": {"spec": {"nodeSelector": selector}}}}
        self._run(
            "patch",
            f"deployment/{workload_id}",
            "-n",
            self.namespace,
            "--type=merge",
            "-p",
            json.dumps(patch),
        )
        self._run(
            "rollout",
            "status",
            f"deployment/{workload_id}",
            "-n",
            self.namespace,
            f"--timeout={timeout_seconds}s",
            timeout=timeout_seconds + 10,
        )

    def logs(self, deployment: str, since_time: str) -> str:
        return self._run(
            "logs",
            f"deployment/{deployment}",
            "-n",
            self.namespace,
            "--timestamps=true",
            f"--since-time={since_time}",
            check=False,
        ).stdout

    def scale(self, deployment: str, replicas: int) -> None:
        self._run("scale", f"deployment/{deployment}", "-n", self.namespace, f"--replicas={replicas}")

    def set_env(self, deployment: str, values: list[str]) -> None:
        self._run("set", "env", f"deployment/{deployment}", "-n", self.namespace, *values)

    def wait_rollout(self, deployment: str, timeout_seconds: int) -> None:
        self._run(
            "rollout",
            "status",
            f"deployment/{deployment}",
            "-n",
            self.namespace,
            f"--timeout={timeout_seconds}s",
            timeout=timeout_seconds + 10,
        )

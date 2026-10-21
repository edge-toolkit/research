"""Resource sampling and summaries."""

from __future__ import annotations

import json
import statistics
import threading
from dataclasses import dataclass, field
from typing import Any

from et_experiments.utils import iso, parse_cpu, parse_memory, percentile, utc_now


@dataclass
class Sampler:
    backend: Any
    interval: float
    stop: threading.Event = field(default_factory=threading.Event)
    samples: list[dict[str, Any]] = field(default_factory=list)
    metrics_available: bool = True
    thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, name="et-benchmark-sampler", daemon=True)
        self.thread.start()

    def finish(self) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=self.interval + 5)

    def _loop(self) -> None:
        while not self.stop.is_set():
            sample: dict[str, Any] = {"at": iso(utc_now()), "pods": {}}
            pods = self.backend.kubectl("get", "pods", "-n", self.backend.namespace, "-o", "json", check=False)
            if pods.returncode == 0:
                for pod in json.loads(pods.stdout).get("items", []):
                    labels = pod.get("metadata", {}).get("labels", {})
                    app = labels.get("app")
                    if not app:
                        continue
                    statuses = pod.get("status", {}).get("containerStatuses", [])
                    sample["pods"][app] = {
                        "name": pod["metadata"]["name"],
                        "node": pod.get("spec", {}).get("nodeName"),
                        "phase": pod.get("status", {}).get("phase"),
                        "ready": all(item.get("ready", False) for item in statuses) if statuses else False,
                        "restarts": sum(int(item.get("restartCount", 0)) for item in statuses),
                    }
            top = self.backend.kubectl("top", "pods", "-n", self.backend.namespace, "--no-headers", check=False)
            metrics_observed = False
            if top.returncode == 0:
                for line in top.stdout.splitlines():
                    columns = line.split()
                    if len(columns) >= 3:
                        pod_name, cpu, memory = columns[:3]
                        for details in sample["pods"].values():
                            if details["name"] == pod_name:
                                details["cpu_millicores"] = parse_cpu(cpu)
                                details["memory_mib"] = parse_memory(memory)
                                metrics_observed = True

            # Stock k3d commonly has no Metrics Server. The kubelet summary is
            # an admin-only, read-only fallback and also provides cumulative
            # network counters that `kubectl top` does not expose.
            nodes = {details.get("node") for details in sample["pods"].values() if details.get("node")}
            for node in nodes:
                summary = self.backend.kubectl("get", "--raw", f"/api/v1/nodes/{node}/proxy/stats/summary", check=False)
                if summary.returncode:
                    continue
                for pod in json.loads(summary.stdout).get("pods", []):
                    reference = pod.get("podRef", {})
                    if reference.get("namespace") != self.backend.namespace:
                        continue
                    details = next(
                        (item for item in sample["pods"].values() if item["name"] == reference.get("name")),
                        None,
                    )
                    if details is None:
                        continue
                    containers = pod.get("containers", [])
                    cpu_nano = sum(int(item.get("cpu", {}).get("usageNanoCores") or 0) for item in containers)
                    memory_bytes = sum(int(item.get("memory", {}).get("workingSetBytes") or 0) for item in containers)
                    if "cpu_millicores" not in details and cpu_nano:
                        details["cpu_millicores"] = cpu_nano / 1_000_000
                    if "memory_mib" not in details and memory_bytes:
                        details["memory_mib"] = memory_bytes / (1024 * 1024)
                    network = pod.get("network", {})
                    if network:
                        details["network_rx_bytes"] = int(network.get("rxBytes") or 0)
                        details["network_tx_bytes"] = int(network.get("txBytes") or 0)
                    metrics_observed = metrics_observed or cpu_nano > 0 or memory_bytes > 0
            if not metrics_observed:
                self.metrics_available = False
            self.samples.append(sample)
            self.stop.wait(self.interval)


def resource_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def counter_increase(observations: list[dict[str, Any]], key: str) -> int | None:
        total = 0
        seen = False
        previous: dict[str, int] = {}
        for item in observations:
            if key not in item:
                continue
            seen = True
            pod_name = item["name"]
            value = int(item[key])
            if pod_name in previous:
                total += max(0, value - previous[pod_name])
            previous[pod_name] = value
        return total if seen else None

    result: dict[str, Any] = {}
    apps = sorted({app for sample in samples for app in sample["pods"]})
    for app in apps:
        observations = [sample["pods"][app] for sample in samples if app in sample["pods"]]
        cpu = [item["cpu_millicores"] for item in observations if "cpu_millicores" in item]
        memory = [item["memory_mib"] for item in observations if "memory_mib" in item]
        names = {item["name"] for item in observations}
        result[app] = {
            "pod_instances": len(names),
            "pod_replacements": max(0, len(names) - 1),
            "peak_restarts": max((item["restarts"] for item in observations), default=0),
            "not_ready_samples": sum(not item["ready"] for item in observations),
            "observed_availability_ratio": (
                sum(bool(item["ready"]) for item in observations) / len(observations) if observations else None
            ),
            "cpu_millicores_mean": statistics.fmean(cpu) if cpu else None,
            "cpu_millicores_p95": percentile(cpu, 0.95),
            "cpu_millicores_peak": max(cpu) if cpu else None,
            "memory_mib_mean": statistics.fmean(memory) if memory else None,
            "memory_mib_p95": percentile(memory, 0.95),
            "memory_mib_peak": max(memory) if memory else None,
            "network_rx_bytes": counter_increase(observations, "network_rx_bytes"),
            "network_tx_bytes": counter_increase(observations, "network_tx_bytes"),
        }
    return result

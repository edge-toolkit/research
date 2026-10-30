"""Temporarily block an allowlisted worker's placement while preserving its desired replica count."""

import hashlib
import json
import time

from et_experiments.utils import BenchmarkError, iso, utc_now

SELECTOR = "benchmark.edge-toolkit.io/unavailable"
SELECTOR_PATH = "/spec/template/spec/nodeSelector/benchmark.edge-toolkit.io~1unavailable"


class SchedulingFaults:
    """Track temporary selectors so normal fault teardown and final cleanup can both restore placement."""

    def __init__(self, backend, workers):
        self.backend, self.workers = backend, set(workers)
        self.pending = {}

    def deployment(self, target):
        """Read placement and desired state without changing the workload."""
        result = self.backend.kubectl("get", f"deployment/{target}", "-n", self.backend.namespace, "-o", "json")
        return json.loads(result.stdout)

    def restore(self):
        """Remove only selectors owned by this injector, preserving concurrent unrelated placement settings."""
        for target, token in list(self.pending.items()):
            selectors = self.deployment(target)["spec"]["template"]["spec"].get("nodeSelector", {})
            if SELECTOR not in selectors:
                del self.pending[target]
                continue
            if selectors[SELECTOR] != token:
                raise BenchmarkError(f"temporary placement selector ownership changed for {target}")
            self.backend.kubectl(
                "patch",
                f"deployment/{target}",
                "-n",
                self.backend.namespace,
                "--type=json",
                "-p",
                json.dumps(
                    [
                        {"op": "test", "path": SELECTOR_PATH, "value": token},
                        {"op": "remove", "path": SELECTOR_PATH},
                    ]
                ),
            )
            del self.pending[target]

    def inject(self, target, seconds, job_id):
        """Hold an unsatisfiable placement constraint, with desired replicas fixed at one, then release it."""
        if target not in self.workers:
            raise BenchmarkError(f"scheduling outages are not allowed for {target}")
        original = self.deployment(target)
        if original["spec"].get("replicas", 1) != 1:
            raise BenchmarkError("scheduling outage requires exactly one desired replica")
        selectors = original["spec"]["template"]["spec"].get("nodeSelector", {})
        if SELECTOR in selectors or target in self.pending:
            raise BenchmarkError(f"a temporary placement constraint already exists for {target}")
        token = hashlib.sha256(job_id.encode()).hexdigest()[:16]
        nodes = self.backend.kubectl(
            "get",
            "nodes",
            "-l",
            f"{SELECTOR}={token}",
            "-o",
            "json",
        )
        if json.loads(nodes.stdout)["items"]:
            raise BenchmarkError("temporary placement selector unexpectedly matches a node")
        self.pending[target] = token
        result = {"mechanism": "unsatisfiable-node-selector", "desired_replicas": 1, "hold_seconds": seconds}
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
                        {"op": "test", "path": "/spec/replicas", "value": 1},
                        {
                            "op": "add",
                            "path": "/spec/template/spec/nodeSelector",
                            "value": {**selectors, SELECTOR: token},
                        },
                    ]
                ),
            )
            result["active_at"] = iso(utc_now())
            time.sleep(seconds)
        finally:
            self.restore()
            result["released_at"] = iso(utc_now())
        return {**result, "status": "restored"}

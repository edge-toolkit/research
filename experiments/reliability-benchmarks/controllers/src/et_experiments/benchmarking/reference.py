"""Record declared workload conditions and observed deployment resources for comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict

from et_experiments.utils import BenchmarkError


def fingerprint(value):
    """Hash sorted JSON conditions for comparison, rejecting non-finite numeric values."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def hardware_snapshot(application, backend):
    """Record assigned nodes, immutable image identities and available node capacity metadata."""
    pods = backend.kubectl("get", "pods", "-n", application.config.namespace, "-o", "json", check=False, timeout=15)
    if pods.returncode:
        raise BenchmarkError("pod placement unavailable")
    placement = {}
    for pod in json.loads(pods.stdout).get("items", []):
        workload = pod.get("metadata", {}).get("labels", {}).get("app")
        if workload in application.config.workloads and not pod.get("metadata", {}).get("deletionTimestamp"):
            placement[workload] = {
                "node": pod["spec"].get("nodeName"),
                "images": sorted(
                    s.get("imageID", "")
                    for s in pod.get("status", {}).get("containerStatuses", [])
                    if s["name"] != "benchmark-netem"
                ),
            }
    nodes = backend.kubectl("get", "nodes", "-o", "json", check=False, timeout=15)
    if nodes.returncode:
        raise BenchmarkError("node capacity unavailable")
    used = {p["node"] for p in placement.values()}
    capacities = {
        node["metadata"]["name"]: {
            "allocatable": node["status"].get("allocatable", {}),
            "system": node["status"].get("nodeInfo", {}),
        }
        for node in json.loads(nodes.stdout).get("items", [])
        if node["metadata"]["name"] in used
    }
    if set(placement) != set(application.config.workloads) or not used.issubset(capacities):
        raise BenchmarkError("incomplete workload placement")
    return {"placement": placement, "nodes": capacities}


def capture_reference(application, scenario, backend):
    """Keep public resource metadata while excluding environment values and credentials."""
    declared = dict(scenario.get("reference", {}))
    describe = getattr(application, "reference_conditions", None)
    if callable(describe):
        declared.update(describe(scenario))
    workloads = asdict(application.config)
    resources, missing = [], []
    if not declared:
        missing.append("declared-workload-conditions")
    try:
        response = backend.kubectl(
            "get",
            "deployments",
            *application.config.workloads,
            "-n",
            application.config.namespace,
            "-o",
            "json",
            check=False,
            timeout=15,
        )
        if response.returncode:
            raise BenchmarkError("deployment metadata unavailable")
        items = json.loads(response.stdout).get("items", [])
        for item in items:
            spec = item["spec"]["template"]["spec"]
            resources.append(
                {
                    "workload": item["metadata"]["name"],
                    "replicas": item["spec"].get("replicas", 1),
                    "node_selector": spec.get("nodeSelector", {}),
                    "containers": [
                        {"name": c["name"], "image": c["image"], "resources": c.get("resources", {})}
                        for c in spec["containers"]
                        if c["name"] != "benchmark-netem"
                    ],
                }
            )
        if {r["workload"] for r in resources} != set(application.config.workloads):
            missing.append("incomplete-deployment-resources")
    except (BenchmarkError, subprocess.TimeoutExpired, OSError, ValueError, KeyError, TypeError):
        missing.append("deployment-resources")
    hardware = None
    try:
        hardware = hardware_snapshot(application, backend)
    except (BenchmarkError, subprocess.TimeoutExpired, OSError, ValueError, KeyError, TypeError):
        missing.append("node-hardware-or-placement")
    missing.extend(declared.get("missing_conditions", []))
    conditions = {
        "declared": declared,
        "workload": workloads,
        "resources": sorted(resources, key=lambda x: x["workload"]),
        "hardware": hardware,
    }
    return {"schema_version": 1, "conditions": conditions, "fingerprint": fingerprint(conditions), "missing": missing}


def compare_baseline(report, baseline):
    """Compare completed runs with matching reference conditions and an actual fault-free baseline."""
    reference = report.get("reference", {})
    other = baseline.get("reference", {})
    valid = (
        reference.get("fingerprint") is not None
        and reference.get("fingerprint") == other.get("fingerprint")
        and not reference.get("missing")
        and not other.get("missing")
        and baseline.get("fault", {}).get("type") == "none"
        and baseline.get("status") == "passed"
        and report.get("attribution", {}).get("recovery_mode") == baseline.get("attribution", {}).get("recovery_mode")
    )
    result = {
        "status": "compared" if valid else "incompatible",
        "baseline_job_id": baseline.get("job_id"),
        "metrics": {},
    }
    if valid:
        for name, value in report.get("sli", {}).items():
            previous = baseline.get("sli", {}).get(name)
            if (
                type(value) in (int, float)
                and type(previous) in (int, float)
                and math.isfinite(value)
                and math.isfinite(previous)
            ):
                result["metrics"][name] = {
                    "baseline": previous,
                    "fault_run": value,
                    "delta": value - previous,
                    "ratio": value / previous if previous != 0 else None,
                }
    return result

"""Check schedulable requests and memory headroom before creating a scaling point."""

import json

from measurements import quantities


def request_totals(pods, node, namespace):
    """Count schedulable requests, including overlapping restartable init containers and pod overhead."""
    result = {"cpu": 0.0, "memory": 0.0}
    for pod in pods:
        if (
            pod["metadata"].get("namespace") == namespace
            or pod["spec"].get("nodeName") != node
            or pod.get("status", {}).get("phase") in ("Succeeded", "Failed")
        ):
            continue
        for resource in result:

            def requested(container, resource=resource):
                value = container.get("resources", {}).get("requests", {}).get(resource, "0")
                return quantities(value, resource == "memory")

            regular = sum(requested(c) for c in pod["spec"].get("containers", []))
            sidecars, initial = 0.0, 0.0
            for container in pod["spec"].get("initContainers", []):
                value = requested(container)
                if container.get("restartPolicy") == "Always":
                    sidecars += value
                    initial = max(initial, sidecars)
                else:
                    initial = max(initial, sidecars + value)
            overhead = quantities(pod["spec"].get("overhead", {}).get(resource, "0"), resource == "memory")
            pod_request = pod["spec"].get("resources", {}).get("requests", {}).get(resource)
            effective = (
                quantities(pod_request, resource == "memory")
                if pod_request is not None
                else max(regular + sidecars, initial)
            )
            result[resource] += effective + overhead
    return result


def check_capacity(cluster, settings, snapshot, count, reports):
    """Keep quotas unchanged; permit memory-limit overcommit only with twice measured 12-client peak headroom."""
    pods = cluster.get("pods", "--all-namespaces")["items"]
    existing = request_totals(pods, snapshot["node"], settings["namespace"])
    node_metrics = json.loads(cluster.call("get", "--raw", "/apis/metrics.k8s.io/v1beta1/nodes/" + snapshot["node"]))
    existing["memory"] = max(existing["memory"], quantities(node_metrics["usage"]["memory"], True))
    resources = settings["resources"]
    requested = {}
    for key in ("cpu", "memory"):
        requested[key] = sum(
            multiplier * quantities(resources[role]["requests"][key], key == "memory")
            for role, multiplier in (("client", count), ("coordinator", 1), ("hub", 1))
        )
        available = quantities(snapshot["allocatable"][key], key == "memory")
        if existing[key] + requested[key] > available * 0.85:
            raise RuntimeError(f"{count}-client {key} requests leave insufficient node headroom")
    memory = sum(
        multiplier * quantities(resources[role]["limits"]["memory"], True)
        for role, multiplier in (("client", count), ("coordinator", 1), ("hub", 1))
    )
    mode = "declared memory limits"
    reference = None
    limit = quantities(snapshot["allocatable"]["memory"], True) * 0.85
    if existing["memory"] + memory > limit:
        references = [
            (key, r)
            for key, r in reports.items()
            if not key.startswith("smoke-")
            and r["clients"] == 12
            and r.get("status") == "passed"
            and r.get("cleanup_verified")
        ]
        if not references:
            raise RuntimeError(
                "Memory quotas exceed conservative headroom; import or finish a verified 12-client point first"
            )
        memory = 0
        for role, multiplier in (("client", count), ("coordinator", 1), ("hub", 1)):
            peaks = [
                values["peak_memory_mib"]
                for _, report in references
                for name, values in report["summary"]["resources"]["roles"].items()
                if (role == "client" and name.startswith("vessel-") and name != "vessel-coordinator")
                or (role == "coordinator" and name == "vessel-coordinator")
                or (role == "hub" and name == "ws-server")
            ]
            if not peaks:
                raise RuntimeError("Missing measured memory reference for " + role)
            per_role = max(2 * max(peaks), quantities(resources[role]["requests"]["memory"], True))
            memory += multiplier * per_role
        if existing["memory"] + memory > limit:
            raise RuntimeError("Twice measured peak memory leaves insufficient node headroom")
        mode = "twice worst measured 12-client peak or request, whichever is larger"
        reference = [key for key, _ in references]
    result = {
        "clients": count,
        "existing_requests": existing,
        "study_requests": requested,
        "study_memory_mib_for_admission": memory,
        "memory_basis": mode,
        "reference_jobs": reference,
        "node_allocatable": snapshot["allocatable"],
        "node_headroom_fraction": 0.15,
        "node_metrics": node_metrics,
        "note": "Per-role limits unchanged. CPU quotas can exceed host cores; 24 clients may contend for CPU.",
    }
    print(f"Capacity check passed for {count} clients: {mode}", flush=True)
    return result

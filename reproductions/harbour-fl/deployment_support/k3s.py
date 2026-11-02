"""Harbour-FL deployment support: k3s."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import DATASET_PATH, ROOT_MODULE, RUNNER_IMAGE, detector_environment
from .mlflow import mlflow_documents

REGISTRY_INITIALIZER = """
import ipaddress
import pathlib
import sys
import yaml

path = pathlib.Path(sys.argv[1])
original = path.read_bytes() if path.exists() else None
records = yaml.safe_load(original) if original is not None else {}
if records is None:
    records = {}
if not isinstance(records, dict):
    raise ValueError('agent registry must be a mapping')
migrated = False
for agent_id, record in records.items():
    if not isinstance(record, dict):
        raise ValueError(f'invalid registry record for {agent_id}')
    address = record.get('last_known_ip')
    if address is None:
        record['last_known_ip'] = '0.0.0.0'
        migrated = True
    elif not isinstance(address, str) or '%' in address:
        raise ValueError(f'invalid registry IP address for {agent_id}')
    else:
        ipaddress.ip_address(address)
for agent_id in sys.argv[2:]:
    records.setdefault(agent_id, {'state': 'disconnected', 'last_known_ip': '0.0.0.0'})
backup = path.with_name(path.name + '.pre-ip-required')
if migrated and original is not None and not backup.exists():
    backup.write_bytes(original)
temporary = path.with_name(path.name + '.tmp')
temporary.write_text(yaml.safe_dump(records))
temporary.replace(path)
"""


def environment(container: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a Kubernetes container's environment entries by name."""
    return {entry["name"]: entry for entry in container.get("env", [])}


def dataset_volume(document: dict[str, Any]) -> None:
    """Mount the node-local prepared dataset into one FL deployment."""
    pod = document["spec"]["template"]["spec"]
    container = pod["containers"][0]
    container.setdefault("volumeMounts", []).append(
        {"name": "vessel-data", "mountPath": DATASET_PATH, "readOnly": True}
    )
    environment(container)["HARBOUR_FL_DATA_ROOT"]["value"] = DATASET_PATH
    pod.setdefault("volumes", []).append(
        {"name": "vessel-data", "hostPath": {"path": DATASET_PATH, "type": "Directory"}}
    )


def configure_k3s(path: Path) -> None:
    """Add the demo's runner, dataset, root module, and MLflow service to generated k3s YAML."""
    import yaml

    documents = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    detector_settings = detector_environment(container=True)
    namespace = "et-harbour-fl"
    seen_root = False
    seen_supervisor = False
    for document in documents:
        if not isinstance(document, dict) or document.get("kind") != "Deployment":
            continue
        name = document["metadata"]["name"]
        namespace = document["metadata"].get("namespace", namespace)
        container = document["spec"]["template"]["spec"]["containers"][0]
        if name != "openobserve":
            document["spec"]["strategy"] = {"type": "Recreate"}
        if name == "ws-server":
            environment(container)["MODULES_ROOT"]["value"] = ROOT_MODULE
            container.setdefault("env", [])[:] = [
                entry for entry in container.get("env", []) if entry["name"] != "WS_AGENT_REGISTRY"
            ]
            container["env"].append({"name": "WS_AGENT_REGISTRY", "value": "/app/storage/registry.yaml"})
            document["spec"]["template"]["spec"]["initContainers"] = [
                {
                    "name": "agent-registry",
                    "image": RUNNER_IMAGE,
                    "imagePullPolicy": "IfNotPresent",
                    "command": [
                        "/opt/cpython/bin/python3",
                        "-c",
                        REGISTRY_INITIALIZER,
                        "/app/storage/registry.yaml",
                        "vessel-1",
                        "vessel-2",
                        "vessel-3",
                        "vessel-coordinator",
                        "mlflow-bridge",
                        "recovery-supervisor",
                    ],
                    "volumeMounts": [{"name": "ws-server-storage", "mountPath": "/app/storage"}],
                }
            ]
            seen_root = True
        if name in {"vessel-1", "vessel-2", "vessel-3", "vessel-coordinator"}:
            container["image"] = RUNNER_IMAGE
            for key, value in detector_settings.items():
                entries = container.setdefault("env", [])
                entries[:] = [entry for entry in entries if entry["name"] != key]
                container["env"].append({"name": key, "value": value})
            gpu_count = int(os.environ.get("HARBOUR_FL_K3S_GPUS_PER_CLIENT", "0"))
            if gpu_count < 0:
                raise ValueError("HARBOUR_FL_K3S_GPUS_PER_CLIENT cannot be negative")
            gpu_clients = {
                value.strip()
                for value in os.environ.get("HARBOUR_FL_K3S_GPU_CLIENTS", "vessel-1,vessel-2,vessel-3").split(",")
                if value.strip()
            }
            if not gpu_clients.issubset({"vessel-1", "vessel-2", "vessel-3"}):
                raise ValueError("HARBOUR_FL_K3S_GPU_CLIENTS must contain only vessel-1, vessel-2, vessel-3")
            if gpu_count and name in gpu_clients:
                resources = container.setdefault("resources", {})
                for section in ("requests", "limits"):
                    resources.setdefault(section, {})["nvidia.com/gpu"] = gpu_count
            dataset_volume(document)
            container.setdefault("env", []).append(
                {
                    "name": "HARBOUR_FL_READINESS_FILE",
                    "value": "/tmp/harbour-fl-ready",
                }
            )
            container["readinessProbe"] = {
                "exec": {
                    "command": [
                        "/opt/cpython/bin/python3",
                        "-c",
                        "import sys; from pathlib import Path; sys.exit(not Path('/tmp/harbour-fl-ready').is_file())",
                    ]
                },
                "initialDelaySeconds": 1,
                "periodSeconds": 2,
                "timeoutSeconds": 2,
                "failureThreshold": 1,
            }
        if name == "mlflow-bridge":
            container["image"] = RUNNER_IMAGE
            environment(container)["MLFLOW_TRACKING_URI"]["value"] = "http://mlflow:5000"
        if name == "recovery-supervisor":
            container["image"] = "et-recovery-supervisor:local"
            seen_supervisor = True
    if not seen_root:
        raise RuntimeError(f"generated deployment {path} has no ws-server MODULES_ROOT")
    if not seen_supervisor:
        raise RuntimeError("Core did not generate recovery-supervisor; use the matching recovery-capable Core release")
    documents.extend(mlflow_documents(namespace))
    path.write_text(yaml.safe_dump_all(documents, sort_keys=False), encoding="utf-8")

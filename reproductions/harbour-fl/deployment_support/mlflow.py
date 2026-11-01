"""Harbour-FL deployment support: mlflow."""

from __future__ import annotations

from typing import Any

from . import RUNNER_IMAGE


def mlflow_documents(namespace: str) -> list[dict[str, Any]]:
    """Return the local demonstration's MLflow service and deployment."""
    labels = {"app": "mlflow"}
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "mlflow", "namespace": namespace, "labels": labels},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [
                        {
                            "name": "mlflow",
                            "image": RUNNER_IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "command": [
                                "/opt/cpython/bin/python3",
                                "-m",
                                "mlflow",
                                "server",
                                "--host",
                                "0.0.0.0",
                                "--port",
                                "5000",
                                "--backend-store-uri",
                                "sqlite:////data/mlflow.db",
                                "--artifacts-destination",
                                "/data/artifacts",
                                "--workers",
                                "1",
                                "--allowed-hosts",
                                "mlflow:5000,mlflow,127.0.0.1:*,localhost:*",
                            ],
                            "ports": [{"containerPort": 5000, "name": "http"}],
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 5000},
                                "periodSeconds": 5,
                                "failureThreshold": 30,
                            },
                            "volumeMounts": [{"name": "data", "mountPath": "/data"}],
                        }
                    ],
                    "volumes": [{"name": "data", "persistentVolumeClaim": {"claimName": "mlflow-data"}}],
                },
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "mlflow", "namespace": namespace, "labels": labels},
        "spec": {"selector": labels, "ports": [{"name": "http", "port": 5000, "targetPort": 5000}]},
    }
    claim = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": "mlflow-data", "namespace": namespace},
        "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": "2Gi"}}},
    }
    return [claim, deployment, service]

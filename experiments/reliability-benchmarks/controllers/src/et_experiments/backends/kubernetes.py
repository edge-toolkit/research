"""Kubernetes commands and workload observations."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from typing import Any

from et_experiments.utils import iso, progress, run


class KubernetesBackend:
    def __init__(self, context: str, namespace: str, primary: str, observed: tuple[str, ...]):
        self.context, self.namespace = context, namespace
        self.primary, self.observed = primary, observed

    def kubectl(self, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return run("kubectl", "--context", self.context, *args, **kwargs)

    def wait_rollouts(
        self,
        deployments: tuple[str, ...],
        timeout: int = 300,
        *,
        started: dt.datetime | None = None,
    ) -> None:
        for deployment in deployments:
            if started:
                progress(started, f"Waiting for deployment/{deployment}")
            self.kubectl("rollout", "status", f"deployment/{deployment}", "-n", self.namespace, f"--timeout={timeout}s")

    def primary_logs(self, since: dt.datetime) -> str:
        completed = self.kubectl(
            "logs",
            f"deployment/{self.primary}",
            "-n",
            self.namespace,
            "--timestamps=true",
            f"--since-time={iso(since)}",
            check=False,
        )
        return completed.stdout

    def collect_logs(self, since: dt.datetime) -> dict[str, str]:
        result: dict[str, str] = {}
        for deployment in self.observed:
            current = self.kubectl(
                "logs",
                f"deployment/{deployment}",
                "-n",
                self.namespace,
                "--timestamps=true",
                f"--since-time={iso(since)}",
                check=False,
            )
            previous = self.kubectl(
                "logs",
                f"deployment/{deployment}",
                "-n",
                self.namespace,
                "--timestamps=true",
                f"--since-time={iso(since)}",
                "--previous",
                check=False,
            )
            result[deployment] = previous.stdout + current.stdout
        return result

    def warning_events(self, since: dt.datetime) -> list[dict[str, Any]]:
        response = self.kubectl("get", "events", "-n", self.namespace, "-o", "json", check=False)
        if response.returncode:
            return []
        events = []
        for event in json.loads(response.stdout).get("items", []):
            timestamp = event.get("eventTime") or event.get("lastTimestamp")
            if timestamp:
                parsed = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if parsed < since:
                    continue
            if event.get("type") == "Warning":
                events.append(
                    {
                        "reason": event.get("reason"),
                        "message": event.get("message"),
                        "object": event.get("involvedObject", {}).get("name"),
                        "at": timestamp,
                    }
                )
        return events

"""Application instrumentation for Core supervision, with separate research diagnostic logs."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid


def finite_values(value):
    """Keep diagnostics strict JSON even when a measured metric is non-finite."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_values(item) for item in value]
    return value


class Events:
    """Emit application diagnostics using the experiment log wire format, without importing its controller."""

    def __init__(self, job_id, workload):
        self.job_id = job_id
        self.workload = workload
        self.instance = uuid.uuid4().hex

    def emit(self, kind, **fields):
        event = {
            "schema_version": 1,
            "id": uuid.uuid4().hex,
            "instance": self.instance,
            "at": time.time(),
            "job_id": self.job_id,
            "workload": self.workload,
            "kind": kind,
            **fields,
        }
        event = finite_values(event)
        logging.getLogger("et.resilience").info("ET_EVENT %s", json.dumps(event, allow_nan=False))
        return event


class ApplicationRecovery(Events):
    """Delegate checkpoint integrity and lifecycle transport to the installed Edge Toolkit SDK."""

    def __init__(self, send, job_id: str, workload: str, supervisor: str = "", source: str = ""):
        """Load the Core checkpoint SDK and create lifecycle delivery only when a supervisor is configured."""
        super().__init__(job_id, workload)
        try:
            from edge_toolkit.recovery import Checkpoints, Lifecycle
        except ImportError as exc:
            raise RuntimeError(
                "Core recovery SDK is missing; install the matching release's Python wheels before running Harbour-FL"
            ) from exc
        self.checkpoint_class = Checkpoints
        self.lifecycle = Lifecycle(send, supervisor, source, job_id) if supervisor else None
        self.started = False
        self.progress = 0

    def emit(self, kind, progress=None, **fields):
        """Keep experiment-specific events in logs; send only Core lifecycle kinds to the supervisor."""
        if progress is not None:
            fields["progress"] = progress
        if "bytes_count" in fields:
            fields["bytes"] = fields["bytes_count"]
        if "duration_millis" in fields:
            fields["seconds"] = fields["duration_millis"] / 1000
        event = super().emit(kind, **fields)
        action = kind
        if (kind == "coordinator-status" and fields.get("status") == "run-started") or (
            kind == "client-status" and fields.get("status") == "training"
        ):
            action = "started"
        elif kind in {"client-status", "coordinator-status"} and fields.get("status") == "failed":
            action = "failed"
        if action not in {
            "started",
            "progress",
            "failed",
            "completed",
            "checkpoint-saved",
            "checkpoint-restored",
            "resume-completed",
        }:
            return event
        if progress is not None:
            self.progress = max(self.progress, progress)
        if self.lifecycle is not None:
            try:
                if not self.started:
                    self.lifecycle.emit("started", 0)
                    self.started = True
                if action != "started":
                    self.lifecycle.emit(
                        action,
                        self.progress,
                        bytes_count=int(fields.get("bytes_count", fields.get("bytes", 0))),
                        duration_millis=int(fields.get("duration_millis", fields.get("seconds", 0) * 1000)),
                    )
            except (OSError, RuntimeError, ValueError):
                logging.getLogger(__name__).exception("Core recovery lifecycle delivery failed")
        return event

    def checkpoints(self, storage, agent_id):
        """Use a bounded job-specific key and Core's agent-scoped, integrity-checked pointer format."""
        return self.checkpoint_class(storage, agent_id, checkpoint_key(self.job_id), lifecycle=self)


def checkpoint_key(job_id: str) -> str:
    """Keep long benchmark names inside the Core SDK's 64-character checkpoint-key limit."""
    return "fl-" + hashlib.sha256(job_id.encode()).hexdigest()[:32]

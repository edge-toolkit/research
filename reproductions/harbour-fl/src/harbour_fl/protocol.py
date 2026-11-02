"""Browser status and hub message helpers for one runtime instance."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .settings import CLIENT_STATUS_TYPE, COORDINATOR_STATUS_TYPE, SCHEMA_VERSION, STATUS_TYPE


class ProtocolMixin:
    def _mark_failed(self) -> None:
        """Withdraw readiness before publishing a terminal application failure."""
        self.state["application_failed"] = True
        self.state["application_ready"] = False
        filename = os.environ.get("HARBOUR_FL_READINESS_FILE", "")
        if filename:
            try:
                Path(filename).unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).exception("Unable to remove the failed application's readiness file")

    def _mark_ready(self) -> None:
        """Application state, not just the runner process, is usable by the deployment."""
        filename = os.environ.get("HARBOUR_FL_READINESS_FILE", "")
        if filename:
            Path(filename).write_text("ready\n")
        self.state["application_ready"] = True
        if self.state.get("service_checks"):
            from .service_checks import start_network_checks

            start_network_checks(self)

    def _handle_command(self, message: dict[str, Any]) -> None:
        """Accept one browser start command for this configured job."""
        if message.get("action") == "status":
            self._handle_status_request(message)
            return
        if message.get("action") != "start":
            return
        job_id = os.environ.get("HARBOUR_FL_JOB_ID", "harbour-fl-demo")
        if message.get("job_id") != job_id:
            return
        with self.state["start_lock"]:
            if self.state["started"]:
                status = "already-started"
            else:
                self.state["started"] = True
                self.state["start"].set()
                status = "accepted"
        self._send_status(job_id, status)

    def _send_status(self, job_id: str, status: str, **fields) -> None:
        self._send_json(
            {"type": STATUS_TYPE, "schema_version": SCHEMA_VERSION, "job_id": job_id, "status": status, **fields}
        )

    def _handle_status_request(self, message: dict[str, Any]) -> None:
        """Replay current role status to a browser that connected after startup."""
        if message.get("action") != "status":
            return
        if message.get("job_id") != os.environ.get("HARBOUR_FL_JOB_ID", "harbour-fl-demo"):
            return
        if self.state["role"] == "client":
            if message.get("client_id") != self.state["client_id"]:
                return
            status = self.state.get("client_status")
        else:
            if message.get("role") != "coordinator":
                return
            status = self.state.get("coordinator_status")
        if status:
            self._send_json(status)

    def _send_client_status(self, status: str, **fields) -> None:
        self._event("client-status", status=status, **fields)
        value = {
            "type": CLIENT_STATUS_TYPE,
            "schema_version": SCHEMA_VERSION,
            "job_id": os.environ.get("HARBOUR_FL_JOB_ID", "harbour-fl-demo"),
            "client_id": self.state["client_id"],
            "status": status,
            **fields,
        }
        self.state["client_status"] = value
        self._send_json(value)

    def _send_coordinator_status(self, status: str, **fields) -> None:
        self._event("coordinator-status", status=status, **fields)
        value = {
            "type": COORDINATOR_STATUS_TYPE,
            "schema_version": SCHEMA_VERSION,
            "job_id": os.environ.get("HARBOUR_FL_JOB_ID", "harbour-fl-demo"),
            "status": status,
            **fields,
        }
        self.state["coordinator_status"] = value
        self._send_json(value)

    def _event(self, kind: str, **fields) -> None:
        """Forward a diagnostic event when application recovery instrumentation is available."""
        if self.state.get("events") is not None:
            self.state["events"].emit(kind, **fields)

    def _send_json(self, value: dict[str, Any]) -> None:
        self.state["send"].text(json.dumps(value, separators=(",", ":"), sort_keys=True))

"""Runner hooks and ownership of per-process application state."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from pathlib import Path

from .client import ClientRole
from .coordinator import CoordinatorRole
from .instrumentation import ApplicationRecovery
from .protocol import ProtocolMixin
from .service_checks import TaskChecks
from .settings import COMMAND_TYPE, COMPLETE_TYPE, DEFAULT_CLIENTS, ROUND_TYPE, SCHEMA_VERSION, UPDATE_TYPE
from .tracking import TrackingMixin

_logger = logging.getLogger("harbour_fl")


class VesselRuntime(ClientRole, CoordinatorRole, TrackingMixin, ProtocolMixin):
    def __init__(self):
        self.state = {
            "send": None,
            "storage": None,
            "agent_id": None,
            "role": None,
            "client_id": None,
            "stop": threading.Event(),
            "start": threading.Event(),
            "start_lock": threading.Lock(),
            "started": False,
            "completed": False,
            "client_status": None,
            "coordinator_status": None,
            "worker": None,
            "seen_rounds": set(),
            "responses": {},
            "update_queue": queue.Queue(),
            "events": None,
        }

    def init(self, send, storage) -> None:
        """Capture runner handles and freeze this process's role."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        _logger.setLevel(logging.INFO)
        logging.getLogger("et.resilience").setLevel(logging.INFO)
        readiness = os.environ.get("HARBOUR_FL_READINESS_FILE", "")
        if readiness:
            Path(readiness).unlink(missing_ok=True)
        role = os.environ.get("HARBOUR_FL_ROLE", "client").strip().lower()
        if role not in {"client", "coordinator"}:
            raise RuntimeError("HARBOUR_FL_ROLE must be 'client' or 'coordinator'")
        client_id = os.environ.get("HARBOUR_FL_CLIENT_ID", "").strip()
        if role == "client" and client_id not in DEFAULT_CLIENTS:
            raise RuntimeError(f"HARBOUR_FL_CLIENT_ID must be one of {DEFAULT_CLIENTS!r}")
        self.state["send"] = send
        self.state["storage"] = storage
        self.state["role"] = role
        self.state["client_id"] = client_id or None
        self.state["application_ready"] = False
        self.state["service_checks"] = TaskChecks(self) if os.environ.get("HARBOUR_FL_PROBES_ENABLED") == "1" else None
        _logger.info("harbour FL module initialised with immutable role=%s client_id=%s", role, client_id or "-")

    def on_connect(self, agent_id: str) -> None:
        """Start the role-specific background worker after hub registration."""
        self.state["agent_id"] = agent_id
        if self.state.get("gpu_failed"):
            return  # A hub reconnect must not reuse a failed accelerator context; Core replaces the process.
        if self.state["completed"]:
            self._event("reconnected")
            return
        existing = self.state.get("worker")
        if existing is not None and existing.is_alive():
            self._event("reconnected")
            return
        job_id = os.environ.get("HARBOUR_FL_JOB_ID", "harbour-fl-demo")
        if os.environ.get("RECOVERY_JOB_ID", job_id) != job_id:
            raise RuntimeError("HARBOUR_FL_JOB_ID and Core RECOVERY_JOB_ID must match")
        self.state["events"] = ApplicationRecovery(
            self.state["send"].text,
            job_id,
            agent_id,
            os.environ.get("RECOVERY_SUPERVISOR_AGENT_ID", ""),
            "/modules/@edge-toolkit/harbour-fl-engine",
        )
        self.state["application_failed"] = False
        worker = threading.Thread(target=self._worker_main, name=f"harbour-fl-{self.state['role']}", daemon=True)
        self.state["worker"] = worker
        worker.start()
        _logger.info("harbour FL %s connected as %s", self.state["role"], agent_id)

    def _worker_main(self) -> None:
        """Report initialization failures too; Core, not an application restart loop, decides how to recover."""
        try:
            target = self._coordinator_main if self.state["role"] == "coordinator" else self._client_main
            target()
        except Exception:  # Final worker boundary: record initialization failure for Core supervision.
            self._mark_failed()
            self._event("failed")
            _logger.exception("harbour FL worker initialization failed")

    def on_text_frame(self, text: str) -> None:
        """Route protocol messages; long-running work remains off the dispatch thread."""
        try:
            message = json.loads(text)
        except (TypeError, ValueError):
            return
        if not isinstance(message, dict):
            return
        if message.get("type") == "et-agent-message" and self.state.get("service_checks"):
            if self.state["service_checks"].handle(message) and message.get("message_id"):
                self._send_json({"type": "et-message-ack", "message_id": message["message_id"]})
            return
        if message.get("schema_version") != SCHEMA_VERSION:
            return
        if self.state["role"] == "client":
            if message.get("type") == "harbour-fl-contribution-accepted":
                round_number = message.get("round")
                if (
                    message.get("job_id") == os.environ.get("HARBOUR_FL_JOB_ID", "harbour-fl-demo")
                    and message.get("client_id") == self.state["client_id"]
                    and (type(round_number) is int)
                    and (round_number in self.state["responses"])
                ):
                    self._event("progress", progress=round_number + 1)
            elif message.get("type") == COMPLETE_TYPE:
                if message.get("job_id") == os.environ.get("HARBOUR_FL_JOB_ID", "harbour-fl-demo"):
                    self._event("completed")
            elif message.get("type") == ROUND_TYPE:
                if self.state.get("gpu_failed"):
                    return
                round_number = message.get("round")
                if not isinstance(round_number, int) or round_number < 0:
                    return
                if round_number in self.state["seen_rounds"]:
                    cached = self.state["responses"].get(round_number)
                    if cached and cached["job_id"] == message.get("job_id"):
                        self._send_json(cached)
                    return
                self.state["seen_rounds"].add(round_number)
                self.state["update_queue"].put(message)
            elif message.get("type") == COMMAND_TYPE:
                self._handle_status_request(message)
        elif self.state["role"] == "coordinator":
            if message.get("type") == UPDATE_TYPE:
                self.state["update_queue"].put(message)
            elif message.get("type") == COMMAND_TYPE:
                self._handle_command(message)

    def on_shutdown(self) -> None:
        """Stop background loops and give their current operation a brief grace period."""
        self.state["stop"].set()
        self.state["start"].set()
        self.state["update_queue"].put(None)
        worker = self.state.get("worker")
        if worker is not None:
            worker.join(timeout=10.0)
        probe = self.state.get("network_check_thread")
        if probe is not None:
            probe.join(timeout=float(os.environ.get("HARBOUR_FL_PROBE_TIMEOUT_SECONDS", "5")) + 1)

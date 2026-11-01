"""Opt-in task canaries and client-to-hub RTT checks; neither advances training or recovery progress."""

from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid

VERSION = 1
REQUEST = "harbour-fl-probe-request"
TASK = "harbour-fl-probe-task"
WORKER_RESULT = "harbour-fl-probe-worker-result"
RESULT = "harbour-fl-probe-result"
TYPES = {REQUEST, TASK, WORKER_RESULT, RESULT}
WORKERS = {"vessel-1", "vessel-2", "vessel-3"}


def observer_id(job_id):
    return "probe-" + hashlib.sha256(job_id.encode()).hexdigest()[:20]


class TaskChecks:
    """Route a bounded arithmetic task through coordinator dispatch and the worker's application handler."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.job_id = os.environ.get("HARBOUR_FL_JOB_ID", "harbour-fl-demo")
        self.observer = os.environ["HARBOUR_FL_PROBE_OBSERVER_ID"]
        self.timeout = float(os.environ.get("HARBOUR_FL_PROBE_TIMEOUT_SECONDS", "5"))
        self.pending = {}

    def send(self, destination, kind, **fields):
        self.runtime._send_json(
            {
                "type": "et-send-agent-message",
                "to_agent_id": destination,
                "message": {"type": kind, "schema_version": 1, "job_id": self.job_id, **fields},
            }
        )

    def ready(self):
        """Require usable application state without terminal failure, shutdown or a failed GPU context."""
        state = self.runtime.state
        return bool(
            state.get("application_ready")
            and not state.get("application_failed")
            and not state["stop"].is_set()
            and not state.get("gpu_failed")
        )

    def handle(self, envelope):
        """Use hub-supplied sender identity, fixed roles and this job's observer; never accept arbitrary work."""
        message = envelope.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("type"), str) or message["type"] not in TYPES:
            return False
        if message.get("schema_version") != 1 or message.get("job_id") != self.job_id:
            return True
        identifier = message.get("probe_id")
        if (
            not isinstance(identifier, str)
            or len(identifier) != 32
            or any(c not in "0123456789abcdef" for c in identifier)
        ):
            return True
        sender, kind = envelope.get("from_agent_id"), message["type"]
        state = self.runtime.state
        if state["role"] == "coordinator":
            now = time.monotonic()
            self.pending = {key: item for key, item in self.pending.items() if item["expires"] > now}
            if kind == REQUEST and sender == self.observer:
                target = message.get("target")
                if not isinstance(target, str) or target not in WORKERS:
                    return True
                if not self.ready() or len(self.pending) >= 512:
                    self.send(self.observer, RESULT, probe_id=identifier, target=target, status="unavailable")
                elif identifier not in self.pending:
                    self.pending[identifier] = {"target": target, "expires": now + self.timeout}
                    self.send(target, TASK, probe_id=identifier, operands=[17, 25])
            elif kind == WORKER_RESULT and identifier in self.pending:
                pending = self.pending[identifier]
                if sender != pending["target"]:
                    return True
                del self.pending[identifier]
                result = message.get("result")
                success = message.get("status") == "success" and type(result) is int and result == 42
                self.send(
                    self.observer,
                    RESULT,
                    probe_id=identifier,
                    target=sender,
                    status="success" if success else "failure",
                    result=result,
                    worker_instance=message.get("worker_instance"),
                )
        elif kind == TASK and sender == "vessel-coordinator":
            operands = message.get("operands")
            if (
                not isinstance(operands, list)
                or len(operands) != 2
                or any(type(n) is not int or abs(n) > 1000 for n in operands)
            ):
                return True
            ready = self.ready()
            self.send(
                sender,
                WORKER_RESULT,
                probe_id=identifier,
                status="success" if ready else "unavailable",
                result=sum(operands) if ready else None,
                worker_instance=getattr(state.get("events"), "instance", None),
            )
        return True


def rtt_once(connection, payload, timeout):
    """Measure a correlated WebSocket Ping/Pong exchange using one endpoint's monotonic clock."""
    import websocket

    started = time.monotonic()
    connection.ping(payload)
    while True:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("RTT response deadline exceeded")
        connection.settimeout(remaining)
        opcode, data = connection.recv_data(control_frame=True)
        if opcode == websocket.ABNF.OPCODE_PONG and data == payload:
            return (time.monotonic() - started) * 1000
        if opcode == websocket.ABNF.OPCODE_CLOSE:
            raise ConnectionError("hub closed the RTT connection")


def start_network_checks(runtime):
    """Use a separate client-side socket so the RTT path excludes kubectl and the external observer."""
    if runtime.state["role"] != "client" or runtime.state.get("network_check_thread"):
        return
    interval = float(os.environ.get("HARBOUR_FL_PROBE_INTERVAL_SECONDS", "2"))
    timeout = float(os.environ.get("HARBOUR_FL_PROBE_TIMEOUT_SECONDS", "5"))
    payload_size = int(os.environ.get("HARBOUR_FL_PROBE_PAYLOAD_BYTES", "32"))
    url = os.environ.get("WS_SERVER_URL", "")

    def loop():
        """Emit correlated client-to-hub RTT outcomes, reconnecting after socket failures until shutdown."""
        connection = None
        sequence = 0
        try:
            import websocket

            while not runtime.state["stop"].is_set():
                began = time.monotonic()
                sequence += 1
                result = {
                    "sequence": sequence,
                    "payload_bytes": payload_size,
                    "scope": "client-to-hub WebSocket Ping/Pong; connection setup excluded",
                }
                try:
                    if connection is None:
                        connection = websocket.create_connection(url, timeout=timeout)
                    payload = (uuid.uuid4().bytes * 8)[:payload_size]
                    result.update(status="success", rtt_ms=rtt_once(connection, payload, timeout))
                except (TimeoutError, websocket.WebSocketTimeoutException):
                    result.update(status="timeout")
                    if connection is not None:
                        connection.close(timeout=0)
                    connection = None
                except (OSError, websocket.WebSocketException) as error:
                    result.update(status="failure", error_type=type(error).__name__)
                    if connection is not None:
                        connection.close(timeout=0)
                    connection = None
                runtime._event("communication-probe", **result)
                # One outstanding RTT check per client; a timed-out check can extend the spacing.
                if runtime.state["stop"].wait(max(0, interval - (time.monotonic() - began))):
                    break
        except Exception as error:  # noqa: BLE001 - thread boundary retains unknown observer failures
            runtime._event("communication-probe-observer-error", error_type=type(error).__name__)
        finally:
            if connection is not None:
                connection.close(timeout=0)

    thread = threading.Thread(target=loop, name="client-hub-rtt", daemon=True)
    runtime.state["network_check_thread"] = thread
    thread.start()

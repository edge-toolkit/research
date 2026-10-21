"""A separate observer connection for application task checks through the existing Core hub protocol."""

from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time

from et_experiments.benchmarking.probes import ObserverUnavailable, ServiceUnavailable
from et_experiments.utils import BenchmarkError


class HubTaskProbeTransport:
    """Keep observer failures distinguishable from a task path that cannot provide its expected result."""

    def __init__(self, backend, job_id, observer, coordinator, request_type, result_type):
        self.backend, self.job_id, self.observer = backend, job_id, observer
        self.coordinator, self.request_type, self.result_type = coordinator, request_type, result_type
        self.forward = self.socket = None
        self.port = None

    def _port_forward(self):
        if self.forward is not None and self.forward.poll() is None:
            return
        self.close()
        self.forward = subprocess.Popen(
            [
                "kubectl",
                "--context",
                self.backend.context,
                "-n",
                self.backend.namespace,
                "port-forward",
                "--address",
                "127.0.0.1",
                "service/ws-server",
                ":8080",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        lines = queue.Queue()
        stream = self.forward.stdout

        def read_output():
            for line in stream:
                lines.put(line)

        threading.Thread(target=read_output, name="probe-port-forward", daemon=True).start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.forward.poll() is not None:
                raise ObserverUnavailable("probe port-forward exited")
            try:
                line = lines.get(timeout=min(0.2, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            match = re.search(r"Forwarding from 127\.0\.0\.1:(\d+)", line)
            if match:
                self.port = int(match[1])
                return
        self.close()
        raise ObserverUnavailable("probe port-forward did not become ready")

    def _connect(self):
        import websocket

        self._port_forward()
        if self.socket is not None:
            return
        try:
            self.socket = websocket.create_connection(
                f"ws://127.0.0.1:{self.port}/ws",
                timeout=2,
                http_no_proxy=["127.0.0.1", "localhost"],
            )
            self.socket.send(json.dumps({"type": "et-connect", "agent_id": self.observer}))
            reply = json.loads(self.socket.recv())
            if (
                not isinstance(reply, dict)
                or reply.get("type") != "et-connect-ack"
                or reply.get("agent_id") != self.observer
            ):
                raise ServiceUnavailable("observer registration failed")
        except (OSError, ValueError, websocket.WebSocketException, ServiceUnavailable) as error:
            self._close_socket()
            raise ServiceUnavailable("hub connection could not be established") from error

    def start(self):
        try:
            self._connect()
        except (ObserverUnavailable, ServiceUnavailable) as error:
            self.close()
            raise BenchmarkError("Could not establish the functional-probe observer before measurement") from error

    def send(self, identifier, target):
        import websocket

        self._connect()
        try:
            self.socket.send(
                json.dumps(
                    {
                        "type": "et-send-agent-message",
                        "to_agent_id": self.coordinator,
                        "message": {
                            "schema_version": 1,
                            "type": self.request_type,
                            "job_id": self.job_id,
                            "probe_id": identifier,
                            "target": target,
                        },
                    }
                )
            )
        except (OSError, websocket.WebSocketException) as error:
            self._close_socket()
            raise ServiceUnavailable("could not submit functional check") from error

    def receive(self, timeout):
        import websocket

        if self.socket is None:
            time.sleep(timeout)
            return None
        try:
            self.socket.settimeout(timeout)
            text = self.socket.recv()
            if not text:
                raise ConnectionError("observer connection closed")
            envelope = json.loads(text)
        except websocket.WebSocketTimeoutException:
            return None
        except ValueError:
            return None
        except (OSError, websocket.WebSocketException) as error:
            self._connection_failed(error)
        if not isinstance(envelope, dict) or envelope.get("type") != "et-agent-message":
            return None
        message = envelope.get("message")
        if (
            envelope.get("from_agent_id") != self.coordinator
            or not isinstance(message, dict)
            or message.get("type") != self.result_type
            or message.get("job_id") != self.job_id
            or message.get("schema_version") != 1
        ):
            return None
        if envelope.get("message_id"):
            try:
                self.socket.send(json.dumps({"type": "et-message-ack", "message_id": envelope["message_id"]}))
            except (OSError, websocket.WebSocketException) as error:
                self._connection_failed(error)
        return message

    def _connection_failed(self, error):
        """Discard a failed connection and retain the distinction between observer and service failure."""
        self._close_socket()
        if self.forward is None or self.forward.poll() is not None:
            raise ObserverUnavailable("observer port-forward failed") from error
        raise ServiceUnavailable("hub connection closed") from error

    def _close_socket(self):
        connection, self.socket = self.socket, None
        if connection is not None:
            try:
                connection.close(timeout=0)
            except OSError:
                pass

    def close(self):
        self.port = None
        self._close_socket()
        process, self.forward = self.forward, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            if process.stdout:
                process.stdout.close()

"""Exercise browser-origin restrictions against the actual local HTTP server."""

import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from et_experiments.web.controller import BenchmarkController, RunState, handler_for


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.controller = Mock()
        self.controller.start.return_value = RunState("example", "baseline", {})
        self.controller.cancel.return_value = RunState("example", "baseline", {}, status="cancelling")
        self.controller.public_catalog.return_value = {"name": "example"}
        handler = handler_for(self.controller, b"example UI")
        self.quiet = patch.object(handler, "log_message")
        self.quiet.start()
        self.addCleanup(self.quiet.stop)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.addCleanup(self.server.server_close)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.shutdown)
        self.authority = f"127.0.0.1:{self.server.server_port}"

    def request(self, method, path, headers=None, body=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            payload = json.dumps({} if body is None else body) if method == "POST" else None
            connection.request(method, path, body=payload, headers=headers or {})
            response = connection.getresponse()
            return response.status, json.loads(response.read(), parse_constant=lambda token: self.fail(token))
        finally:
            connection.close()

    def test_negative_content_length_is_rejected_without_waiting_for_connection_close(self):
        status, value = self.request("POST", "/api/runs", {"Content-Type": "application/json", "Content-Length": "-1"})
        self.assertEqual(status, 400)
        self.assertIn("nonnegative", value["error"])
        self.controller.start.assert_not_called()

    def test_malformed_scenarios_and_numeric_values_return_json_errors_without_starting_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "adapter": {"command": ["python", "adapter.py"]},
                        "scenarios": {
                            "baseline": {
                                "parameters": {
                                    "rounds": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
                                    "pressure": {"type": "number", "default": 0.5, "minimum": 0, "maximum": 1},
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            controller = BenchmarkController(path)
            self.controller.start.side_effect = controller.start
            requests = [{"scenario": value} for value in ([], {}, None, False, 17, "unknown")]
            requests.extend(
                {"scenario": "baseline", "parameters": {name: value}}
                for name in ("rounds", "pressure")
                for value in (10**400, -(10**400), float("inf"), float("nan"))
            )
            for body in requests:
                with self.subTest(body=body):
                    status, value = self.request("POST", "/api/runs", {"Content-Type": "application/json"}, body)
                    self.assertEqual(status, 400)
                    self.assertIsInstance(value["error"], str)
            self.assertEqual(controller.runs, {})
            self.assertEqual(self.request("GET", "/api/catalog")[0], 200)

    def test_run_and_cancel_reject_foreign_origins_hosts_and_plain_text(self):
        for path in ("/api/runs", "/api/runs/example/cancel"):
            for headers, expected in (
                ({"Content-Type": "text/plain"}, 415),
                ({}, 415),
                ({"Content-Type": "application/json", "Origin": "https://foreign.example"}, 403),
                ({"Content-Type": "application/json", "Origin": "null"}, 403),
                ({"Content-Type": "application/json", "Host": f"foreign.example:{self.server.server_port}"}, 403),
            ):
                with self.subTest(path=path, headers=headers):
                    status, _ = self.request("POST", path, headers)
                    self.assertEqual(status, expected)
        self.controller.start.assert_not_called()
        self.controller.cancel.assert_not_called()

    def test_same_origin_json_and_operator_requests_are_accepted(self):
        for origin in (None, f"http://{self.authority}"):
            headers = {"Content-Type": "application/json; charset=utf-8"}
            if origin:
                headers["Origin"] = origin
            with self.subTest(origin=origin):
                self.assertEqual(self.request("POST", "/api/runs", headers)[0], 202)
                self.assertEqual(self.request("POST", "/api/runs/example/cancel", headers)[0], 200)
        self.assertEqual(self.controller.start.call_count, 2)
        self.assertEqual(self.controller.cancel.call_count, 2)

    def test_rebound_hosts_cannot_read_state(self):
        for path in ("/", "/api/catalog", "/api/runs/example"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path, {"Host": "foreign.example"})[0], 403)
        self.controller.get.assert_not_called()
        self.controller.public_catalog.assert_not_called()

    def test_localhost_alias_can_read_catalog(self):
        status, value = self.request("GET", "/api/catalog", {"Host": f"localhost:{self.server.server_port}"})
        self.assertEqual(status, 200)
        self.assertEqual(value["name"], "example")

    def test_run_state_returns_null_for_nonfinite_metrics(self):
        self.controller.get.return_value = RunState(
            "example",
            "baseline",
            {},
            status="passed",
            results=[{"status": "passed", "sli": {"loss": float("nan")}}],
            summary={"mttr_seconds": float("inf")},
        )
        status, value = self.request("GET", "/api/runs/example")
        self.assertEqual(status, 200)
        self.assertEqual(value["status"], "passed")
        self.assertIsNone(value["results"][0]["sli"]["loss"])
        self.assertIsNone(value["summary"]["mttr_seconds"])

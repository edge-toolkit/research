"""Exercise functional task routing independently of training and the deployed cluster."""

import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from harbour_fl.service_checks import REQUEST, RESULT, TASK, TaskChecks, observer_id, rtt_once


class TaskCheckTests(unittest.TestCase):
    def setUp(self):
        """Create coordinator and client probe handlers connected by an in-memory message route."""
        self.observer = observer_id("job")
        self.env = patch.dict(os.environ, {"HARBOUR_FL_JOB_ID": "job", "HARBOUR_FL_PROBE_OBSERVER_ID": self.observer})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.routes = []
        self.results = []
        self.runtimes = {}
        self.handlers = {}
        for name, role in (("vessel-coordinator", "coordinator"), ("vessel-1", "client")):
            runtime = SimpleNamespace(
                state={
                    "role": role,
                    "application_ready": True,
                    "worker": Mock(is_alive=Mock(return_value=True)),
                    "stop": threading.Event(),
                },
                _event=Mock(),
            )
            runtime._send_json = lambda message, source=name: self.routes.append((source, message))
            self.runtimes[name] = runtime
            self.handlers[name] = TaskChecks(runtime)

    def request(self, **overrides):
        return {
            "from_agent_id": self.observer,
            "message": {
                "type": REQUEST,
                "schema_version": 1,
                "job_id": "job",
                "probe_id": "a" * 32,
                "target": "vessel-1",
                **overrides,
            },
        }

    def drain(self):
        """Deliver queued probe messages between handlers and collect observer-facing results."""
        while self.routes:
            source, route = self.routes.pop(0)
            destination = route["to_agent_id"]
            if destination == self.observer:
                self.results.append(route["message"])
            else:
                self.handlers[destination].handle({"from_agent_id": source, "message": route["message"]})

    def test_full_path_verifies_result_without_training_progress(self):
        """Verify full path verifies result without training progress."""
        self.handlers["vessel-coordinator"].handle(self.request())
        self.drain()
        self.assertEqual(len(self.results), 1)
        self.assertEqual(self.results[0]["type"], RESULT)
        self.assertEqual(self.results[0]["status"], "success")
        self.assertEqual(self.results[0]["result"], 42)
        for runtime in self.runtimes.values():
            runtime._event.assert_not_called()

    def test_wrong_job_sender_and_malformed_target_are_ignored(self):
        """Verify wrong job sender and malformed target are ignored."""
        for envelope in (
            self.request(job_id="other"),
            self.request(target={}),
            {**self.request(), "from_agent_id": "foreign"},
        ):
            self.handlers["vessel-coordinator"].handle(envelope)
        self.assertEqual(self.routes, [])

    def test_unready_worker_is_an_unsuccessful_functional_check(self):
        self.runtimes["vessel-1"].state["application_ready"] = False
        self.handlers["vessel-coordinator"].handle(self.request())
        self.drain()
        self.assertEqual(self.results[0]["status"], "failure")

    def test_arbitrary_or_unbounded_work_is_rejected(self):
        """Verify arbitrary or unbounded work is rejected."""
        for operands in ([True, 1], [1001, 1], [1], "1+1"):
            self.handlers["vessel-1"].handle(
                {
                    "from_agent_id": "vessel-coordinator",
                    "message": {**self.request()["message"], "type": TASK, "operands": operands},
                }
            )
        self.assertEqual(self.routes, [])

    def test_expired_or_wrong_worker_result_does_not_verify_recovery(self):
        from harbour_fl.service_checks import WORKER_RESULT

        handler = self.handlers["vessel-coordinator"]
        handler.handle(self.request())
        self.routes.clear()
        response = {
            "from_agent_id": "vessel-2",
            "message": {**self.request()["message"], "type": WORKER_RESULT, "status": "success", "result": 42},
        }
        handler.handle(response)
        self.assertEqual(self.routes, [])
        handler.pending["a" * 32]["expires"] = 0
        handler.handle({**response, "from_agent_id": "vessel-1"})
        self.assertEqual(self.routes, [])

    def test_rtt_requires_matching_pong_and_close_is_failure(self):
        import websocket

        connection = Mock()
        connection.recv_data.side_effect = [(websocket.ABNF.OPCODE_PONG, b"old"), (websocket.ABNF.OPCODE_PONG, b"new")]
        self.assertGreaterEqual(rtt_once(connection, b"new", 1), 0)
        self.assertEqual(connection.recv_data.call_count, 2)
        connection.recv_data.side_effect = [(websocket.ABNF.OPCODE_CLOSE, b"")]
        with self.assertRaises(ConnectionError):
            rtt_once(connection, b"new", 1)


class AdapterProbeTests(unittest.TestCase):
    def test_adapter_exposes_availability_and_per_client_rtt_without_changing_recovery_metrics(self):
        import sys
        from pathlib import Path

        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[2] / "benchmarking"), *sys.path]):
            from integration import HarbourFLAdapter

        report = {
            "service_probes": {"settings": {"enabled": True}, "availability_ratio": 0.9, "observation_coverage": 1},
            "diagnostic_events": [
                {
                    "kind": "communication-probe",
                    "workload": "vessel-1",
                    "at": 1,
                    "status": "success",
                    "rtt_ms": 2,
                    "payload_bytes": 32,
                },
                {
                    "kind": "communication-probe",
                    "workload": "vessel-1",
                    "at": 2,
                    "status": "timeout",
                    "payload_bytes": 32,
                },
            ],
            "resilience": {"observed_availability_ratio": None},
        }
        with patch("integration.enrich_report", return_value={"model_accuracy": 0.5}):
            slis = HarbourFLAdapter().enrich_report(report)
        self.assertEqual(slis["functional_availability_ratio"], 0.9)
        self.assertEqual(slis["model_accuracy"], 0.5)
        self.assertEqual(report["communication_probes"]["vessel-1"]["success_ratio"], 0.5)
        self.assertEqual(report["communication_probes"]["vessel-1"]["rtt_ms"]["samples"], 1)
        self.assertIsNone(report["resilience"]["observed_availability_ratio"])
        self.assertEqual(report["communication_probes"]["vessel-2"]["status"], "not-observed")


if __name__ == "__main__":
    unittest.main()

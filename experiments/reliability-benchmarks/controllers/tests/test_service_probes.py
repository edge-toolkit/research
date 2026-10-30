"""Functional availability treats observer gaps, deadlines and worker quorum separately."""

import queue
import time
import unittest

from et_experiments.benchmarking.probes import (
    ObserverUnavailable,
    ServiceProbeMonitor,
    settings,
    summarize,
)
from et_experiments.utils import BenchmarkError


class AvailabilityTests(unittest.TestCase):
    def test_port_forward_timeout_discards_stale_port_and_allows_retry(self):
        import io
        from unittest.mock import Mock, patch

        from et_experiments.backends.hub_probes import HubTaskProbeTransport

        transport = HubTaskProbeTransport(Mock(), "job", "observer", "coordinator", "request", "result")
        transport.port = 1234
        transport.forward = Mock()
        transport.forward.poll.return_value = 1
        timed_out, ready = Mock(), Mock()
        timed_out.poll.return_value = ready.poll.return_value = None
        timed_out.stdout = io.StringIO("")
        ready.stdout = io.StringIO("Forwarding from 127.0.0.1:5678 -> 8080\n")
        with (
            patch("et_experiments.backends.hub_probes.subprocess.Popen", side_effect=[timed_out, ready]) as spawn,
            patch("et_experiments.backends.hub_probes.threading.Thread") as thread,
            patch("et_experiments.backends.hub_probes.time.monotonic", side_effect=[0, 11, 20, 20, 20]),
        ):
            thread.return_value.start.side_effect = lambda: thread.call_args.kwargs["target"]()
            with self.assertRaisesRegex(ObserverUnavailable, "did not become ready"):
                transport._port_forward()
            self.assertIsNone(transport.forward)
            self.assertIsNone(transport.port)
            timed_out.terminate.assert_called_once()
            timed_out.wait.assert_called_once_with(timeout=3)
            self.assertTrue(timed_out.stdout.closed)
            transport._port_forward()
            self.assertEqual(spawn.call_count, 2)
            self.assertIs(transport.forward, ready)
            self.assertEqual(transport.port, 5678)
        transport.close()
        self.assertIsNone(transport.port)

    def test_ack_send_failures_are_classified_and_close_the_socket(self):
        import json
        from unittest.mock import Mock

        import websocket
        from et_experiments.backends.hub_probes import HubTaskProbeTransport
        from et_experiments.benchmarking.probes import ServiceUnavailable

        message = {"type": "result", "schema_version": 1, "job_id": "job", "probe_id": "probe"}
        envelope = {"type": "et-agent-message", "from_agent_id": "coordinator", "message_id": "id", "message": message}
        for error in (
            OSError("socket closed"),
            websocket.WebSocketConnectionClosedException("hub closed"),
            websocket.WebSocketTimeoutException("acknowledgement timed out"),
        ):
            for forwarding in (True, False):
                with self.subTest(error=type(error).__name__, forwarding=forwarding):
                    transport = HubTaskProbeTransport(Mock(), "job", "observer", "coordinator", "request", "result")
                    transport.forward = Mock()
                    transport.forward.poll.return_value = None if forwarding else 1
                    connection = Mock()
                    connection.recv.return_value = json.dumps(envelope)
                    connection.send.side_effect = error
                    transport.socket = connection
                    expected = ServiceUnavailable if forwarding else ObserverUnavailable
                    with self.assertRaises(expected):
                        transport.receive(0.1)
                    self.assertIsNone(transport.socket)
                    connection.close.assert_called_once()
                    transport.socket = Mock()
                    transport.socket.recv.return_value = json.dumps(envelope)
                    self.assertEqual(transport.receive(0.1), message)

    def test_missing_observations_withhold_point_estimate(self):
        config = {"interval_seconds": 2, "required_workers": 2}
        records = [
            {"cycle": 0, "target": "a", "status": "success", "elapsed_ms": 1},
            {"cycle": 0, "target": "b", "status": "success", "elapsed_ms": 2},
            {"cycle": 1, "target": "a", "status": "timeout"},
            {"cycle": 1, "target": "b", "status": "success", "elapsed_ms": 1},
        ]
        result = summarize(records, ("a", "b"), config, 100, 105)
        self.assertIsNone(result["availability_ratio"])
        self.assertEqual(result["available_seconds"], 2)
        self.assertEqual(result["unavailable_seconds"], 2)
        self.assertEqual(result["unknown_seconds"], 1)
        self.assertEqual(result["availability_lower_bound"], 0.4)
        self.assertEqual(result["availability_upper_bound"], 0.6)
        self.assertEqual(result["by_worker"]["a"]["unknown"], 1)

    def test_sufficient_workers_make_quorum_available_even_with_one_unknown(self):
        result = summarize(
            [{"cycle": 0, "target": "a", "status": "success"}],
            ("a", "b"),
            {"interval_seconds": 2, "required_workers": 1},
            0,
            1,
        )
        self.assertEqual(result["availability_ratio"], 1)
        self.assertEqual(result["by_worker"]["b"]["observation_coverage"], 0)

    def test_opt_in_and_invalid_settings(self):
        self.assertEqual(settings({}, 3), {"enabled": False})
        self.assertEqual(settings({"service_probes_enabled": True}, 3)["required_workers"], 3)
        for value in ({"interval_seconds": float("nan")}, {"required_workers": 4}, {"rtt_payload_bytes": 126}):
            with self.assertRaises(BenchmarkError):
                settings({"service_probes": {"enabled": True, **value}}, 3)

    def test_independent_monitor_filters_duplicate_and_wrong_results(self):
        class Transport:
            def __init__(self):
                self.responses = queue.Queue()
                self.closed = False
                self.sent = 0

            def start(self):
                pass

            def send(self, identifier, target):
                self.sent += 1
                if target == "unknown":
                    raise ObserverUnavailable()
                if target == "silent":
                    return
                row = {
                    "probe_id": identifier,
                    "target": target,
                    "status": "success",
                    "result": 42 if target == "good" else 41,
                }
                self.responses.put(row)
                self.responses.put(row)

            def receive(self, timeout):
                try:
                    return self.responses.get(timeout=timeout)
                except queue.Empty:
                    return None

            def close(self):
                self.closed = True

        transport = Transport()
        monitor = ServiceProbeMonitor(
            transport,
            ("good", "bad", "silent", "unknown"),
            {"interval_seconds": 1, "timeout_seconds": 0.1, "required_workers": 4},
        )
        monitor.start()
        deadline = time.monotonic() + 2
        while transport.sent < 4 and time.monotonic() < deadline:
            time.sleep(0.005)
        result = monitor.finish()
        self.assertTrue(transport.closed)
        self.assertEqual(len(result["records"]), 4)
        self.assertEqual(
            {r["target"]: r["status"] for r in result["records"]},
            {"good": "success", "bad": "failure", "silent": "timeout", "unknown": "unknown"},
        )
        self.assertEqual(result["availability_ratio"], 0)
        self.assertEqual(monitor.finish()["window_finished_at"], result["window_finished_at"])


class EngineProbeTests(unittest.TestCase):
    def test_opt_in_monitor_stops_and_preserves_raw_outcomes_before_cleanup(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from subprocess import CompletedProcess
        from unittest.mock import Mock

        from et_experiments.benchmarking.engine import BenchmarkEngine
        from test_engine import ImageOnlyAdapter

        app = ImageOnlyAdapter()
        backend = Mock()
        backend.kubectl.return_value = CompletedProcess([], 0, stdout=app.config.context)
        backend.primary_logs.return_value = "image-only progress"
        backend.collect_logs.return_value = {}
        backend.warning_events.return_value = []
        probe = Mock()
        probe.finish.return_value = {
            "status": "measured",
            "records": [{"status": "success"}],
            "settings": {"enabled": True},
            "availability_ratio": 1,
        }
        app.service_probe_monitor = Mock(return_value=probe)
        engine = BenchmarkEngine(app, backend=backend)
        engine.sampler_factory = Mock(return_value=Mock(samples=[], metrics_available=False))
        engine.faults.remove_network_fault_sidecar = Mock()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            report = engine.execute(
                "baseline",
                {"batches": 1, "recovery_enabled": False, "service_probes": {"enabled": True}},
                Path(directory),
                2,
                False,
            )
            self.assertEqual(report["status"], "passed")
            raw = Path(directory) / report["service_probes"]["observations_file"]
            self.assertEqual(json.loads(raw.read_text())["records"], [{"status": "success"}])
        probe.start.assert_called_once()
        probe.finish.assert_called_once()
        self.assertTrue(app.restored)
        args = app.service_probe_monitor.call_args.args
        self.assertIs(args[0], backend)
        self.assertEqual(args[1], report["job_id"])
        self.assertIsInstance(args[2], dict)


class HubTransportTests(unittest.TestCase):
    def test_only_correlated_job_results_from_coordinator_are_accepted(self):
        import json
        from unittest.mock import Mock

        from et_experiments.backends.hub_probes import HubTaskProbeTransport

        transport = HubTaskProbeTransport(Mock(), "job", "observer", "coordinator", "request", "result")
        transport.socket = Mock()
        result = {"type": "result", "schema_version": 1, "job_id": "job", "probe_id": "a" * 32}
        transport.socket.recv.side_effect = [
            json.dumps({"type": "et-agent-message", "from_agent_id": "foreign", "message": result}),
            json.dumps(
                {
                    "type": "et-agent-message",
                    "from_agent_id": "coordinator",
                    "message": result,
                    "message_id": "delivery",
                }
            ),
        ]
        self.assertIsNone(transport.receive(0.1))
        self.assertEqual(transport.receive(0.1), result)
        self.assertEqual(
            json.loads(transport.socket.send.call_args.args[0]), {"type": "et-message-ack", "message_id": "delivery"}
        )


if __name__ == "__main__":
    unittest.main()

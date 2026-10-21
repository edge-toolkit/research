"""Test O2 delivery against an isolated HTTP ingestion fixture."""

import json
import os
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from et_experiments.tracking.o2 import O2Config, deliver, publish_report
from et_experiments.tracking.o2_dashboard import dashboard
from et_experiments.tracking.o2_records import records
from et_experiments.web.controller import BenchmarkController


@contextmanager
def server():
    """Provide real HTTP ingestion, partial acknowledgements and redirect responses."""
    received = []
    responses = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, dict(self.headers), payload))
            mode = responses.pop(0) if responses else "success"
            if mode == "redirect":
                self.send_response(302)
                self.send_header("Location", "/untrusted-target")
                self.end_headers()
                return
            self.send_response(503 if mode == "offline" else 200)
            self.end_headers()
            body = {
                "code": 200,
                "status": [{"name": "reliability_benchmarks", "successful": len(payload), "failed": 0}],
            }
            if mode == "partial":
                body["status"][0].update(successful=0, failed=len(payload))
            if mode == "malformed":
                body["status"] = ["invalid"]
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *_args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    environment = {
        "BENCHMARK_O2_URL": f"http://127.0.0.1:{http.server_port}",
        "BENCHMARK_O2_USER": "test-ingestion",
        "BENCHMARK_O2_PASSWORD": "test-only",
    }
    try:
        with patch.dict(os.environ, environment, clear=True):
            yield received, responses
    finally:
        http.shutdown()
        thread.join(timeout=5)
        http.server_close()


def report():
    """Provide recovered and unrecovered incidents with their recorded measurements."""
    return {
        "job_id": "job-1",
        "scenario": "dropout",
        "status": "passed",
        "finished_at": "2026-10-20T10:05:00Z",
        "fault": {"type": "dropout"},
        "resilience": {"mttr_seconds": 8, "not_measured": None, "invalid": float("nan")},
        "workload": {"units": [{"metrics": {"test_loss": 0.5}}]},
        "slos": {"mttr_seconds": {"status": "passed", "value": 8, "objective": {"max": 20}}},
        "attribution": {"recovery_mode": "controller-assisted", "recovery_actors": ["edge-toolkit-core"]},
        "recovery": {
            "provider": "edge-toolkit-core",
            "clock_basis": "observer-monotonic-seconds",
            "incidents": [
                {
                    "workload": "worker",
                    "status": "recovered",
                    "observed_failure_at": 1,
                    "detected_at": 3,
                    "recovered_at": 9,
                    "actions": [{"kind": "restart-requested"}],
                },
                {
                    "workload": "worker",
                    "status": "unrecovered",
                    "observed_failure_at": 10,
                    "detected_at": 12,
                    "recovered_at": None,
                },
            ],
        },
        "samples": [
            {
                "at": "2026-10-20T10:00:00Z",
                "pods": {
                    "worker": {"name": "pod-1", "ready": True, "cpu_millicores": 123, "memory_mib": 456},
                },
            }
        ],
        "events": [
            {"id": "event-1", "kind": "checkpoint-restored", "at": 8, "clock_basis": "observer-monotonic-seconds"}
        ],
        "resources": {"worker": {"observed_availability_ratio": 0.9}},
        "warning_events": [{"reason": "Unhealthy", "message": "private error details"}],
    }


class RecordTests(unittest.TestCase):
    def test_evidence_is_typed_finite_and_preserves_measurement_boundaries(self):
        values = records(report(), {"application": "image-classifier", "repetition": 2})
        kinds = {item["record_type"] for item in values}
        self.assertTrue(
            {
                "run",
                "metric",
                "slo",
                "incident",
                "action",
                "event",
                "resource_sample",
                "resource_metric",
                "unit_metric",
                "warning",
            }
            <= kinds
        )
        self.assertTrue(all(item["application"] == "image-classifier" for item in values))
        json.dumps(values, allow_nan=False)
        self.assertNotIn("private error details", json.dumps(values))
        metrics = [item for item in values if item["record_type"] == "metric"]
        self.assertNotIn("not_measured", {item["metric_name"] for item in metrics})
        incidents = [item for item in values if item["record_type"] == "incident"]
        self.assertEqual(incidents[0]["failure_detection_seconds"], 2)
        self.assertEqual(incidents[0]["recovery_seconds"], 8)
        self.assertIsNone(incidents[1]["recovery_seconds"])
        sample = next(item for item in values if item["record_type"] == "resource_sample")
        self.assertEqual(sample["_timestamp"], "2026-10-20T10:00:00Z")
        event = next(item for item in values if item["record_type"] == "event")
        self.assertEqual(event["_timestamp"], report()["finished_at"])
        self.assertEqual(event["observer_offset_seconds"], 8)

    def test_record_ids_are_stable_and_repetitions_remain_distinct(self):
        first = records(report(), {"repetition": 1})
        again = records(report(), {"repetition": 1})
        second = records(report(), {"repetition": 2})
        self.assertEqual(first, again)
        self.assertTrue({item["record_id"] for item in first}.isdisjoint(item["record_id"] for item in second))

    def test_missing_slo_and_minimal_adapter_are_supported(self):
        value = {"status": "failed", "slos": {"accuracy": {"status": "not-measured", "value": None}}}
        result = records(value, {"job_id": "external-1", "scenario": "baseline"})
        self.assertEqual(result[0]["job_id"], "external-1")
        self.assertEqual(result[1]["status"], "not-measured")
        self.assertIsNone(result[1]["metric_value"])

    def test_dashboard_covers_evidence_and_deduplicates_replays(self):
        value = dashboard("custom_stream")
        self.assertEqual(value["version"], 8)
        self.assertEqual(len(value["tabs"]), 4)
        panels = [panel for tab in value["tabs"] for panel in tab["panels"]]
        self.assertGreaterEqual(len(panels), 20)
        for panel in panels:
            query = panel["queries"][0]["query"]
            self.assertIn('"custom_stream"', query)
            self.assertIn("GROUP BY record_id", query)
            self.assertIn("record_data", query)
        with self.assertRaises(ValueError):
            dashboard('unsafe"stream')


class DeliveryTests(unittest.TestCase):
    def test_disabled_by_default_and_configuration_rejects_unsafe_endpoints(self):
        with patch.dict(os.environ, {}, clear=True), tempfile.TemporaryDirectory() as directory:
            self.assertEqual(publish_report(report(), Path(directory)), {"status": "disabled"})
            self.assertEqual(list(Path(directory).iterdir()), [])
        for url in ("ftp://host", "http://user:pass@host", "https://host?secret=1", "http://remote-host"):
            with (
                self.subTest(url=url),
                patch.dict(os.environ, {"BENCHMARK_O2_URL": url}, clear=True),
                self.assertRaises(ValueError),
            ):
                O2Config.from_environment()

    def test_success_and_replay_do_not_resend_acknowledged_batches(self):
        with server() as (received, _responses), tempfile.TemporaryDirectory() as directory:
            result = publish_report(report(), Path(directory))
            self.assertEqual(result["status"], "exported")
            self.assertEqual(len(received), 1)
            path, headers, payload = received[0]
            self.assertEqual(path, "/api/default/reliability_benchmarks/_json")
            self.assertTrue(headers["Authorization"].startswith("Basic "))
            self.assertNotIn(headers["Authorization"], repr(O2Config.from_environment()))
            self.assertEqual(len(payload), result["records"])
            deliver(Path(result["outbox"]), O2Config.from_environment())
            self.assertEqual(len(received), 1)
            files = "".join(item.read_text() for item in Path(directory).rglob("*.json"))
            self.assertNotIn("test-only", files)
            self.assertNotIn(headers["Authorization"], files)

    def test_partial_delivery_retains_outbox_and_resumes_after_last_acknowledged_batch(self):
        value = {**report(), "events": [{"id": str(i), "kind": "progress", "at": i} for i in range(230)]}
        with server() as (received, responses), tempfile.TemporaryDirectory() as directory:
            responses.extend(["success", "partial"])
            result = publish_report(value, Path(directory))
            self.assertEqual(result["status"], "unavailable")
            path = Path(result["outbox"])
            self.assertTrue(path.exists())
            self.assertEqual(len(received), 2)
            self.assertEqual(json.loads(path.with_suffix(".receipt.json").read_text())["sent"], 200)
            self.assertEqual(deliver(path, O2Config.from_environment())["status"], "exported")
            self.assertEqual(received[1][2], received[2][2])

    def test_http_failure_and_redirect_do_not_change_benchmark_status(self):
        for mode in ("offline", "redirect", "malformed"):
            with self.subTest(mode=mode), server() as (received, responses), tempfile.TemporaryDirectory() as directory:
                responses.append(mode)
                value = report()
                result = publish_report(value, Path(directory))
                self.assertEqual(result["status"], "unavailable")
                self.assertEqual(value["status"], "passed")
                self.assertEqual(len(received), 1)
                self.assertTrue(Path(result["outbox"]).exists())

    def test_generic_browser_adapter_exports_without_application_specific_integration(self):
        import sys

        catalog = Path(__file__).resolve().parents[2] / "examples/minimal-adapter/catalog.json"
        with server() as (received, _responses), tempfile.TemporaryDirectory() as directory:
            controller = BenchmarkController(catalog, Path(directory))
            controller.catalog["adapter"]["command"][0] = sys.executable
            state = controller.start({"scenario": "baseline"})
            deadline = time.monotonic() + 10
            while state.status in {"queued", "running"} and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(state.status, "passed", state.logs)
            self.assertEqual(state.results[0]["o2"]["status"], "exported")
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0][2][0]["application"], "Minimal module adapter example")
            self.assertNotIn("test-only", json.dumps(state.public()))

    def test_same_job_in_different_repetitions_has_separate_outboxes(self):
        with server() as (received, _responses), tempfile.TemporaryDirectory() as directory:
            first = publish_report(report(), Path(directory), {"repetition": 1})
            second = publish_report(report(), Path(directory), {"repetition": 2})
            self.assertNotEqual(first["outbox"], second["outbox"])
            self.assertEqual(len(received), 2)

    def test_generic_browser_adapter_keeps_success_when_o2_is_unavailable(self):
        import sys

        catalog = Path(__file__).resolve().parents[2] / "examples/minimal-adapter/catalog.json"
        with server() as (_received, responses), tempfile.TemporaryDirectory() as directory:
            responses.append("offline")
            controller = BenchmarkController(catalog, Path(directory))
            controller.catalog["adapter"]["command"][0] = sys.executable
            state = controller.start({"scenario": "baseline"})
            deadline = time.monotonic() + 10
            while state.status in {"queued", "running"} and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(state.status, "passed", state.logs)
            self.assertEqual(state.results[0]["o2"]["status"], "unavailable")
            self.assertTrue(Path(state.results[0]["o2"]["outbox"]).exists())

    def test_timeout_budget_and_invalid_credentials_are_bounded(self):
        with server(), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outbox.json"
            path.write_text(json.dumps(records(report())))
            with self.assertRaises(TimeoutError):
                deliver(path, O2Config.from_environment(), budget_seconds=0)
            with patch.dict(os.environ, {"BENCHMARK_O2_PASSWORD": ""}):
                self.assertEqual(publish_report(report(), Path(directory))["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()

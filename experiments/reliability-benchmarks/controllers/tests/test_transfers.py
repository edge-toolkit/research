"""Missing transfer outcomes must not produce successful throughput populations."""

import unittest

from et_experiments.benchmarking.transfers import transfer_summary


def event(kind, transfer_id, at, **fields):
    return {
        "kind": kind,
        "transfer_id": transfer_id,
        "at": at,
        "instance": "process",
        "workload": "client",
        "direction": "download",
        "progress": 0,
        **fields,
    }


class TransferTests(unittest.TestCase):
    def test_success_timeout_and_duplicates_have_explicit_denominators(self):
        events = [
            event("model-transfer-started", "one", 0),
            event("model-transfer-finished", "one", 1, status="success", seconds=0.5, payload_bytes=1048576),
            event("model-transfer-started", "two", 2),
            event("model-transfer-finished", "two", 3, status="timeout", seconds=1, payload_bytes=None),
        ]
        result = transfer_summary(events + events, ("client",))
        row = result["by_client"]["client"]["download"]
        self.assertEqual(row["attempts"], 2)
        self.assertEqual(row["success_ratio"], 0.5)
        self.assertEqual(row["timeouts"], 1)
        self.assertEqual(row["mib_per_second"]["mean"], 2)
        self.assertEqual(row["duration_seconds"]["samples"], 1)

    def test_process_replacement_and_orphan_outcomes_do_not_create_false_success(self):
        events = [
            event("model-transfer-started", "one", 0),
            event(
                "model-transfer-finished",
                "one",
                1,
                instance="replacement",
                status="success",
                seconds=1,
                payload_bytes=1,
            ),
        ]
        row = transfer_summary(events, ("client",))["by_client"]["client"]["download"]
        self.assertEqual(row["unknown"], 1)
        self.assertEqual(row["orphan_outcomes"], 1)
        self.assertIsNone(row["success_ratio"])
        self.assertEqual(row["mib_per_second"]["samples"], 0)

    def test_engine_waits_for_final_evaluation_and_fails_if_it_never_completes(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from subprocess import CompletedProcess
        from unittest.mock import Mock, patch

        from et_experiments.benchmarking.engine import BenchmarkEngine
        from et_experiments.utils import BenchmarkError
        from test_engine import ImageOnlyAdapter

        for completes in (True, False):
            app = ImageOnlyAdapter()
            app.configure_run = lambda _backend, _faults, job_id, _scenario, _started, app=app: setattr(
                app, "job", job_id
            )
            backend = Mock()
            backend.kubectl.return_value = CompletedProcess([], 0, stdout=app.config.context)
            backend.collect_logs.return_value = {}
            backend.warning_events.return_value = []
            counter = [0]

            def logs(_since, counter=counter, completes=completes, app=app):
                counter[0] += 1
                if completes and counter[0] >= 2:
                    return "ET_EVENT " + json.dumps(
                        {
                            "id": "finish",
                            "instance": "process",
                            "schema_version": 1,
                            "job_id": app.job,
                            "workload": "trainer",
                            "kind": "completed",
                            "progress": 1,
                            "at": 100,
                        }
                    )
                return ""

            backend.primary_logs.side_effect = logs
            engine = BenchmarkEngine(app, backend=backend)
            engine.sampler_factory = Mock(return_value=Mock(samples=[], metrics_available=False))
            engine.faults.remove_network_fault_sidecar = Mock()
            with (
                tempfile.TemporaryDirectory() as directory,
                contextlib.redirect_stdout(io.StringIO()),
                patch("et_experiments.benchmarking.engine.time.sleep"),
            ):
                scenario = {
                    "batches": 1,
                    "recovery_enabled": False,
                    "completion_event": "completed",
                    "timeout_seconds": 0.1,
                }
                if completes:
                    result = engine.execute("baseline", scenario, Path(directory), 2, False)
                    self.assertEqual(result["status"], "passed")
                    self.assertGreaterEqual(counter[0], 2)
                else:
                    with self.assertRaises(BenchmarkError):
                        engine.execute("baseline", scenario, Path(directory), 2, False)
                self.assertTrue(app.restored)

    def test_zero_duration_does_not_create_infinite_throughput(self):
        events = [
            event("model-transfer-started", "one", 0),
            event("model-transfer-finished", "one", 1, status="success", seconds=0, payload_bytes=1),
        ]
        row = transfer_summary(events, ("client",))["by_client"]["client"]["download"]
        self.assertEqual(row["successful"], 1)
        self.assertEqual(row["mib_per_second"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()

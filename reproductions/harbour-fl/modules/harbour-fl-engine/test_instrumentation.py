"""The application adapter delegates Core mechanisms and keeps experiment diagnostics separate."""

import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from harbour_fl.instrumentation import ApplicationRecovery, checkpoint_key


class ApplicationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.lifecycle = Mock()
        self.sdk = SimpleNamespace(Checkpoints=Mock(), Lifecycle=Mock(return_value=self.lifecycle))
        self.patch = patch.dict(sys.modules, {"edge_toolkit.recovery": self.sdk})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.events = ApplicationRecovery(Mock(), "job", "worker", "supervisor", "/modules/example")

    def test_idle_connection_and_research_diagnostics_do_not_activate_supervision(self):
        self.events.emit("dataset-loaded")
        self.events.emit("adaptation", progress=99)
        self.lifecycle.emit.assert_not_called()
        self.assertEqual(self.events.progress, 0)

    def test_started_precedes_progress_and_duplicate_start_does_not_reset_work(self):
        """Verify started precedes progress and duplicate start does not reset work."""
        self.events.emit("client-status", status="training")
        self.events.emit("progress", progress=2)
        self.events.emit("client-status", status="training")
        self.events.emit("completed")
        actions = self.lifecycle.emit.call_args_list
        self.assertEqual([call.args[0] for call in actions], ["started", "progress", "completed"])
        self.assertEqual(actions[-1].args[1], 2)

    def test_checkpoint_state_machine_is_delegated_to_core(self):
        storage = Mock()
        result = self.events.checkpoints(storage, "worker")
        self.assertIs(result, self.sdk.Checkpoints.return_value)
        self.sdk.Checkpoints.assert_called_once_with(storage, "worker", checkpoint_key("job"), lifecycle=self.events)
        self.assertLessEqual(len(checkpoint_key("bench-" + "a" * 100)), 64)
        self.assertNotEqual(checkpoint_key("one"), checkpoint_key("two"))

    def test_core_checkpoint_callback_is_logged_and_delivered(self):
        with self.assertLogs("et.resilience") as captured:
            event = self.events.emit("checkpoint-restored", 2, bytes_count=100, duration_millis=125)
        self.assertEqual(event["seconds"], 0.125)
        self.assertEqual(event["bytes"], 100)
        self.assertIn("ET_EVENT", captured.output[0])
        self.lifecycle.emit.assert_called_with("checkpoint-restored", 2, bytes_count=100, duration_millis=125)

    def test_nested_nonfinite_metrics_are_null_without_interrupting_event_delivery(self):
        metrics = {"train_loss": float("nan"), "nested": [float("inf"), {"negative": float("-inf"), "valid": 1.5}]}
        with self.assertLogs("et.resilience") as captured:
            event = self.events.emit("client-status", status="waiting", metrics=metrics)
        payload = captured.output[0].split("ET_EVENT ", 1)[1]
        decoded = json.loads(payload, parse_constant=lambda value: self.fail(f"non-JSON number: {value}"))
        expected = {"train_loss": None, "nested": [None, {"negative": None, "valid": 1.5}]}
        self.assertEqual(decoded["metrics"], expected)
        self.assertEqual(event["metrics"], expected)
        self.assertIsNotNone(metrics["train_loss"])

    def test_failed_transport_does_not_discard_successful_checkpoint_evidence(self):
        self.lifecycle.emit.side_effect = OSError("hub unavailable")
        with self.assertLogs("harbour_fl.instrumentation", level="ERROR"):
            event = self.events.emit("checkpoint-saved", 1, bytes_count=5)
        self.assertEqual(event["kind"], "checkpoint-saved")

    def test_unsupervised_local_run_still_uses_core_checkpoints(self):
        local = ApplicationRecovery(Mock(), "local-job", "worker")
        local.emit("progress", progress=1)
        self.assertIsNone(local.lifecycle)
        local.checkpoints(Mock(), "worker")
        self.assertEqual(self.sdk.Checkpoints.call_count, 1)
